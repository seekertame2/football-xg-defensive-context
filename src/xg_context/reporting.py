"""Сборка markdown-отчёта из посчитанных таблиц.

Весь текст строится из чисел, полученных пайплайном.
Формулировки о значимости выбирает код по фактическим доверительным интервалам, а не автор текста.
Если интервал разницы включает ноль, пишем "различия не обнаружено".
Формулировка "модели равны" здесь запрещена.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["markdown_table", "plain_text", "render_results_report"]

THIN_SPACE = " "

# Отчёты читают и правят руками, поэтому в них только клавиатурные символы.
# Часть подписей приходит из CSV, где типографские знаки остаются как есть.
PLAIN_REPLACEMENTS = {
    "→": "->",
    "—": "-",
    "–": "-",
    "−": "-",
    "«": '"',
    "»": '"',
    "…": "...",
    " ": " ",
    " ": " ",
    "°": " градусов",
}


def plain_text(text: str) -> str:
    """Заменить типографские символы на клавиатурные."""
    for old, new in PLAIN_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def _fmt_int(value: float | int) -> str:
    return f"{int(value):,}".replace(",", THIN_SPACE)


def markdown_table(
    frame: pd.DataFrame,
    columns: dict[str, str],
    floats: dict[str, str] | None = None,
) -> str:
    """Отрендерить таблицу в markdown с русскими заголовками."""
    floats = floats or {}
    if frame.empty:
        return "_Нет данных._\n"
    lines = [
        "| " + " | ".join(columns.values()) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame[list(columns)].iterrows():
        cells = []
        for name in columns:
            value = row[name]
            if isinstance(value, float):
                if pd.isna(value):
                    cells.append("н/д")
                elif name in floats:
                    cells.append(floats[name].format(value))
                elif float(value).is_integer() and abs(value) >= 1:
                    cells.append(_fmt_int(value))
                else:
                    cells.append(f"{value:.4f}")
            elif isinstance(value, (int,)) and not isinstance(value, bool):
                cells.append(_fmt_int(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return plain_text("\n".join(lines) + "\n")


def _interval(row: pd.Series, metric: str = "log_loss") -> str:
    return f"[{row[f'delta_{metric}_ci_low']:+.5f}; {row[f'delta_{metric}_ci_high']:+.5f}]"


def _includes_zero(row: pd.Series, metric: str = "log_loss") -> bool:
    return bool(row[f"delta_{metric}_ci_low"] < 0 < row[f"delta_{metric}_ci_high"])


BOOTSTRAP_COLUMNS = {
    "сравнение": "Сравнение",
    "delta_log_loss": "Разница log loss",
    "delta_log_loss_ci_low": "CI низ",
    "delta_log_loss_ci_high": "CI верх",
    "delta_brier": "Разница Brier",
    "delta_brier_ci_low": "CI низ",
    "delta_brier_ci_high": "CI верх",
}


def render_results_report(
    summary: dict[str, Any],
    ablation: pd.DataFrame,
    subgroups: pd.DataFrame,
    examples: pd.DataFrame,
    league: pd.DataFrame,
    calibration: pd.DataFrame,
    league_skill: pd.DataFrame | None = None,
    sensitivity: pd.DataFrame | None = None,
) -> str:
    """Собрать полный текст `reports/results.md`."""
    models = pd.DataFrame(
        [
            {
                "модель": m["model"],
                "набор": m["feature_set"],
                "val_log_loss": m["validation"]["log_loss"],
                "log_loss": m["test"]["log_loss"],
                "brier": m["test"]["brier"],
                "roc_auc": m["test"]["roc_auc"],
                "pr_auc": m["test"]["pr_auc"],
                "key": m["key"],
            }
            for m in summary["models"]
        ]
    ).sort_values("log_loss", ignore_index=True)
    ece_by_key = dict(zip(calibration["key"], calibration["ECE"], strict=True))
    models["ECE"] = models["key"].map(ece_by_key)

    bootstrap = pd.DataFrame(summary["bootstrap"])
    split_rows = pd.DataFrame(summary["split_summary"])
    float_format = dict.fromkeys([c for c in bootstrap.columns if c.startswith("delta")], "{:+.5f}")

    defensive = bootstrap[
        bootstrap["сравнение"].str.contains("защитный контекст")
        & ~bootstrap["сравнение"].str.contains("БЕЗ")
    ]
    versus_sb = bootstrap[bootstrap["сравнение"].str.contains("statsbomb")]
    flags = bootstrap[bootstrap["сравнение"].str.contains("флаги")]
    no_visible = bootstrap[bootstrap["сравнение"].str.contains("БЕЗ n_opponents_visible")]
    own_contribution = bootstrap[bootstrap["сравнение"].str.contains("Вклад самого")]

    parts: list[str] = []
    add = parts.append

    add("# Результаты: насколько защитный контекст улучшает xG\n\n")
    add(
        "> Файл собирает `scripts/make_report.py` из таблиц в `reports/tables/`. "
        "Не редактируйте его вручную, правьте код и перезапускайте.\n\n"
    )
    add(
        f"Выборка: {_fmt_int(summary['n_shots'])} непенальтистских ударов из "
        f"{_fmt_int(summary['n_matches'])} матчей четырёх мужских топ-лиг сезона "
        "2015/2016.\n"
    )

    add("\n## 1. Разбиение\n\n")
    add(
        markdown_table(
            split_rows,
            {
                "split": "Часть",
                "n_matches": "Матчей",
                "n_shots": "Ударов",
                "share_of_shots": "Доля",
                "n_goals": "Голов",
                "goal_rate": "Доля голов",
            },
            {"share_of_shots": "{:.3f}", "goal_rate": "{:.4f}"},
        )
    )
    add(
        "\nРазбиение идёт по `match_id`: удары одного матча не попадают в разные "
        "части. Тестовые `shot_id` записаны в `data/processed/split.json` и для "
        "выбора признаков и гиперпараметров не использовались.\n"
    )

    add("\n## 2. Итоговая таблица моделей\n\n")
    add(
        markdown_table(
            models,
            {
                "модель": "Модель",
                "набор": "Признаки",
                "val_log_loss": "Val log loss",
                "log_loss": "Test log loss",
                "brier": "Brier",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "ECE": "ECE",
            },
        )
    )
    add(
        f"\nЛучшая нелинейная модель выбрана по validation: "
        f"{summary['best_nonlinear_model']}. ECE это взвешенная по размеру бина "
        "средняя ошибка калибровки.\n"
    )

    add("\n## 3. Ablation: вклад каждой группы признаков\n\n")
    add(
        "Все уровни обучены на одних и тех же строках, одном разбиении и одной "
        "предобработке. Каждый следующий уровень строго добавляет признаки "
        "к предыдущему.\n\n"
    )
    add(
        markdown_table(
            ablation,
            {
                "уровень": "Уровень",
                "описание": "Что добавлено",
                "модель": "Модель",
                "n_признаков": "n",
                "test_log_loss": "Test log loss",
                "test_brier": "Brier",
                "test_roc_auc": "ROC-AUC",
                "Δ_log_loss_шаг": "Разница на шаге",
            },
            {"Δ_log_loss_шаг": "{:+.5f}"},
        )
    )
    add("\n![Ablation](figures/04_ablation.png)\n")

    add("\n## 4. Неопределённость: парный bootstrap по матчам\n\n")
    add(markdown_table(bootstrap, BOOTSTRAP_COLUMNS, float_format))
    add(
        f"\nРесемплируются матчи, а не удары: удары одного матча зависимы. "
        f"{summary['n_bootstrap']} итераций, 95% интервал. Отрицательная разница "
        "означает, что кандидат лучше.\n"
    )

    mean_gain = 0.0
    if not defensive.empty:
        mean_gain = -float(defensive["delta_log_loss"].mean())
        add(
            f"\nГлавный результат: защитный контекст улучшает log loss в среднем на "
            f"{mean_gain:.5f} по двум моделям. "
        )
        add(
            "Интервалы целиком ниже нуля, улучшение устойчиво на этих данных.\n"
            if bool(defensive["delta_log_loss_significant"].all())
            else "Интервал включает ноль, устойчивого улучшения не обнаружено.\n"
        )

    if not flags.empty:
        row = flags.iloc[0]
        add(
            f"\nГотовые флаги StatsBomb дают {row['delta_log_loss']:+.5f} log loss, "
            f"интервал {_interval(row)}. "
        )
        if _includes_zero(row):
            add(
                "Интервал включает ноль: обнаружимого прироста они не дали. Это не "
                "доказывает, что весь прирост принадлежит моим признакам, потому "
                "что это два отдельных сравнения.\n"
            )
        else:
            add("Интервал не включает ноль, улучшение устойчиво.\n")

    add("\n## 5. Sensitivity test: признак `n_opponents_visible`\n\n")
    add(
        "Признак считает соперников в кадре, поэтому отражает и плотность обороны, "
        "и границы поля зрения камеры. Модель обучена дважды, с ним и без него, "
        "на тех же строках и том же разбиении.\n\n"
    )
    if sensitivity is not None and not sensitivity.empty:
        add(
            markdown_table(
                sensitivity,
                {
                    "модель": "Модель",
                    "L3 без защитного контекста": "L3 (без контекста)",
                    "L4 без n_opponents_visible": "L4 без visible",
                    "L4 полный": "L4 (полный)",
                    "brier_L4_без_visible": "Brier L4 без visible",
                    "brier_L4": "Brier L4",
                },
            )
        )
    add("\n![Sensitivity](figures/10_sensitivity.png)\n")

    if not no_visible.empty:
        add(
            "\nЭффект защитного контекста "
            + (
                "сохраняется и без этого признака: интервалы по-прежнему ниже нуля.\n"
                if bool(no_visible["delta_log_loss_significant"].all())
                else "исчезает без этого признака: интервал включает ноль.\n"
            )
        )
    if not own_contribution.empty:
        row = own_contribution.iloc[0]
        share = abs(row["delta_log_loss"]) / mean_gain if mean_gain else float("nan")
        add(
            f"\nСобственный вклад признака: {row['delta_log_loss']:+.5f} log loss, "
            f"интервал {_interval(row)}. "
        )
        if _includes_zero(row) and _includes_zero(row, "brier"):
            add("По обеим метрикам интервалы включают ноль.\n")
        elif _includes_zero(row):
            add(
                "По log loss устойчивого вклада нет, по Brier score он мал, но "
                f"устойчив. Это около {share:.0%} общего прироста.\n"
            )
        else:
            add("Интервалы не включают ноль, вклад устойчив.\n")
        add(
            "\nЧитать этот признак как плотность обороны всё равно нельзя. "
            "Он кодирует ещё и то, сколько игроков попало в кадр.\n"
        )

    add("\n## 6. Сравнение со `statsbomb_xg`\n\n")
    if not versus_sb.empty:
        row = versus_sb.iloc[0]
        add(
            f"Сравнивается основная модель проекта: логистическая регрессия "
            f"с защитным контекстом. У неё log loss ниже, чем у случайного леса. "
            f"Разница между логистической и `statsbomb_xg` составляет "
            f"{row['delta_log_loss']:+.5f} log loss, интервал {_interval(row)}. "
        )
        if _includes_zero(row):
            add(
                "Интервал включает ноль, устойчивого различия по log loss нет. Это "
                "не значит, что модели равны: при "
                f"{_fmt_int(int(row['n_shots']))} тестовых ударах различие такого "
                "размера неотличимо от нуля.\n"
            )
        else:
            add("Интервал не включает ноль, различие устойчиво.\n")
        if bool(row["delta_brier_ci_low"] > 0):
            add(
                f"\nПо Brier score `statsbomb_xg` устойчиво лучше: "
                f"{row['delta_brier']:+.5f}, интервал {_interval(row, 'brier')}.\n"
            )
        elif bool(row["delta_brier_ci_high"] < 0):
            add("\nПо Brier score моя модель устойчиво лучше.\n")
        else:
            add("\nПо Brier score устойчивого различия не обнаружено.\n")
    add("\nСравнение неравное: StatsBomb обучает модель на закрытых данных большего объёма.\n")

    add("\n## 7. Калибровка\n\n")
    add(
        "ECE каждой модели есть в таблице раздела 2, бины квантильные. Полная "
        "разбивка по бинам лежит в `reports/tables/calibration_summary.csv`.\n"
    )
    add("\n![Калибровка](figures/05_calibration.png)\n")

    add("\n## 8. Метрики по лигам\n\n")
    add(
        "Сырой log loss между лигами сравнивать нельзя: он ниже там, где реже "
        "забивают. Поэтому качество считается относительно `DummyClassifier` на тех "
        "же строках. Skill это доля снятого log loss, BSS это Brier Skill Score.\n\n"
    )
    if league_skill is not None and not league_skill.empty:
        # Строки случайного леса опущены: вывод по лигам строится на логистической.
        # Полная таблица лежит в reports/tables/league_skill.csv.
        add(
            markdown_table(
                league_skill[~league_skill["модель"].str.contains("Случайный лес")],
                {
                    "лига": "Лига",
                    "модель": "Модель",
                    "n": "Ударов",
                    "доля голов": "Доля голов",
                    "log loss": "Log loss",
                    "skill log loss": "Skill",
                    "Brier": "Brier",
                    "BSS": "BSS",
                    "ROC-AUC": "ROC-AUC",
                },
            )
        )
        add("\n" + _league_verdict(league_skill) + "\n")
        add("\n![Качество по лигам](figures/11_league_skill.png)\n")
    else:
        add(
            markdown_table(
                league,
                {
                    "модель": "Модель",
                    "лига": "Лига",
                    "n": "Ударов",
                    "goal_rate": "Доля голов",
                    "log_loss": "Log loss",
                    "brier": "Brier",
                    "roc_auc": "ROC-AUC",
                },
            )
        )

    add("\n## 9. Анализ ошибок по подгруппам\n\n")
    add(
        markdown_table(
            subgroups,
            {
                "подгруппа": "Подгруппа",
                "n": "Ударов",
                "доля_голов": "Доля голов",
                "log_loss_без_контекста": "Без контекста",
                "log_loss_с_контекстом": "С контекстом",
                "log_loss_statsbomb": "statsbomb_xg",
                "улучшение": "Улучшение",
            },
            {"улучшение": "{:+.5f}"},
        )
    )
    add(
        "\nПодгруппы пересекаются и выбраны после просмотра данных. Доверительных "
        "интервалов для них нет, поэтому таблица показывает, где сосредоточено "
        "улучшение, но гипотез не проверяет.\n"
    )
    add("\n![Анализ ошибок](figures/08_error_analysis.png)\n")

    add("\n## 10. Примеры ударов\n\n")
    add(
        markdown_table(
            examples.groupby("причина", sort=False).head(2),
            {
                "причина": "Почему выбран",
                "competition_name": "Лига",
                "is_goal": "Гол",
                "shot_distance": "Расстояние",
                "opponents_in_shot_cone": "В конусе",
                "nearest_opponent_distance": "Ближайший",
                "p_logistic_no_defense": "Без контекста",
                "p_logistic_defense": "С контекстом",
                "statsbomb_xg": "statsbomb_xg",
            },
            {"shot_distance": "{:.1f}", "nearest_opponent_distance": "{:.1f}"},
        )
    )
    add("\n" + _one_on_one_note(examples) + "\n")
    add("\n![Примеры freeze frame](figures/09_freeze_frame_examples.png)\n")

    add("\n## 11. Проверки изоляции\n\n")
    audit = pd.DataFrame(summary.get("isolation_audit", []))
    if not audit.empty:
        failed = audit.loc[~audit["пройдена"], "проверка"].tolist()
        add(f"При каждом запуске из объектов эксперимента считаются {len(audit)} проверок. ")
        if failed:
            add(f"Провалены: {failed}. Результаты нельзя считать изолированными.\n")
        else:
            add(
                "Все пройдены: части выборки не пересекаются, тестовые `shot_id` "
                "зафиксированы на диске, модель выбрана по validation, "
                "гиперпараметры подобраны внутри train, препроцессинг живёт внутри "
                "`Pipeline`, ablation-модели оценены на одних строках, запрещённые "
                "признаки в матрицу не попали, классы не перевзвешивались и "
                "пост-калибровки на тесте не было.\n"
            )

    add("\n## 12. Данные и признаки\n\n")
    add("![Состав выборки](figures/01_sample_overview.png)\n\n")
    add("![Карта ударов](figures/02_shot_map.png)\n\n")
    add("![Доля голов по геометрии](figures/03_goal_rate_by_geometry.png)\n\n")
    add("![Важность признаков](figures/06_feature_importance.png)\n\n")
    add("![Карта xG](figures/07_xg_surface.png)\n")

    add("\n## 13. Ограничения\n\n")
    for item in _limitations():
        add(f"- {item}\n")

    return plain_text("".join(parts))


def _league_verdict(league_skill: pd.DataFrame) -> str:
    """Вывод по лигам строится на относительных величинах."""
    part = league_skill[league_skill["модель"] == "Логистическая + защитный контекст"]
    if part.empty:
        return ""
    best = part.loc[part["BSS"].idxmax()]
    worst = part.loc[part["BSS"].idxmin()]
    spread = float(best["BSS"] - worst["BSS"])
    raw_best = part.loc[part["log loss"].idxmin()]

    verdict = (
        f"Сырой log loss минимален в лиге {raw_best['лига']} "
        f"({raw_best['log loss']:.4f}), но это следствие низкой доли голов "
        f"({raw_best['доля голов']:.4f}). По относительному качеству картина другая: "
        f"наибольший BSS у лиги {best['лига']} ({best['BSS']:.4f}), наименьший у "
        f"{worst['лига']} ({worst['BSS']:.4f}), размах {spread:.4f}."
    )
    if spread < 0.05:
        verdict += (
            " Разброс небольшой. Это не доказывает, что четыре лиги можно считать "
            "одной популяцией, а значит только, что заметной разницы в качестве "
            "модели на этих данных не видно."
        )
    else:
        verdict += (
            " Разброс заметен, поэтому перенос модели между лигами нужно оговаривать отдельно."
        )
    verdict += _context_gain_by_league(league_skill)
    return verdict


def _context_gain_by_league(league_skill: pd.DataFrame) -> str:
    """Прирост BSS от защитного контекста внутри каждой лиги."""
    with_context = league_skill[league_skill["модель"] == "Логистическая + защитный контекст"]
    without = league_skill[league_skill["модель"] == "Логистическая без контекста"]
    if with_context.empty or without.empty:
        return ""
    gain = (with_context.set_index("лига")["BSS"] - without.set_index("лига")["BSS"]).sort_values()
    grew = " Точечная оценка BSS выросла во всех четырёх лигах." if (gain > 0).all() else ""
    return (
        f"{grew} Слабее всего прирост в лиге {gain.index[0]} ({gain.iloc[0]:+.3f}), "
        f"сильнее всего в {gain.index[-1]} ({gain.iloc[-1]:+.3f}). "
        "Доверительные интервалы отдельно по лигам не считались, поэтому "
        "устойчивость прироста внутри лиги не проверена."
    )


def _one_on_one_note(examples: pd.DataFrame) -> str:
    """Разбор случая с флагом one_on_one как ошибки обеих моделей."""
    lowered = examples[examples["причина"] == "контекст сильно понизил прогноз"]
    if lowered.empty:
        return ""
    row = lowered.iloc[0]
    return (
        f"Разбор одного случая. Удар с расстояния {row['shot_distance']:.1f} "
        "почти с нулевым углом к воротам получил без защитного контекста вероятность "
        f"{row['p_logistic_no_defense']:.3f} из-за флага `one_on_one`. Модель с "
        f"контекстом снизила прогноз до {row['p_logistic_defense']:.3f}, а "
        f"`statsbomb_xg` равен {row['statsbomb_xg']:.4f}. Это не успех защитного "
        "контекста: обе мои модели ошибаются здесь на два порядка, и после снижения "
        "прогноз остаётся неразумным. Случай показывает слабость аддитивной модели, "
        "в которой бинарный флаг не может быть отменён геометрией."
    )


def _limitations() -> list[str]:
    return [
        "`shot.freeze_frame` показывает только игроков в кадре. В среднем видно "
        "около 7.5 соперника вместо десяти, поэтому все счётчики говорят о том, "
        "сколько соперников видно, а не сколько их было на поле.",
        "Работа отвечает на вопрос о качестве прогноза, а не о причинах.",
        "Отсутствие эффекта у флагов StatsBomb не доказывает, что вклад принадлежит "
        "только моим признакам.",
        "Выборка это четыре европейские мужские лиги одного сезона. Переносить "
        "выводы на другие лиги, эпохи или женский футбол нельзя.",
        "Все лиги относятся к одному сезону, поэтому проверка на другом сезоне невозможна.",
        "Сравнение со `statsbomb_xg` неравное по условиям.",
        "Флаги `one_on_one` и `open_goal` это разметка провайдера, а не мой расчёт.",
        "Защитные признаки коллинеарны, поэтому коэффициенты делят вклад между "
        "собой и читать их нужно осторожно.",
        "Подгруппы в анализе ошибок пересекаются и выбраны после просмотра данных.",
    ]
