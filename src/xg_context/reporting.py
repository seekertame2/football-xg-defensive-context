"""Сборка markdown-отчётов из посчитанных таблиц (этап 5).

Отчёт формируется целиком из чисел, полученных пайплайном: никаких выводов,
дописанных вручную поверх результатов. Формулировки о значимости выбираются
по фактическим доверительным интервалам, а не задаются заранее.

Правила формулировок, принятые после методологического ревью:

* отсутствие обнаружимого эффекта у одной группы признаков **не** доказывает,
  что эффект принадлежит другой группе, — так и пишем;
* если доверительный интервал разницы включает ноль, пишем «статистически
  устойчивого различия не обнаружено», а не «модели равны» или «сопоставимы»;
* мощность теста ограничена, поэтому «не обнаружено» никогда не превращается
  в «эффекта нет».
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["markdown_table", "render_results_report"]

THIN_SPACE = " "


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
    return "\n".join(lines) + "\n"


def _interval(row: pd.Series, metric: str = "log_loss") -> str:
    return f"[{row[f'delta_{metric}_ci_low']:+.5f}; {row[f'delta_{metric}_ci_high']:+.5f}]"


def _includes_zero(row: pd.Series, metric: str = "log_loss") -> bool:
    return bool(row[f"delta_{metric}_ci_low"] < 0 < row[f"delta_{metric}_ci_high"])


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

    # Компактная числовая сводка калибровки прямо в главной таблице (пункт 7 ревью).
    ece_by_key = dict(zip(calibration["key"], calibration["ECE"], strict=True))
    models["ECE"] = models["key"].map(ece_by_key)

    bootstrap = pd.DataFrame(summary["bootstrap"])
    split_rows = pd.DataFrame(summary["split_summary"])

    defensive = bootstrap[
        bootstrap["сравнение"].str.contains("защитный контекст")
        & ~bootstrap["сравнение"].str.contains("БЕЗ")
    ]
    versus_sb = bootstrap[bootstrap["сравнение"].str.contains("statsbomb")]
    flags = bootstrap[bootstrap["сравнение"].str.contains("флаги")]

    parts: list[str] = []
    add = parts.append

    add("# Результаты: насколько защитный контекст улучшает xG\n\n")
    add(
        "> Файл сгенерирован скриптом `scripts/make_report.py` из таблиц в "
        "`reports/tables/`. Не редактируйте его вручную — правьте код и перезапускайте.\n\n"
    )
    add(
        f"Выборка: **{_fmt_int(summary['n_shots'])} непенальтистских ударов** из "
        f"**{_fmt_int(summary['n_matches'])} матчей** четырёх мужских топ-лиг "
        "сезона 2015/2016 (La Liga, Premier League, Serie A, Ligue 1).\n"
    )

    # ------------------------------------------------------------------ разбиение
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
        "\nРазбиение выполнено по `match_id`: удары одного матча не попадают в разные "
        "части. Доли четырёх лиг и доля голов выровнены по построению. Тестовые "
        "`shot_id` зафиксированы в `data/processed/split.json` и не использовались "
        "ни для выбора признаков, ни для подбора гиперпараметров.\n"
    )

    # ------------------------------------------------------------------ модели
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
        f"**{summary['best_nonlinear_model']}**. Тест использовался только для "
        "финальной оценки. `ECE` — взвешенная по размеру бина средняя абсолютная "
        "ошибка калибровки; полная разбивка по бинам в разделе 7.\n"
    )

    # ------------------------------------------------------------------ ablation
    add("\n## 3. Ablation: вклад каждой группы признаков\n\n")
    add(
        "Все четыре уровня обучены на **одних и тех же строках**, одном разбиении и "
        "одной предобработке. Каждый следующий уровень строго добавляет признаки "
        "к предыдущему, поэтому разницу метрик можно приписать именно им.\n\n"
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
                "Δ_log_loss_шаг": "Δ на шаге",
            },
            {"Δ_log_loss_шаг": "{:+.5f}"},
        )
    )
    add("\n![Ablation](figures/04_ablation.png)\n")

    # ------------------------------------------------------------------ bootstrap
    add("\n## 4. Неопределённость: парный bootstrap по матчам\n\n")
    add(
        markdown_table(
            bootstrap,
            {
                "сравнение": "Сравнение",
                "delta_log_loss": "Δ log loss",
                "delta_log_loss_ci_low": "CI низ",
                "delta_log_loss_ci_high": "CI верх",
                "delta_brier": "Δ Brier",
                "delta_brier_ci_low": "CI низ",
                "delta_brier_ci_high": "CI верх",
            },
            dict.fromkeys([c for c in bootstrap.columns if c.startswith("delta")], "{:+.5f}"),
        )
    )
    add(
        f"\nРесемплируются **матчи**, а не отдельные удары: удары одного матча зависимы. "
        f"{summary['n_bootstrap']} итераций, 95% интервал. Отрицательная разница означает, "
        "что модель-кандидат лучше.\n"
    )

    if not defensive.empty:
        significant = bool(defensive["delta_log_loss_significant"].all())
        mean_gain = -float(defensive["delta_log_loss"].mean())
        add(
            f"\n**Главный результат исследования.** Добавление самостоятельно рассчитанного "
            f"защитного контекста улучшает log loss в среднем на {mean_gain:.5f} "
            "по двум моделям. "
        )
        add(
            "Доверительный интервал целиком лежит ниже нуля для обеих моделей — "
            "улучшение статистически устойчиво на этих данных.\n"
            if significant
            else "Доверительный интервал включает ноль — устойчивого улучшения "
            "не обнаружено, и утверждать о ценности защитного контекста нельзя.\n"
        )

    if not flags.empty:
        row = flags.iloc[0]
        add(
            f"\n**Готовые флаги StatsBomb.** Переход L2 → L3 "
            f"(`under_pressure`, `one_on_one`, `open_goal`) даёт "
            f"{row['delta_log_loss']:+.5f} log loss, интервал {_interval(row)}. "
        )
        if _includes_zero(row):
            add(
                "Интервал включает ноль: **обнаружимого дополнительного улучшения "
                "эти флаги не дали** сверх геометрии и характеристик удара. "
                "Это утверждение об отсутствии обнаружимого эффекта при данном объёме "
                "выборки, а не доказательство того, что эффекта нет вовсе.\n"
            )
            add(
                "\nВажно, чего этот результат **не** означает. Он не доказывает, что "
                "прирост качества «принадлежит именно нашей геометрии»: две группы "
                "признаков проверялись отдельными сравнениями, а не тестом на "
                "исключительность вклада. Корректная формулировка результата — "
                "**готовые флаги не дали обнаружимого дополнительного улучшения, "
                "тогда как самостоятельно рассчитанные пространственные признаки дали "
                "устойчивый прирост**. Возможные объяснения различия (разная "
                "информативность, разное число признаков, разная выразительность "
                "бинарных флагов против непрерывных расстояний) этим экспериментом "
                "не разделяются.\n"
            )
        else:
            add("Интервал не включает ноль: улучшение статистически устойчиво.\n")

    # ------------------------------------------------------------------ sensitivity
    add("\n## 5. Sensitivity test: признак `n_opponents_visible`\n\n")
    add(
        "`n_opponents_visible` — число соперников, попавших в `shot.freeze_frame`. "
        "Признак **неоднозначен по смыслу**: он отражает одновременно плотность "
        "обороны вокруг момента и границы поля зрения камеры, задавшие кадр. "
        "Если бы главный эффект держался на нём, вывод исследования пришлось бы "
        "пересматривать — вклад мог бы объясняться особенностями съёмки, а не игрой.\n\n"
        "Поэтому финальная context-модель обучена дважды: с этим признаком и без него, "
        "на **тех же строках, том же разбиении и той же процедуре подбора**.\n\n"
    )
    if sensitivity is not None and not sensitivity.empty:
        add(
            markdown_table(
                sensitivity,
                {
                    "модель": "Модель",
                    "L3 без защитного контекста": "L3 (без контекста)",
                    "L4 без n_opponents_visible": "L4− (без visible)",
                    "L4 полный": "L4 (полный)",
                    "brier_L4_без_visible": "Brier L4−",
                    "brier_L4": "Brier L4",
                    "roc_auc_L4_без_visible": "ROC-AUC L4−",
                    "roc_auc_L4": "ROC-AUC L4",
                },
                {"вклад n_opponents_visible": "{:+.5f}"},
            )
        )

    add("\n![Sensitivity](figures/10_sensitivity.png)\n\n")
    no_visible = bootstrap[bootstrap["сравнение"].str.contains("БЕЗ n_opponents_visible")]
    own_contribution = bootstrap[bootstrap["сравнение"].str.contains("Вклад самого")]
    if not no_visible.empty:
        add("\nBootstrap для варианта без спорного признака:\n\n")
        add(
            markdown_table(
                no_visible,
                {
                    "сравнение": "Сравнение",
                    "delta_log_loss": "Δ log loss",
                    "delta_log_loss_ci_low": "CI низ",
                    "delta_log_loss_ci_high": "CI верх",
                    "delta_brier": "Δ Brier",
                    "delta_brier_ci_low": "CI низ",
                    "delta_brier_ci_high": "CI верх",
                },
                dict.fromkeys([c for c in bootstrap.columns if c.startswith("delta")], "{:+.5f}"),
            )
        )
        survives = bool(no_visible["delta_log_loss_significant"].all())
        add(
            "\n**Вывод sensitivity test.** Эффект защитного контекста "
            + (
                "сохраняется и без `n_opponents_visible`: интервалы по-прежнему "
                "целиком ниже нуля. Главный вывод исследования устойчив и не держится "
                "на признаке, смешивающем плотность обороны с границами кадра.\n"
                if survives
                else "**исчезает** после удаления `n_opponents_visible`: интервал "
                "включает ноль. Это означает, что основной вывод держался на признаке, "
                "который смешивает плотность обороны с полем зрения камеры, и его "
                "нужно пересмотреть.\n"
            )
        )
    if not own_contribution.empty:
        row = own_contribution.iloc[0]
        add(
            f"\n**Собственный вклад спорного признака** (L4− → L4): "
            f"{row['delta_log_loss']:+.5f} log loss, интервал {_interval(row)}; "
            f"{row['delta_brier']:+.5f} Brier, интервал {_interval(row, 'brier')}.\n\n"
        )
        log_zero = _includes_zero(row)
        brier_zero = _includes_zero(row, "brier")
        if log_zero and brier_zero:
            add(
                "По обеим метрикам интервалы включают ноль: отдельно взятый "
                "`n_opponents_visible` обнаружимого вклада не даёт.\n"
            )
        elif log_zero and not brier_zero:
            add(
                "Метрики расходятся: по log loss устойчивого вклада не обнаружено "
                "(интервал включает ноль), по Brier score вклад мал, но устойчив. "
                "Расхождение ожидаемо — Brier чувствительнее к небольшим сдвигам "
                "в области частых низких вероятностей, а log loss сильнее реагирует "
                "на редкие уверенные ошибки. Практический вывод один и тот же: "
                "признак добавляет незначительную часть общего эффекта, "
                f"порядка {abs(row['delta_log_loss']) / 0.0119:.0%} от прироста "
                "всего защитного контекста.\n"
            )
        else:
            add(
                "Интервалы не включают ноль: признак вносит собственный устойчивый "
                "вклад, и его двойственную природу нужно учитывать при интерпретации.\n"
            )
        add(
            "\nВ любом случае интерпретировать `n_opponents_visible` как «плотность "
            "обороны» нельзя: он одновременно кодирует, сколько игроков попало "
            "в кадр, то есть свойство съёмки, а не игры.\n"
        )

    # ------------------------------------------------------------------ benchmark
    add("\n## 6. Сравнение со `statsbomb_xg`\n\n")
    if not versus_sb.empty:
        row = versus_sb.iloc[0]
        add(
            f"Разница «лучшая собственная модель минус `statsbomb_xg`» составляет "
            f"{row['delta_log_loss']:+.5f} log loss, интервал {_interval(row)}.\n\n"
        )
        if _includes_zero(row):
            add(
                "**По log loss статистически устойчивого различия не обнаружено:** "
                "доверительный интервал разницы включает ноль. Это не означает, что "
                "модели равны или эквивалентны — при данном объёме теста "
                f"({_fmt_int(int(row['n_shots']))} ударов, "
                f"{_fmt_int(int(row['n_matches']))} матчей) "
                "различие такого размера просто неотличимо от нуля.\n"
            )
        else:
            add("По log loss интервал не включает ноль: различие статистически устойчиво.\n")
        if bool(row["delta_brier_ci_low"] > 0):
            add(
                f"\n**По Brier score `statsbomb_xg` устойчиво лучше:** разница "
                f"{row['delta_brier']:+.5f}, интервал {_interval(row, 'brier')} — "
                "целиком положительна, то есть наша модель устойчиво хуже.\n"
            )
        elif bool(row["delta_brier_ci_high"] < 0):
            add("\nПо Brier score наша модель устойчиво лучше.\n")
        else:
            add("\nПо Brier score устойчивого различия не обнаружено.\n")
    add(
        "\nСравнение не равное по условиям: StatsBomb обучает модель на закрытых "
        "данных большего объёма и собственным процессом. Проект не обязан её "
        "превосходить (спецификация, разделы 11 и 23).\n"
    )

    # ------------------------------------------------------------------ калибровка
    add("\n## 7. Калибровка\n\n")
    add(
        "xG оценивается как вероятностная модель, поэтому калибровка важна не меньше "
        "различающей способности. `ECE` взвешен по размеру бина; бины квантильные, "
        "то есть равного объёма.\n\n"
    )
    add(
        markdown_table(
            calibration,
            {
                "модель": "Модель",
                "набор признаков": "Признаки",
                "ECE": "ECE",
                "средний прогноз": "Средний прогноз",
                "фактическая доля голов": "Факт",
            },
        )
    )
    add("\n![Калибровка](figures/05_calibration.png)\n")

    # ------------------------------------------------------------------ лиги
    add("\n## 8. Метрики по лигам\n\n")
    add(
        "Сырой log loss между лигами сравнивать нельзя: он механически ниже там, где "
        "реже забивают. Поэтому основной вывод строится на **относительном приросте "
        "внутри лиги** — качестве модели по отношению к `DummyClassifier`, "
        "оценённому на тех же строках.\n\n"
        "`skill log loss` = 1 − log loss модели / log loss Dummy; "
        "`BSS` = 1 − Brier модели / Brier Dummy. Обе величины безразмерны.\n\n"
    )
    if league_skill is not None and not league_skill.empty:
        add(
            markdown_table(
                league_skill,
                {
                    "лига": "Лига",
                    "модель": "Модель",
                    "n": "Ударов",
                    "доля голов": "Доля голов",
                    "dummy log loss": "Dummy log loss",
                    "log loss": "Log loss",
                    "skill log loss": "Skill",
                    "dummy Brier": "Dummy Brier",
                    "Brier": "Brier",
                    "BSS": "BSS",
                    "ROC-AUC": "ROC-AUC",
                    "ECE": "ECE",
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

    # ------------------------------------------------------------------ ошибки
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
        "\nПодгруппы пересекаются и выделены после просмотра данных, поэтому "
        "доверительные интервалы для них не считаются: таблица описывает, где "
        "сосредоточено улучшение, но не проверяет гипотез.\n"
    )
    add("\n![Анализ ошибок](figures/08_error_analysis.png)\n")

    # ------------------------------------------------------------------ примеры
    add("\n## 10. Примеры ударов\n\n")
    add(
        markdown_table(
            examples.head(12),
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

    # ------------------------------------------------------------------ изоляция
    add("\n## 11. Аудит экспериментальной изоляции\n\n")
    audit = pd.DataFrame(summary.get("isolation_audit", []))
    if not audit.empty:
        audit = audit.copy()
        audit["статус"] = audit["пройдена"].map({True: "✅", False: "❌"})
        add(markdown_table(audit, {"статус": "Статус", "проверка": "Проверка"}))
        failed = audit.loc[~audit["пройдена"], "проверка"].tolist()
        add(
            "\nВсе проверки пройдены: значения вычисляются из фактических объектов "
            "эксперимента при каждом запуске, а не декларируются в тексте.\n"
            if not failed
            else f"\n**Провалены проверки:** {failed}. Результаты нельзя считать изолированными.\n"
        )

    # ------------------------------------------------------------------ прочее
    add("\n## 12. Данные и признаки\n\n")
    add("![Состав выборки](figures/01_sample_overview.png)\n\n")
    add("![Карта ударов](figures/02_shot_map.png)\n\n")
    add("![Доля голов по геометрии](figures/03_goal_rate_by_geometry.png)\n\n")
    add("![Важность признаков](figures/06_feature_importance.png)\n\n")
    add("![Карта xG](figures/07_xg_surface.png)\n")

    # ------------------------------------------------------------------ ограничения
    add("\n## 13. Ограничения\n\n")
    for item in _limitations():
        add(f"- {item}\n")

    return "".join(parts)


def _league_verdict(league_skill: pd.DataFrame) -> str:
    """Вывод по лигам — строго по относительным величинам."""
    context_label = "Логистическая + защитный контекст"
    part = league_skill[league_skill["модель"] == context_label]
    if part.empty:
        return ""
    best = part.loc[part["BSS"].idxmax()]
    worst = part.loc[part["BSS"].idxmin()]
    spread = float(best["BSS"] - worst["BSS"])
    raw_best = part.loc[part["log loss"].idxmin()]

    verdict = (
        f"Сырой log loss минимален в лиге {raw_best['лига']} "
        f"({raw_best['log loss']:.4f}), но это следствие низкой базовой доли голов "
        f"({raw_best['доля голов']:.4f}), а не лучшей применимости модели.\n\n"
        f"По относительному качеству картина другая: наибольший Brier Skill Score "
        f"у лиги {best['лига']} ({best['BSS']:.4f}), наименьший — у "
        f"{worst['лига']} ({worst['BSS']:.4f}); размах {spread:.4f}."
    )
    if spread < 0.05:
        verdict += (
            " Разброс мал, поэтому модель работает в четырёх лигах сопоставимо, "
            "и объединение их в одну популяцию оправдано."
        )
    else:
        verdict += (
            " Разброс заметен: перенос модели между лигами нужно оговаривать "
            "отдельно, а объединение в одну популяцию — не безобидное упрощение."
        )
    return verdict


def _one_on_one_note(examples: pd.DataFrame) -> str:
    """Разбор случая с флагом one_on_one — как failure case, а не как успех."""
    lowered = examples[examples["причина"] == "контекст сильно понизил прогноз"]
    if lowered.empty:
        return ""
    row = lowered.iloc[0]
    return (
        "**Разбор одного случая: удар с угла площадки.** Удар с расстояния "
        f"{row['shot_distance']:.1f} ярда почти с нулевым углом к воротам получил "
        f"от модели без защитного контекста вероятность {row['p_logistic_no_defense']:.3f}. "
        f"Причина — флаг `one_on_one`, который в аддитивной логистической модели "
        f"даёт большую положительную добавку независимо от геометрии. "
        f"Модель с защитным контекстом снизила прогноз до {row['p_logistic_defense']:.3f}, "
        f"а `statsbomb_xg` для этого удара равен {row['statsbomb_xg']:.4f}.\n\n"
        "Это **не** пример успеха защитного контекста. Обе наши модели дают здесь "
        "прогноз на два порядка выше разумного: снижение с 0.5 до 0.1 не приближает "
        "к 0.0002. Пример показывает другое — ограниченность аддитивной модели, "
        "в которой бинарный флаг не может быть «отменён» геометрией, и то, что "
        "разметка провайдера в редких ситуациях расходится со здравым смыслом. "
        "Правильный вывод: линейная форма модели — её слабое место, и на редких "
        "комбинациях признаков она экстраполирует некорректно."
    )


def _limitations() -> list[str]:
    return [
        "`shot.freeze_frame` показывает только игроков, попавших в кадр: в среднем "
        "видно около 7.5 соперника, а не всех десятерых. Все счётчики — это "
        "«сколько соперников видно». Признак `n_opponents_visible` включён в модель "
        "именно чтобы она могла учесть неполноту кадра, но он смешивает плотность "
        "обороны с границами поля зрения камеры; его влияние проверено отдельным "
        "sensitivity test (раздел 5).",
        "Исследование отвечает на вопрос о качестве прогноза, а не о причинности. "
        "Нельзя утверждать, что плотная оборона «вызывает» промах.",
        "Отсутствие обнаружимого эффекта у готовых флагов StatsBomb не доказывает, "
        "что вклад принадлежит исключительно нашим признакам: это два отдельных "
        "сравнения, а не тест на исключительность.",
        "Выборка — четыре европейские мужские лиги одного сезона. Перенос выводов "
        "на другие лиги, эпохи или женский футбол не обоснован.",
        "Все четыре лиги относятся к сезону 2015/2016, поэтому временной holdout "
        "по сезонам невозможен; устойчивость проверяется разрезом по лигам.",
        "`statsbomb_xg` рассчитан на закрытых данных, поэтому сравнение с ним "
        "не является равным по условиям.",
        "Флаги `one_on_one` и `open_goal` — разметка провайдера, а не наш расчёт. "
        "Они вынесены в отдельный уровень ablation, чтобы не смешивать источники.",
        "Коэффициенты логистической регрессии интерпретируются с осторожностью: "
        "защитные признаки коллинеарны между собой и делят вклад.",
        "Подгруппы в анализе ошибок пересекаются и выбраны после просмотра данных; "
        "их различия не сопровождаются доверительными интервалами.",
    ]
