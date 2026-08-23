"""Сборка markdown-отчётов из посчитанных таблиц (этап 5).

Отчёт формируется целиком из чисел, полученных пайплайном: никаких выводов,
дописанных вручную поверх результатов. Формулировки о значимости выбираются
по фактическим доверительным интервалам, а не задаются заранее.
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


def render_results_report(
    summary: dict[str, Any],
    ablation: pd.DataFrame,
    subgroups: pd.DataFrame,
    examples: pd.DataFrame,
    league: pd.DataFrame,
    calibration: pd.DataFrame,
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
            }
            for m in summary["models"]
        ]
    ).sort_values("log_loss", ignore_index=True)

    bootstrap = pd.DataFrame(summary["bootstrap"])
    split_rows = pd.DataFrame(summary["split_summary"])
    defensive = bootstrap[bootstrap["сравнение"].str.contains("защитный контекст")]
    versus_sb = bootstrap[bootstrap["сравнение"].str.contains("statsbomb")]

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
            },
        )
    )
    add(
        f"\nЛучшая нелинейная модель выбрана по validation: **{summary['best_nonlinear_model']}**. "
        "Тест использовался только для финальной оценки.\n"
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
            f"по двум моделям. "
        )
        add(
            "Доверительный интервал целиком лежит ниже нуля для обеих моделей — "
            "улучшение статистически устойчиво, а не случайно.\n"
            if significant
            else "Доверительный интервал включает ноль — улучшение статистически "
            "неубедительно, и утверждать о ценности защитного контекста нельзя.\n"
        )

    flags = bootstrap[bootstrap["сравнение"].str.contains("флаги")]
    if not flags.empty:
        row = flags.iloc[0]
        flags_significant = bool(row["delta_log_loss_significant"])
        add(
            f"\n**Отдельная находка.** Готовые контекстные флаги StatsBomb "
            f"(`under_pressure`, `one_on_one`, `open_goal`) дают "
            f"{row['delta_log_loss']:+.5f} log loss, интервал "
            f"[{row['delta_log_loss_ci_low']:+.5f}; {row['delta_log_loss_ci_high']:+.5f}]. "
        )
        add(
            "Улучшение статистически устойчиво.\n"
            if flags_significant
            else "Интервал включает ноль: сверх характеристик удара эти флаги "
            "измеримой ценности не добавляют. Это делает результат по собственному "
            "пространственному контексту содержательнее — он не дублирует разметку "
            "провайдера.\n"
        )

    # ------------------------------------------------------------------ benchmark
    add("\n## 5. Сравнение со `statsbomb_xg`\n\n")
    if not versus_sb.empty:
        row = versus_sb.iloc[0]
        add(
            f"Лучшая собственная модель отличается от `statsbomb_xg` на "
            f"{row['delta_log_loss']:+.5f} log loss, интервал "
            f"[{row['delta_log_loss_ci_low']:+.5f}; {row['delta_log_loss_ci_high']:+.5f}].\n\n"
        )
        includes_zero = bool(row["delta_log_loss_ci_low"] < 0 < row["delta_log_loss_ci_high"])
        add(
            "По log loss интервал включает ноль: **различие статистически неубедительно**, "
            "модели сопоставимы.\n"
            if includes_zero
            else "По log loss интервал не включает ноль: различие устойчиво.\n"
        )
        if bool(row["delta_brier_ci_low"] > 0):
            add(
                "\nПо Brier score интервал целиком положителен: здесь `statsbomb_xg` "
                "устойчиво точнее нашей модели.\n"
            )
        elif bool(row["delta_brier_ci_high"] < 0):
            add("\nПо Brier score наша модель устойчиво точнее.\n")
        else:
            add("\nПо Brier score различие статистически неубедительно.\n")
    add(
        "\nСравнение не полностью равное: StatsBomb обучает модель на закрытых данных "
        "большего объёма и собственным процессом. Проект не обязан её превосходить "
        "(спецификация, разделы 11 и 23).\n"
    )

    # ------------------------------------------------------------------ калибровка
    add("\n## 6. Калибровка\n\n")
    add(
        "xG оценивается как вероятностная модель, поэтому калибровка важна не меньше "
        "различающей способности. ECE взвешен по размеру бина.\n\n"
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
    add("\n## 7. Метрики по лигам\n\n")
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
    add("\n## 8. Анализ ошибок по подгруппам\n\n")
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
    add("\n![Анализ ошибок](figures/08_error_analysis.png)\n")

    # ------------------------------------------------------------------ примеры
    add("\n## 9. Примеры ударов\n\n")
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
    add("\n![Примеры freeze frame](figures/09_freeze_frame_examples.png)\n")

    # ------------------------------------------------------------------ прочее
    add("\n## 10. Данные и признаки\n\n")
    add("![Состав выборки](figures/01_sample_overview.png)\n\n")
    add("![Карта ударов](figures/02_shot_map.png)\n\n")
    add("![Доля голов по геометрии](figures/03_goal_rate_by_geometry.png)\n\n")
    add("![Важность признаков](figures/06_feature_importance.png)\n\n")
    add("![Карта xG](figures/07_xg_surface.png)\n")

    # ------------------------------------------------------------------ ограничения
    add("\n## 11. Ограничения\n\n")
    for item in _limitations():
        add(f"- {item}\n")

    return "".join(parts)


def _limitations() -> list[str]:
    return [
        "`shot.freeze_frame` показывает только игроков, попавших в кадр: в среднем "
        "видно около 7.5 соперника, а не всех десятерых. Поэтому все счётчики — это "
        "«сколько соперников видно», и признак `n_opponents_visible` включён в модель "
        "именно чтобы она могла учесть неполноту кадра.",
        "Исследование отвечает на вопрос о качестве прогноза, а не о причинности. "
        "Нельзя утверждать, что плотная оборона «вызывает» промах.",
        "Выборка — четыре европейские мужские лиги одного сезона. Перенос выводов "
        "на другие лиги, эпохи или женский футбол не обоснован.",
        "Все четыре лиги относятся к сезону 2015/2016, поэтому временной holdout "
        "по сезонам невозможен; устойчивость проверяется разрезом по лигам.",
        "`statsbomb_xg` рассчитан на закрытых данных, поэтому сравнение с ним "
        "не является равным по условиям.",
        "Флаги `one_on_one` и `open_goal` — разметка провайдера, а не наш расчёт. "
        "Они вынесены в отдельный уровень ablation, чтобы не завышать вклад "
        "собственной геометрии.",
        "Коэффициенты логистической регрессии интерпретируются с осторожностью: "
        "защитные признаки коллинеарны между собой и делят вклад.",
    ]
