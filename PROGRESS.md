# Ход работы

Журнал выполнения этапов из раздела 19 `PROJECT_CONTEXT.md`.
Принятые решения и их обоснования — в [`DECISIONS.md`](DECISIONS.md).
Результаты — в [`reports/results.md`](reports/results.md).

| Этап | Содержание | Статус |
| --- | --- | --- |
| 0 | Каркас проекта | ✅ завершён 2026-08-23 |
| 1 | Аудит данных | ✅ завершён 2026-08-23 |
| 2 | Датасет и геометрия | ✅ завершён 2026-08-23 |
| 3 | Baseline M0–M2 | ✅ завершён 2026-08-23 |
| 4 | Ансамбли и defensive context | ✅ завершён 2026-08-23 |
| 5 | Benchmark и анализ ошибок | ✅ завершён 2026-08-23 |
| 6 | Упаковка и релиз | ⏸️ не начат (по указанию владельца) |

---

## Что полностью готово

### Этап 0 — Каркас

Структура по разделу 14 спецификации, `pyproject.toml` с настройками pytest
и Ruff, `.gitignore`, CI на Python 3.11 и 3.12, документы `CLAUDE.md`,
`DECISIONS.md`, `PROGRESS.md`, `DATA_SOURCES.md`, `README.md`,
конфигурации `configs/data.yaml` и `configs/experiment.yaml`.

### Этап 1 — Аудит данных

Ревизия источника зафиксирована commit SHA. Проведён аудит всех
80 competition-season, выбрана и утверждена выборка. После построения датасета
отчёт [`reports/data_audit.md`](reports/data_audit.md) **перегенерирован по
полной выборке** — все доли покрытия теперь перепись, а не оценка по 3 матчам
на сезон.

### Этап 2 — Датасет и геометрия

- скачано 1517 файлов событий (4.52 ГБ), кеш в `data/raw/` — 4.9 ГБ;
- разобрано 5 321 459 событий, из них 37 888 ударов;
- после фильтров: **37 488** непенальтистских ударов (`all_eligible_shots`),
  **37 487** с защитным контекстом (`context_eligible_shots`);
- каждый шаг фильтрации записан с числом удалённых строк;
- реализованы и покрыты тестами геометрия удара и пространственный контекст
  обороны, включая конус удара и геометрию вратаря;
- `data/data_manifest.json` содержит SHA источника, хеш списка `match_id`,
  журнал фильтрации и SHA-256 обоих parquet-файлов.

### Этапы 3–5 — Модели, ablation, benchmark

- детерминированное разбиение 70/15/15 по `match_id`; доли лиг расходятся
  не более чем на 0.3 п.п., доли голов — на 0.07 п.п.; тестовые `shot_id`
  зафиксированы в `data/processed/split.json`;
- обучены M0–M5; гиперпараметры подобраны `StratifiedGroupKFold` внутри train;
- ablation по четырём уровням признаков на одних и тех же строках;
- парный bootstrap по матчам (1000 итераций) для всех ключевых сравнений;
- `statsbomb_xg` оценён на тех же тестовых ударах;
- метрики по каждой лиге, калибровка и ECE, анализ ошибок по 16 подгруппам;
- 9 графиков с русскими подписями в `reports/figures/`;
- 5 исполненных ноутбуков в `notebooks/`.

---

## Выполненные команды

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

```bash
python scripts/download_data.py --config configs/data.yaml --metadata-only
```

```bash
python scripts/audit_data.py --config configs/data.yaml
```

```bash
python scripts/download_data.py --config configs/data.yaml --selection
```

```bash
python scripts/build_dataset.py --config configs/data.yaml
```

```bash
python scripts/audit_data.py --selection
```

```bash
python scripts/run_experiments.py --config configs/experiment.yaml
```

```bash
python scripts/make_report.py
```

```bash
pytest -q && ruff check . && ruff format --check .
```

---

## Полученные результаты

| Модель | Признаки | Test log loss | Brier | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| `statsbomb_xg` (benchmark) | — | 0.2404 | 0.0682 | 0.835 |
| Логистическая регрессия | + защитный контекст | 0.2427 | 0.0692 | 0.830 |
| Случайный лес | + защитный контекст | 0.2433 | 0.0694 | 0.830 |
| Градиентный бустинг | без защитного контекста | 0.2523 | 0.0717 | 0.808 |
| Логистическая регрессия | геометрия | 0.2728 | 0.0770 | 0.760 |
| DummyClassifier | — | 0.3137 | 0.0859 | 0.500 |

**Главный результат.** Защитный контекст улучшает log loss на 0.0119
(логистическая регрессия, интервал [−0.0160; −0.0078]) и на 0.0107
(случайный лес, интервал [−0.0153; −0.0063]). Оба интервала целиком ниже нуля.

**Готовые флаги StatsBomb** значимого прироста не дают: −0.0017,
интервал [−0.0035; +0.0002] включает ноль.

**Сравнение со StatsBomb:** разница +0.0028 log loss, интервал
[−0.0009; +0.0065] включает ноль — по log loss модели неразличимы.
По Brier score StatsBomb устойчиво точнее.

---

## Проверки

```text
pytest              213 passed (включая smoke-тест полного пайплайна)
ruff check .        All checks passed!
ruff format --check все файлы отформатированы
notebooks           5 из 5 исполняются без ошибок
```

---

## Что осталось (этап 6, по указанию владельца не начат)

- финальный русский README с полным описанием результата;
- проверка воспроизводимости из чистого clone;
- очистка выводов ноутбуков перед релизом (сейчас выводы сохранены намеренно,
  чтобы результат был виден прямо на GitHub);
- limitations и future work в README;
- release/tag версии для заявки;
- необязательные пункты O-006 … O-008 из `DECISIONS.md`.

Явно **не** входит в проект по указанию владельца: нейросеть, приложение,
сложные признаки StatsBomb 360.

---

## Команда для продолжения

Всё, что нужно для повторения результата с нуля:

```bash
python scripts/download_data.py --config configs/data.yaml --selection && python scripts/build_dataset.py --config configs/data.yaml && python scripts/audit_data.py --selection && python scripts/run_experiments.py --config configs/experiment.yaml && python scripts/make_report.py
```

Если данные уже скачаны, а нужно только пересобрать отчёт и графики:

```bash
python scripts/make_report.py
```

---

## Плановый технический долг

- `--quick`-режимы скриптов не покрыты тестами напрямую: связность пайплайна
  проверяется `tests/test_pipeline_smoke.py` на синтетических событиях,
  а сами CLI-обёртки — нет;
- leave-one-league-out и PCA — необязательные расширения (O-006, O-007).
