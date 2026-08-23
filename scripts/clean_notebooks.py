"""Аккуратная чистка ноутбуков перед релизом (этап 6).

Ноутбуки **не** очищаются полностью: содержательные таблицы, графики и выводы
сохраняются, чтобы результат читался прямо на GitHub. Удаляется только шум:

* потоки ``stderr`` — предупреждения библиотек и служебные сообщения;
* прогресс-бары и перерисовки одной и той же строки (символ возврата каретки);
* ANSI-последовательности цвета;
* пустые выводы;
* дубликаты подряд идущих одинаковых выводов.

Запуск::

    python scripts/clean_notebooks.py
    python scripts/clean_notebooks.py --check   # только проверка, без записи
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xg_context.config import PROJECT_ROOT

logger = logging.getLogger("clean_notebooks")

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PROGRESS_PATTERN = re.compile(r"^.*\r(?!\n)")


def _clean_text(value: Any) -> Any:
    """Убрать ANSI-цвета и перерисовки прогресс-баров из текста вывода."""
    if isinstance(value, list):
        return [_clean_text(item) for item in value]
    if isinstance(value, str):
        text = ANSI_PATTERN.sub("", value)
        # Прогресс-бар перерисовывает строку через \r: оставляем последнее состояние.
        if "\r" in text:
            text = PROGRESS_PATTERN.sub("", text)
        return text
    return value


def _is_noise(output: dict[str, Any]) -> bool:
    """Служебный вывод, который не несёт аналитического содержания."""
    if output.get("output_type") == "stream" and output.get("name") == "stderr":
        return True
    if output.get("output_type") == "stream":
        text = "".join(output.get("text") or [])
        return not text.strip()
    if output.get("output_type") == "error":
        return False  # ошибку прятать нельзя — её надо чинить
    data = output.get("data") or {}
    if not data:
        return True
    # Голая ссылка на matplotlib-объект без картинки не нужна.
    if set(data) == {"text/plain"}:
        text = "".join(data["text/plain"]).strip()
        return text.startswith("<") and text.endswith(">")
    return False


def clean_notebook(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Вернуть очищенный ноутбук и число удалённых выводов."""
    removed = 0
    for cell in payload.get("cells", []):
        outputs = cell.get("outputs")
        if not outputs:
            continue

        kept: list[dict[str, Any]] = []
        for output in outputs:
            if _is_noise(output):
                removed += 1
                continue
            if "text" in output:
                output["text"] = _clean_text(output["text"])
            if "data" in output:
                output["data"] = {key: _clean_text(value) for key, value in output["data"].items()}
            # Подряд идущие одинаковые выводы — дубликат перерисовки.
            if kept and json.dumps(kept[-1], sort_keys=True) == json.dumps(output, sort_keys=True):
                removed += 1
                continue
            kept.append(output)

        cell["outputs"] = kept
        if not kept:
            cell["execution_count"] = cell.get("execution_count")
    return payload, removed


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="только сообщить, что было бы удалено, без изменения файлов",
    )
    args = parser.parse_args(argv)

    total_removed = 0
    for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        before = path.stat().st_size
        cleaned, removed = clean_notebook(payload)
        total_removed += removed

        if args.check:
            logger.info("%-40s удалить выводов: %d", path.name, removed)
            continue

        path.write_text(json.dumps(cleaned, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        after = path.stat().st_size
        logger.info(
            "%-40s выводов удалено: %2d, размер %d -> %d КБ",
            path.name,
            removed,
            before // 1024,
            after // 1024,
        )

    logger.info("Всего удалено служебных выводов: %d", total_removed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
