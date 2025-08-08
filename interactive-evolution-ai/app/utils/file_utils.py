"""Файловые утилиты: сохранение/загрузка геномов и выбор файла.

Зависимости: встроенные модули ``os``, ``pickle``, ``datetime`` и ``tkinter``.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog


ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # interactive-evolution-ai/
MODELS_DIR = ROOT_DIR / "models"


def save_best_genomes(agent_a_genome: Any, agent_b_genome: Any) -> Path:
    """Сохранить лучшие геномы обеих команд и вернуть путь к папке."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = MODELS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "best_agent_BLUE.pkl").open("wb") as fp:
        pickle.dump(agent_a_genome, fp)
    with (out_dir / "best_agent_RED.pkl").open("wb") as fp:
        pickle.dump(agent_b_genome, fp)

    return out_dir


def pick_model_file(title: str | None = None) -> Path | None:  # noqa: D401
    """Открыть диалог выбора файла *.pkl* и вернуть Path или *None*.

    Использует `tkinter`'s ``filedialog``. Диалог работает в "withdrawn"‐режиме
    — главное окно не отображается.
    """
    title = title or "Выберите файл модели (.pkl)"

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    # Гарантируем, что папка с моделями существует
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = filedialog.askopenfilename(
        title=title,
        initialdir=str(MODELS_DIR),
        filetypes=[("Pickle", "*.pkl"), ("Все файлы", "*.*")],
    )
    root.destroy()

    return Path(file_path) if file_path else None
