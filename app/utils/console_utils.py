"""Утилиты для форматированного вывода метрик обучения в консоль.

Использует библиотеку *Rich* [[via context7]].
"""
from __future__ import annotations

from rich.console import Group
from rich.table import Table
import rich.box


def create_metrics_renderable(
    generation: int,
    fitness_a: float,
    fitness_b: float,
    best_a: float | None = None,
    best_b: float | None = None,
    avg_nodes_a: float | None = None,
    avg_conns_a: float | None = None,
    avg_nodes_b: float | None = None,
    avg_conns_b: float | None = None,
) -> Group:
    """Создать Rich-renderable объект с таблицами метрик.

    Параметры
    ----------
    generation: int
        Текущий номер поколения.
    fitness_a: float
        Средний или лучший фитнес команды A.
    fitness_b: float
        Средний или лучший фитнес команды B.
    avg_nodes_a: float | None
        Среднее количество узлов в сетях команды A.
    avg_conns_a: float | None
        Среднее количество связей в сетях команды A.
    avg_nodes_b: float | None
        Среднее количество узлов в сетях команды B.
    avg_conns_b: float | None
        Среднее количество связей в сетях команды B.
    """
    title = "Показатели эволюции" if best_a is not None else "Счёт / Метрики"
    table = Table(title=title, show_lines=False, expand=True)
    table.add_column("Поколение", justify="right", style="cyan")
    table.add_column("Агент BLUE", justify="right", style="blue")
    table.add_column("Агент RED", justify="right", style="red")

    if best_a is None:
        # Старый режим: просто вывод двух чисел
        row_a = f"{fitness_a:.2f}"
        row_b = f"{fitness_b:.2f}"
    else:
        # Новый расширенный режим: среднее и максимум
        row_a = f"avg {fitness_a:.2f} | best {best_a:.2f}"
        row_b = f"avg {fitness_b:.2f} | best {best_b:.2f}"

    table.add_row(str(generation), row_a, row_b)

    renderables = [table]
    if avg_nodes_a is not None and avg_conns_a is not None and avg_nodes_b is not None and avg_conns_b is not None:
        complexity_table = Table(
            title="[dim]Сложность сети[/dim]", show_lines=True, expand=True, show_edge=True, box=rich.box.HEAVY_HEAD
        )
        complexity_table.add_column("Команда", justify="center")
        complexity_table.add_column("Среднее кол-во узлов", justify="center", style="magenta")
        complexity_table.add_column("Среднее кол-во связей", justify="center", style="yellow")
        complexity_table.add_row("[blue]Агент BLUE[/blue]", f"{avg_nodes_a:.1f}", f"{avg_conns_a:.1f}")
        complexity_table.add_row("[red]Агент RED[/red]", f"{avg_nodes_b:.1f}", f"{avg_conns_b:.1f}")
        renderables.append(complexity_table)

    return Group(*renderables)