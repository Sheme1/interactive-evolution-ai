"""Dynamic computation of simulation constants based on user settings.

The goal is to free the end-user from understanding the inner details of
NEAT/physics – the library derives sane values automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
import configparser

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SimConstants:
    """Container for dynamically calculated constants.

    ВАЖНО: Награды и штрафы за действия агентов теперь определяются
    в модуле app.core.fitness с использованием potential-based reward shaping.
    Здесь остаются только базовые параметры симуляции.
    """

    energy_max: int  # Максимальный запас энергии
    move_threshold: float  # Порог чувствительности для движения


def compute_constants(settings) -> SimConstants:  # type: ignore[valid-type]
    """Вычислить базовые константы симуляции из настроек.

    ВАЖНО: Награды и штрафы для reward shaping теперь определяются
    в модуле app.core.fitness (RewardConfig). Здесь остаются только
    параметры симуляции.

    Parameters
    ----------
    settings : app.utils.settings.Settings
        Parsed ``settings.ini`` wrapper.
    """
    field_size = settings.get_int("Field", "field_size")

    # --- ЭНЕРГИЯ ---
    # Запас энергии должен быть достаточным для пересечения поля с запасом на поиск.
    # Коэффициент 1.5 * field_size даёт достаточно времени на исследование.
    energy_max = int(field_size * 1.5)

    # --- ДВИЖЕНИЕ ---
    # Порог считывается напрямую из настроек для гибкости.
    # Добавлена отказоустойчивость на случай опечаток в конфиге.
    try:
        move_threshold = settings.get_float("Simulation", "move_threshold")
        move_threshold = max(0.0, min(1.0, move_threshold))  # Clamp
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        move_threshold = 0.1  # Fallback

    return SimConstants(
        energy_max=energy_max,
        move_threshold=move_threshold,
    )