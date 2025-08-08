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
    """Container for dynamically calculated constants."""

    energy_max: int  # Максимальный запас энергии
    move_threshold: float  # Порог чувствительности для движения
    tick_penalty: float  # Штраф за каждый тик существования
    eater_reward: float  # Награда за поедание еды
    death_penalty: float  # Штраф за смерть от голода
    collision_penalty: float  # Штраф за столкновение со стеной/агентом
    idle_penalty: float  # Штраф за бездействие
    food_proximity_reward: float  # Награда за приближение к еде
    food_proximity_penalty: float  # Штраф за отдаление от еды
    teleporter_proximity_reward: float  # Награда за приближение к телепорту
    teleporter_proximity_penalty: float  # Штраф за отдаление от телепорта


def compute_constants(settings) -> SimConstants:  # type: ignore[valid-type]
    """Derive *all* constants from the user-visible settings file.

    Система наград и штрафов спроектирована так, чтобы быть строгой и
    поощрять только эффективное поведение. Все параметры динамически
    масштабируются в зависимости от размера поля, чтобы поддерживать
    баланс сложности на разных конфигурациях.

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

    # --- НАГРАДЫ И ШТРАФЫ ---
    # 1. Основная награда за съеденную еду. Это главная цель.
    eater_reward = 10.0

    # 2. Суровый штраф за смерть. Он должен перевешивать награду от поедания
    #    одной единицы еды, чтобы отсеивать рискованные стратегии.
    death_penalty = eater_reward * 1.2

    # 3. Штраф за каждый тик жизни. Стимулирует эффективность. Рассчитан так,
    #    чтобы суммарный штраф за жизнь без еды (energy_max тиков) был
    #    ощутимой частью от награды за еду (здесь ~25%).
    #    Это заставляет агентов быть эффективными, а не просто блуждать.
    tick_penalty = eater_reward / (energy_max * 4)

    # 4. Штраф за бездействие. Должен быть небольшим, но больше, чем `tick_penalty`.
    idle_penalty = tick_penalty * 3

    # 5. Штраф за неудачное движение (коллизия). Заметное наказание за неверное решение.
    collision_penalty = tick_penalty * 10

    # 6. Награда/штраф за изменение дистанции до еды (reward shaping).
    #    Поощрение должно быть ощутимым, но меньше, чем штрафы, чтобы
    #    избежать кружения вокруг еды.
    food_proximity_reward = tick_penalty * 10.0
    food_proximity_penalty = tick_penalty * 5.0

    # 7. Награда за приближение к телепорту. Поощряет исследование карты.
    #    Должна быть меньше, чем награда за приближение к еде.
    #    Добавлен симметричный штраф для более стабильного поведения.
    teleporter_proximity_reward = tick_penalty * 4.0
    teleporter_proximity_penalty = tick_penalty * 2.0

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
        tick_penalty=tick_penalty,
        eater_reward=eater_reward,
        death_penalty=death_penalty,
        collision_penalty=collision_penalty,
        idle_penalty=idle_penalty,
        food_proximity_reward=food_proximity_reward,
        food_proximity_penalty=food_proximity_penalty,
        teleporter_proximity_reward=teleporter_proximity_reward,
        teleporter_proximity_penalty=teleporter_proximity_penalty,
    )