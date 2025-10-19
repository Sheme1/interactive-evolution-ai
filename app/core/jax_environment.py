"""JAX-совместимая версия окружения для TensorNEAT.

Эта версия окружения полностью написана на JAX и поддерживает JIT-компиляцию,
что необходимо для работы с TensorNEAT Pipeline. Основные отличия от обычного
Environment:
- Все операции выполняются через JAX (jax.numpy вместо numpy)
- Состояние окружения хранится в dataclass для функционального программирования
- Поддержка vmap для параллельной оценки популяции на GPU
"""
from __future__ import annotations
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
from jax import random
from functools import partial


class EnvState(NamedTuple):
    """Неизменяемое состояние окружения (для функционального стиля JAX)."""

    # Позиции агентов: shape (2, 2) - два агента с координатами (x, y)
    agent_positions: jnp.ndarray

    # Энергия агентов: shape (2,)
    agent_energies: jnp.ndarray

    # Жив ли агент: shape (2,)
    agent_alive: jnp.ndarray

    # Позиции еды: shape (max_food, 2)
    food_positions: jnp.ndarray

    # Маска активной еды: shape (max_food,) - True если еда существует
    food_active: jnp.ndarray

    # Позиции препятствий: shape (max_obstacles, 2)
    obstacles: jnp.ndarray

    # Количество препятствий
    num_obstacles: int

    # Позиции телепортов: shape (max_teleporters, 2)
    teleporter_positions: jnp.ndarray

    # Связи телепортов: shape (max_teleporters,) - индекс связанного телепорта
    teleporter_links: jnp.ndarray

    # Количество телепортов
    num_teleporters: int

    # Текущий тик симуляции
    tick: int

    # JAX random key для детерминизма
    rng_key: jnp.ndarray

    # Параметры окружения
    field_size: int
    energy_max: int
    move_threshold: float
    spawn_interval: int
    spawn_batch: int


class EnvConfig(NamedTuple):
    """Конфигурация окружения (неизменяемая)."""

    field_size: int = 32
    food_quantity: int = 100
    energy_max: int = 15
    move_threshold: float = 0.5
    spawn_interval: int = 10
    spawn_batch: int = 5
    obstacles_percentage: float = 0.05
    teleporters_count: int = 4
    max_ticks: int = 100


@partial(jax.jit, static_argnums=(1,))
def reset_env(rng_key: jnp.ndarray, config: EnvConfig) -> EnvState:
    """Сбросить окружение и сгенерировать начальное состояние.

    Parameters
    ----------
    rng_key : jnp.ndarray
        JAX random key для детерминированной генерации.
    config : EnvConfig
        Конфигурация окружения.

    Returns
    -------
    EnvState
        Начальное состояние окружения.
    """
    # Разделяем ключ для разных генераций
    key_obstacles, key_teleporters, key_food, key_agents, key_next = random.split(rng_key, 5)

    # Генерируем препятствия
    num_obstacles = int(config.field_size * config.field_size * config.obstacles_percentage)
    obstacles_coords = random.randint(
        key_obstacles,
        shape=(num_obstacles, 2),
        minval=0,
        maxval=config.field_size
    )

    # Генерируем телепорты (парами)
    num_teleporter_pairs = config.teleporters_count // 2
    teleporter_positions = random.randint(
        key_teleporters,
        shape=(config.teleporters_count, 2),
        minval=0,
        maxval=config.field_size
    )
    # Связываем телепорты попарно: 0<->1, 2<->3, и т.д.
    teleporter_links = jnp.array([1, 0, 3, 2] if config.teleporters_count >= 4 else [1, 0])

    # Генерируем еду
    max_food = config.food_quantity * 2  # Запас для респавна
    food_positions = random.randint(
        key_food,
        shape=(max_food, 2),
        minval=0,
        maxval=config.field_size
    )
    # Активируем только первые food_quantity единиц
    food_active = jnp.arange(max_food) < config.food_quantity

    # Спавним двух агентов в случайных позициях
    agent_positions = random.randint(
        key_agents,
        shape=(2, 2),
        minval=0,
        maxval=config.field_size
    )
    agent_energies = jnp.full(2, config.energy_max, dtype=jnp.int32)
    agent_alive = jnp.ones(2, dtype=jnp.bool_)

    return EnvState(
        agent_positions=agent_positions,
        agent_energies=agent_energies,
        agent_alive=agent_alive,
        food_positions=food_positions,
        food_active=food_active,
        obstacles=obstacles_coords,
        num_obstacles=num_obstacles,
        teleporter_positions=teleporter_positions,
        teleporter_links=teleporter_links,
        num_teleporters=config.teleporters_count,
        tick=0,
        rng_key=key_next,
        field_size=config.field_size,
        energy_max=config.energy_max,
        move_threshold=config.move_threshold,
        spawn_interval=config.spawn_interval,
        spawn_batch=config.spawn_batch,
    )


@jax.jit
def get_observation(state: EnvState, agent_idx: int) -> jnp.ndarray:
    """Получить эгоцентрическое наблюдение для агента.

    Возвращает окно 5x5 вокруг агента с 4 каналами:
    - Канал 0: Препятствия
    - Канал 1: Еда
    - Канал 2: Телепорты
    - Канал 3: Враги

    Parameters
    ----------
    state : EnvState
        Текущее состояние окружения.
    agent_idx : int
        Индекс агента (0 или 1).

    Returns
    -------
    jnp.ndarray
        Плоский вектор наблюдений shape (100,) = 5x5x4
    """
    window_size = 5
    half_window = window_size // 2

    # Инициализируем 4 канала
    observation = jnp.zeros((window_size, window_size, 4), dtype=jnp.float32)

    agent_pos = state.agent_positions[agent_idx]
    x, y = agent_pos[0], agent_pos[1]

    # Проходим по окну 5x5
    for dx in range(-half_window, half_window + 1):
        for dy in range(-half_window, half_window + 1):
            world_x = x + dx
            world_y = y + dy

            # Индексы в окне наблюдения
            obs_x = dx + half_window
            obs_y = dy + half_window

            # Проверяем границы поля
            in_bounds = (world_x >= 0) & (world_x < state.field_size) & \
                       (world_y >= 0) & (world_y < state.field_size)

            if not in_bounds:
                observation = observation.at[obs_x, obs_y, 0].set(1.0)  # Граница = препятствие
                continue

            # Канал 0: Препятствия
            is_obstacle = jnp.any(
                jnp.all(state.obstacles[:state.num_obstacles] == jnp.array([world_x, world_y]), axis=1)
            )
            observation = observation.at[obs_x, obs_y, 0].set(jnp.where(is_obstacle, 1.0, 0.0))

            # Канал 1: Еда
            is_food = jnp.any(
                jnp.all(
                    (state.food_positions == jnp.array([world_x, world_y])) &
                    state.food_active[:, None],
                    axis=1
                )
            )
            observation = observation.at[obs_x, obs_y, 1].set(jnp.where(is_food, 1.0, 0.0))

            # Канал 2: Телепорты
            is_teleporter = jnp.any(
                jnp.all(
                    state.teleporter_positions[:state.num_teleporters] == jnp.array([world_x, world_y]),
                    axis=1
                )
            )
            observation = observation.at[obs_x, obs_y, 2].set(jnp.where(is_teleporter, 1.0, 0.0))

            # Канал 3: Враги (другой агент)
            enemy_idx = 1 - agent_idx
            is_enemy = jnp.all(state.agent_positions[enemy_idx] == jnp.array([world_x, world_y])) & \
                      state.agent_alive[enemy_idx]
            observation = observation.at[obs_x, obs_y, 3].set(jnp.where(is_enemy, 1.0, 0.0))

    # Возвращаем плоский вектор
    return observation.reshape(-1)


@jax.jit
def step_env(state: EnvState, actions: jnp.ndarray) -> Tuple[EnvState, jnp.ndarray]:
    """Выполнить один шаг симуляции.

    Parameters
    ----------
    state : EnvState
        Текущее состояние окружения.
    actions : jnp.ndarray
        Действия агентов shape (2, 2) - (dx, dy) для каждого агента.

    Returns
    -------
    Tuple[EnvState, jnp.ndarray]
        Новое состояние окружения и награды для каждого агента shape (2,).
    """
    rewards = jnp.zeros(2, dtype=jnp.float32)
    new_positions = state.agent_positions.copy()
    new_energies = state.agent_energies.copy()
    new_alive = state.agent_alive.copy()
    new_food_active = state.food_active.copy()

    # Обрабатываем каждого агента
    for agent_idx in range(2):
        if not state.agent_alive[agent_idx]:
            continue

        # Дискретизация действий
        dx_raw, dy_raw = actions[agent_idx]
        dx = jnp.where(dx_raw < -state.move_threshold, -1,
                      jnp.where(dx_raw > state.move_threshold, 1, 0))
        dy = jnp.where(dy_raw < -state.move_threshold, -1,
                      jnp.where(dy_raw > state.move_threshold, 1, 0))

        # Новая позиция с учётом границ
        old_pos = state.agent_positions[agent_idx]
        new_x = jnp.clip(old_pos[0] + dx, 0, state.field_size - 1)
        new_y = jnp.clip(old_pos[1] + dy, 0, state.field_size - 1)
        new_pos = jnp.array([new_x, new_y])

        # Проверка на препятствия
        is_obstacle = jnp.any(
            jnp.all(state.obstacles[:state.num_obstacles] == new_pos, axis=1)
        )

        # Если не препятствие, обновляем позицию
        new_positions = jnp.where(
            is_obstacle,
            new_positions,
            new_positions.at[agent_idx].set(new_pos)
        )

        # Уменьшаем энергию
        new_energies = new_energies.at[agent_idx].add(-1)

        # Проверка на поедание еды
        final_pos = new_positions[agent_idx]
        food_eaten = jnp.any(
            jnp.all((state.food_positions == final_pos) & state.food_active[:, None], axis=1)
        )

        # Если съедена еда: пополняем энергию и деактивируем еду
        if food_eaten:
            new_energies = new_energies.at[agent_idx].set(state.energy_max)
            # Деактивируем съеденную еду
            food_mask = jnp.all((state.food_positions == final_pos) & state.food_active[:, None], axis=1)
            new_food_active = jnp.where(food_mask, False, new_food_active)
            rewards = rewards.at[agent_idx].add(10.0)  # Награда за еду

        # Проверка на смерть
        is_dead = new_energies[agent_idx] <= 0
        new_alive = new_alive.at[agent_idx].set(~is_dead)

        # Штраф за смерть
        rewards = jnp.where(is_dead, rewards.at[agent_idx].add(-5.0), rewards)

    # Респавн еды (каждые spawn_interval тиков)
    should_spawn = (state.tick % state.spawn_interval) == 0
    if should_spawn:
        # Простой респавн: активируем следующие неактивные единицы еды
        num_active = jnp.sum(new_food_active)
        for i in range(state.spawn_batch):
            idx = num_active + i
            if idx < len(new_food_active):
                new_food_active = new_food_active.at[idx].set(True)

    # Создаём новое состояние
    new_state = state._replace(
        agent_positions=new_positions,
        agent_energies=new_energies,
        agent_alive=new_alive,
        food_active=new_food_active,
        tick=state.tick + 1,
    )

    return new_state, rewards


@jax.jit
def is_done(state: EnvState, max_ticks: int) -> bool:
    """Проверить, завершена ли симуляция.

    Parameters
    ----------
    state : EnvState
        Текущее состояние окружения.
    max_ticks : int
        Максимальное количество тиков.

    Returns
    -------
    bool
        True если симуляция завершена.
    """
    # Завершаем если оба агента мертвы или достигнут лимит тиков
    all_dead = ~jnp.any(state.agent_alive)
    time_limit = state.tick >= max_ticks

    return all_dead | time_limit
