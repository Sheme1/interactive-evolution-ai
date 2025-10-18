"""Вспомогательный модуль для параллельной оценки 1v1 матчей.

Этот модуль содержит функцию, которая выполняется в отдельном процессе
через multiprocessing.Pool. Она должна быть определена на уровне модуля,
чтобы её можно было pickle для передачи между процессами.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Tuple

import neat  # type: ignore

if TYPE_CHECKING:
    from ..utils.settings import Settings


def evaluate_1v1_match(
    genome_a_data: Tuple[int, object, str],  # (key, genome, team)
    genome_b_data: Tuple[int, object, str],
    config_dict: dict,
    settings_dict: dict,
    episode_seed: int,
) -> Tuple[int, float, int, float]:
    """Оценить один 1v1 матч между двумя геномами.

    Функция выполняется в отдельном процессе через multiprocessing.Pool.

    Parameters
    ----------
    genome_a_data : Tuple[int, object, str]
        Данные первого генома: (key, genome, team).
    genome_b_data : Tuple[int, object, str]
        Данные второго генома: (key, genome, team).
    config_dict : dict
        Словарь с настройками NEAT (сериализованными).
    settings_dict : dict
        Словарь с настройками симуляции.
    episode_seed : int
        Seed для детерминированной генерации эпизода.

    Returns
    -------
    Tuple[int, float, int, float]
        (genome_a_key, fitness_a, genome_b_key, fitness_b)
    """
    from .environment import Environment
    from .agent import Agent
    from .fitness import RewardConfig, PotentialShapingTracker, apply_base_rewards

    # Распаковываем данные
    key_a, genome_a, team_a = genome_a_data
    key_b, genome_b, team_b = genome_b_data

    # Восстанавливаем NEAT конфиг (здесь передаём упрощённую версию)
    # В реальности нужно будет передавать полный config
    config = config_dict  # Это будет neat.Config object

    # Параметры симуляции
    field_size = settings_dict["field_size"]
    food_quantity = settings_dict["food_quantity"]
    obstacles_percentage = settings_dict["obstacles_percentage"]
    teleporters_count = settings_dict["teleporters_count"]
    spawn_interval = settings_dict.get("spawn_interval", 10)
    spawn_batch = settings_dict.get("spawn_batch", 3)
    energy_max = settings_dict["energy_max"]
    move_threshold = settings_dict["move_threshold"]

    # Создаём окружение с заданным seed
    env = Environment(
        field_size=field_size,
        food_quantity=food_quantity,
        spawn_interval=spawn_interval,
        spawn_batch=spawn_batch,
        obstacles_percentage_str=obstacles_percentage,
        teleporters_count=teleporters_count,
        seed=episode_seed,
    )

    # Обновляем energy_max для агентов
    Agent.ENERGY_MAX = energy_max

    # Создаём сети для геномов
    net_a = neat.nn.FeedForwardNetwork.create(genome_a, config)
    net_b = neat.nn.FeedForwardNetwork.create(genome_b, config)

    # Создаём агентов
    # Спавним в случайных позициях (используя тот же seed для детерминизма)
    spawn_rng = random.Random(episode_seed + 1000)
    pos_a = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))
    pos_b = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))

    # Убеждаемся, что позиции разные
    while pos_b == pos_a or env.is_obstacle(pos_a) or env.is_obstacle(pos_b):
        pos_a = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))
        pos_b = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))

    agent_a = Agent(id=key_a, team=team_a, position=pos_a, genome=genome_a, net=net_a)
    agent_b = Agent(id=key_b, team=team_b, position=pos_b, genome=genome_b, net=net_b)

    env.add_agent(agent_a)
    env.add_agent(agent_b)

    # Инициализируем reward shaping
    reward_config = RewardConfig.from_energy_max(energy_max)
    shaping_tracker = PotentialShapingTracker(reward_config)
    shaping_tracker.reset(env)

    # Инициализируем фитнес
    genome_a.fitness = 0.0
    genome_b.fitness = 0.0

    # Симуляция матча
    MAX_TICKS = int(energy_max * 1.5)

    for tick in range(MAX_TICKS):
        # Случайный порядок ходов
        agents_order = [agent_a, agent_b]
        random.shuffle(agents_order)

        # Действия агентов
        for agent in agents_order:
            if agent.id not in env.agents:
                continue  # Мёртв

            old_pos = agent.position

            # Получаем действие от сети
            obs = agent.get_observation(env)
            dx_raw, dy_raw = agent.net.activate(obs)

            # Дискретизация
            dx = -1 if dx_raw < -move_threshold else 1 if dx_raw > move_threshold else 0
            dy = -1 if dy_raw < -move_threshold else 1 if dy_raw > move_threshold else 0

            # Движение
            agent.move(dx, dy, field_size, env)

            # Применяем базовые награды и штрафы
            apply_base_rewards(
                agent,
                old_pos,
                (dx, dy),
                ate_food=False,  # Будет обновлено ниже
                died=False,
                config=reward_config
            )

        # Шаг среды
        eaters = env.step()

        # Награды за еду
        for eater in eaters:
            eater.genome.fitness += reward_config.eater_reward

        # Штраф за смерть
        if agent_a.id not in env.agents and not hasattr(agent_a, "_death_penalized"):
            agent_a.genome.fitness -= reward_config.death_penalty
            agent_a._death_penalized = True  # type: ignore
        if agent_b.id not in env.agents and not hasattr(agent_b, "_death_penalized"):
            agent_b.genome.fitness -= reward_config.death_penalty
            agent_b._death_penalized = True  # type: ignore

        # Potential-based shaping
        for agent in [agent_a, agent_b]:
            if agent.id in env.agents:
                shaping_reward = shaping_tracker.compute_shaping_reward(agent, env)
                agent.genome.fitness += shaping_reward

        # Обновление потенциалов при респавне еды
        if tick > 0 and tick % spawn_interval == 0:
            shaping_tracker.on_food_spawn(env)

        # Проверка условий завершения
        if not env.agents:
            break  # Все мертвы

    return (key_a, genome_a.fitness, key_b, genome_b.fitness)
