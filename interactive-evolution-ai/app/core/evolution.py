"""Управление процессом эволюции (NEAT-Python).

Класс ``EvolutionManager`` инкапсулирует создание NEAT ``Population``,
запуск цикла обучения и сохранение лучших геномов.
"""
from __future__ import annotations

import random
from pathlib import Path
from math import hypot
from typing import Callable, Tuple, Optional, TYPE_CHECKING

import neat  # type: ignore
from rich.console import Console
from rich.live import Live
from rich.table import Table

from ..utils.file_utils import save_best_genomes
from ..utils.console_utils import create_metrics_renderable
from .environment import Environment
from .agent import Agent

if TYPE_CHECKING:
    from ..utils.settings import Settings


class EvolutionManager:  # pylint: disable=too-many-instance-attributes
    """Высокоуровневый интерфейс над NEAT-Python."""

    def __init__(self, settings: "Settings", neat_config_path: str | Path) -> None:  # noqa: F821
        self._settings = settings
        # --- Динамические константы, зависящие от настроек пользователя ---
        from ..utils.constants import compute_constants
        self._const = compute_constants(settings)
        # Обновляем запас энергии во всех экземплярах Agent
        from .agent import Agent
        Agent.ENERGY_MAX = self._const.energy_max
        self._config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(neat_config_path),
        )
        # Обновляем размер популяции из settings.ini
        self._config.pop_size = settings.get_int("Simulation", "population_size")

        # Эти параметры будут переопределены в ``_patch_io_sizes``
        self._patch_io_sizes()

        self._console = Console()
        # Объект статистики будет создан в ``run_evolution``
        self._stats: neat.StatisticsReporter | None = None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def run_evolution(
        self,
        visualize: bool,
        continue_from: Optional[Tuple[neat.DefaultGenome, neat.DefaultGenome]] = None,
    ) -> None:
        """Запустить процесс эволюции.

        Parameters
        ----------
        visualize: bool
            Отрисовывать ли процесс обучения.
        continue_from: tuple | None
            Пара предварительно обученных геномов (A, B) для дообучения.
        """
        population = neat.Population(self._config)
 
        if continue_from is not None:
            genome_a, genome_b = continue_from
                        # The population is created with pop_size genomes with keys 1..pop_size.
            # We replace the first two genomes with our loaded ones. To do this
            # cleanly, we re-key our loaded genomes to match the keys of the
            # genomes they are replacing (1 and 2).
            genome_a.key = 1
            population.population[1] = genome_a

            genome_b.key = 2
            population.population[2] = genome_b

            # --- Fix for continuing evolution ---
            # When loading genomes, the node key counter in the new config is not
            # aware of the existing node keys. We must manually find the maximum
            # existing node key and initialize the counter to avoid key collisions.
            from itertools import count
            max_node_key = 0
            for g in population.population.values():
                if g.nodes:
                    max_node_key = max(max_node_key, max(g.nodes.keys()))
            self._config.genome_config.node_indexer = count(max_node_key + 1)

            # After manually modifying the population, we must re-speciate to ensure
            # the species set is consistent with the new population.
            population.species.speciate(population.config, population.population, population.generation)

        # Crucial fix: Ensure all genomes have their fitness initialized to a numeric value
        # before the evolution starts. New genomes are created with fitness=None, and
        # NEAT's stagnation logic fails if it encounters None.
        for g in population.population.values():
            g.fitness = 0.0

        # Репортеры (консольный вывод и статистика)
        # Создаём репортёр статистики, чтобы знать номер поколения
        self._stats = neat.StatisticsReporter()
        population.add_reporter(self._stats)

        # Определяем количество поколений в зависимости от режима
        if continue_from:
            num_generations = self._settings.get_int("Simulation", "continue_generations")
        else:
            num_generations = self._settings.get_int("Simulation", "generations")

        self._console.print("\n[bold underline]Детали запуска эволюции[/]")
        info_table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        info_table.add_column(style="cyan")
        info_table.add_column()
        info_table.add_row("Режим", "Тренировка с нуля" if continue_from is None else "Дообучение")
        info_table.add_row("Размер популяции", str(self._config.pop_size))
        info_table.add_row("Количество поколений", str(num_generations))
        info_table.add_row("Входы нейросети", str(self._config.genome_config.num_inputs))
        info_table.add_row("Выходы нейросети", str(self._config.genome_config.num_outputs))
        info_table.add_row("Порог движения", f"{self._const.move_threshold:.2f}")
        self._console.print(info_table)
        self._console.print("[yellow]Запуск эволюции...[/]\n")

        with Live(console=self._console, auto_refresh=False, transient=False) as live:
            # Передаём `live` в `eval_genomes` через замыкание,
            # чтобы обновлять таблицу метрик на месте, не очищая консоль.
            eval_fn = self._make_eval_genomes(visualize, live)
            winner = population.run(eval_fn, n=num_generations)

        # Сохранение лучшего генома (победителя). В текущей версии NEAT-Python
        # функция `Population.run` возвращает один геном, поэтому сохраняем один
        # и тот же геном для обеих команд, чтобы сохранить совместимость
        # с текущей реализацией `save_best_genomes`.
        output_path = save_best_genomes(winner, winner)
        self._console.print(f"\n[bold green]Эволюция завершена.[/] Лучшие геномы сохранены в [cyan]{output_path}[/cyan]")

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------
    def _patch_io_sizes(self) -> None:
        """Подстроить ``num_inputs`` / ``num_outputs`` под нашу среду.

        Мы знаем, что :pyclass:`Agent.get_observation` возвращает вектор
        наблюдений (векторы до объектов + локальные сенсоры), а действие —
        это (dx, dy) → 2 выхода.
        """
        genome_conf = self._config.genome_config
        # 4 вектора (x,y) до еды/врага/препятствия/телепорта + 8 сенсоров окружения
        genome_conf.num_inputs = 16
        genome_conf.num_outputs = 2

    def _make_eval_genomes(
        self, visualize: bool, live: Optional[Live] = None
    ) -> Callable[[list, neat.Config], None]:
        """Создать замыкание функции ``eval_genomes`` для NEAT."""

        field_size = self._settings.get_int("Field", "field_size")
        food_quantity = self._settings.get_int("Simulation", "food_quantity")
        food_respawn = self._settings.get_bool("Simulation", "food_respawn")
        obstacles_percentage = self._settings.get_str("Environment", "obstacles_percentage")
        teleporters_count = self._settings.get_int("Environment", "teleporters_count")
        # Длина эпизода будет увеличиваться по мере роста поколения
        # Предел по шагам больше не используется – поколение завершается,
        # когда закончится пища **или** умрут все агенты.
        # Оставляем переменные для возможного future-use but not used.

        generation_counter = 0  # счётчик поколений внутри eval_genomes замыканием
        renderer = None  # Renderer будет создан при первом вызове, затем переиспользован

        def eval_genomes(genomes, config):  # noqa: N803 – наименование по NEAT API
            nonlocal generation_counter

            if food_respawn:
                env = Environment(
                    field_size,
                    food_quantity,
                    obstacles_percentage_str=obstacles_percentage,
                    teleporters_count=teleporters_count,
                )
            else:
                env = Environment(
                    field_size,
                    food_quantity,
                    spawn_interval=999_999,
                    spawn_batch=0,
                    obstacles_percentage_str=obstacles_percentage,
                    teleporters_count=teleporters_count,
                )
            # Инициализируем визуализатор **один раз** и переиспользуем между поколениями,
            # чтобы избежать повторных вызовов `pygame.display.set_mode`, которые приводят
            # к постепенному уменьшению окна при флаге SCALED (известный баг SDL/Pygame).
            nonlocal renderer
            if visualize and renderer is None:
                from ..game.renderer import Renderer  # локальный импорт чтобы избежать pygame при headless

                fps_setting = self._settings.get_int("Display", "fps")
                renderer = Renderer(field_size, cell_size=20, fps=fps_setting)

            if visualize and renderer:
                renderer.add_log(f"Поколение {generation_counter + 1}", "GENERATION")

            # --- Групповая оценка: все агенты на одном поле ---
            # Берём ВСЕ геномы поколения и создаём для каждого по агенту.
            # Чётные индексы → команда BLUE, нечётные → команда RED (для визуализации).
            agents = []
            for idx, (gid, genome) in enumerate(genomes):
                # Обнуляем фитнес перед стартом эпизода
                genome.fitness = 0.0
                team = "BLUE" if idx % 2 == 0 else "RED"
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                # Гарантируем уникальную клетку при спавне
                while True:
                    spawn_pos = (
                        random.randint(0, field_size - 1),
                        random.randint(0, field_size - 1),
                    )
                    if not env.is_occupied(spawn_pos):
                        break

                agent = Agent(
                    id=gid,
                    team=team,
                    position=spawn_pos,
                    genome=genome,
                    net=net,
                )
                env.add_agent(agent)
                agents.append(agent)

            # --- Инициализация для reward shaping ---
            # Словарь для хранения предыдущего расстояния до ближайшей цели
            # для каждого агента. Это нужно для "reward shaping" - поощрения
            # за приближение к цели.
            prev_dist_to_food = {}
            if env.food:
                for agent in agents:
                    x, y = agent.position
                    nearest_food_pos = min(env.food, key=lambda p: hypot(p[0] - x, p[1] - y))
                    prev_dist_to_food[agent.id] = hypot(nearest_food_pos[0] - x, nearest_food_pos[1] - y)

            prev_dist_to_teleporter = {}
            if env.teleporters:
                for agent in agents:
                    x, y = agent.position
                    nearest_tele_pos = min(env.teleporters.keys(), key=lambda p: hypot(p[0] - x, p[1] - y))
                    prev_dist_to_teleporter[agent.id] = hypot(
                        nearest_tele_pos[0] - x, nearest_tele_pos[1] - y
                    )

            # Определяем максимальное число тиков одного поколения.
            # Должно быть достаточно, чтобы агенты умерли от голода, если не найдут еду.
            MAX_TICKS = int(self._const.energy_max * 1.5)
            for _ in range(MAX_TICKS):
                # Создаём копию списка живых агентов и перемешиваем её, чтобы
                # порядок действий на каждом тике был случайным. Это устраняет
                # систематическое преимущество у агентов, добавленных в среду
                # первыми (в данном случае — у команды BLUE).
                shuffled_agents = list(env.agents.values())
                random.shuffle(shuffled_agents)
                # --- Действия агентов и штрафы за неэффективность ---
                for agent in shuffled_agents:  # Итерируем по живым агентам
                    old_pos = agent.position
                    dx_raw, dy_raw = agent.net.activate(agent.get_observation(env))

                    threshold = self._const.move_threshold
                    dx = -1 if dx_raw < -threshold else 1 if dx_raw > threshold else 0
                    dy = -1 if dy_raw < -threshold else 1 if dy_raw > threshold else 0
                    agent.move(dx, dy, field_size, env)

                    if visualize and renderer:
                        if agent.position != old_pos:
                            renderer.add_log(f"Агент {agent.id} -> {agent.position}", "MOVE")

                    # Применение штрафов за неэффективные действия
                    if agent.position == old_pos:
                        if (dx, dy) != (0, 0):
                            # Пытался сдвинуться, но не смог (коллизия)
                            agent.genome.fitness -= self._const.collision_penalty
                            if visualize and renderer:
                                renderer.add_log(f"Агент {agent.id} столкновение", "WARNING")
                        else:
                            # Сознательно остался на месте (бездействие)
                            agent.genome.fitness -= self._const.idle_penalty
                            if visualize and renderer:
                                renderer.add_log(f"Агент {agent.id} бездействие", "INFO")

                # --- Reward shaping: поощрение за приближение к целям ---
                # 1. Еда (основная цель)
                if env.food:
                    x, y = agent.position
                    nearest_food_pos = min(env.food, key=lambda p: hypot(p[0] - x, p[1] - y))
                    current_dist = hypot(nearest_food_pos[0] - x, nearest_food_pos[1] - y)
                    last_dist = prev_dist_to_food.get(agent.id, current_dist)

                    if current_dist < last_dist:
                        agent.genome.fitness += self._const.food_proximity_reward
                    elif current_dist > last_dist:
                        agent.genome.fitness -= self._const.food_proximity_penalty
                    prev_dist_to_food[agent.id] = current_dist

                # 2. Телепорты (вторичная цель для исследования)
                if env.teleporters:
                    x, y = agent.position
                    nearest_tele_pos = min(
                        env.teleporters.keys(), key=lambda p: hypot(p[0] - x, p[1] - y)
                    )
                    current_dist = hypot(nearest_tele_pos[0] - x, nearest_tele_pos[1] - y)
                    last_dist = prev_dist_to_teleporter.get(agent.id, current_dist)

                    if current_dist < last_dist:
                        agent.genome.fitness += self._const.teleporter_proximity_reward
                    elif current_dist > last_dist:
                        agent.genome.fitness -= self._const.teleporter_proximity_penalty
                    prev_dist_to_teleporter[agent.id] = current_dist

                # --- Шаг среды и награды за еду ---
                eaters = env.step()
                if visualize and renderer:
                    for eater in eaters:
                        renderer.add_log(f"Агент {eater.id} съел еду", "EAT")

                    # Логирование спавна еды
                    if food_respawn and env._ticks % env._spawn_interval == 0 and env._spawn_batch > 0:
                        renderer.add_log(f"Новая еда ({env._spawn_batch} шт.)", "SPAWN")

                # После спавна новой еды пересчитываем базовые расстояния до неё,
                # чтобы избежать ложных градиентов в reward-shaping.
                if food_respawn and env._ticks % env._spawn_interval == 0 and env._spawn_batch > 0:
                    if env.food:
                        for agent in env.agents.values():
                            x, y = agent.position
                            nearest_food_pos = min(env.food, key=lambda p: hypot(p[0] - x, p[1] - y))
                            prev_dist_to_food[agent.id] = hypot(nearest_food_pos[0] - x, nearest_food_pos[1] - y)

                for eater in eaters:
                    eater.genome.fitness += self._const.eater_reward

                # --- Штрафы за смерть и за тик ---
                # `agents` - это полный список агентов поколения.
                # `env.agents` - словарь только живых агентов.
                for agent in agents: # type: ignore
                    is_alive = agent.id in env.agents
                    if is_alive:
                        # Применяем штраф за тик к живым
                        agent.genome.fitness -= self._const.tick_penalty
                    elif not hasattr(agent, "_death_penalised"):
                        # Применяем штраф за смерть (однократно)
                        agent.genome.fitness -= self._const.death_penalty
                        agent._death_penalised = True
                        if visualize and renderer:
                            renderer.add_log(f"Агент {agent.id} погиб", "DEATH")

                # Визуализация
                if visualize and renderer:
                    renderer.draw_grid()
                    renderer.draw_obstacles(env.obstacles)
                    renderer.draw_teleporters(env.teleporters)
                    renderer.draw_food(env.food)
                    renderer.draw_agents(list(env.agents.values()))
                    renderer.update()

                # --- Условие досрочного завершения поколения ---
                if not env.agents:
                    if visualize and renderer:
                        renderer.add_log("Все агенты погибли.", "GENERATION")
                    break
                if not food_respawn and not env.food:
                    if visualize and renderer:
                        renderer.add_log("Вся еда съедена.", "GENERATION")
                    break

            # Метрики поколения
            genomes_a = [g for idx, (_, g) in enumerate(genomes) if idx % 2 == 0]
            genomes_b = [g for idx, (_, g) in enumerate(genomes) if idx % 2 == 1]
            avg_a = sum(g.fitness for g in genomes_a) / max(1, len(genomes_a))
            avg_b = sum(g.fitness for g in genomes_b) / max(1, len(genomes_b))
            best_a = max((g.fitness for g in genomes_a), default=0.0)
            best_b = max((g.fitness for g in genomes_b), default=0.0)

            # Статистика сложности сети
            if genomes_a:
                avg_nodes_a = sum(len(g.nodes) for g in genomes_a) / len(genomes_a)
                avg_conns_a = sum(len(g.connections) for g in genomes_a) / len(genomes_a)
            else:
                avg_nodes_a = 0.0
                avg_conns_a = 0.0

            if genomes_b:
                avg_nodes_b = sum(len(g.nodes) for g in genomes_b) / len(genomes_b)
                avg_conns_b = sum(len(g.connections) for g in genomes_b) / len(genomes_b)
            else:
                avg_nodes_b = 0.0
                avg_conns_b = 0.0
            
            renderable = create_metrics_renderable(
                generation_counter,
                avg_a,
                avg_b,
                best_a,
                best_b,
                avg_nodes_a,
                avg_conns_a,
                avg_nodes_b,
                avg_conns_b,
            )
            if live:
                live.update(renderable, refresh=True)

            generation_counter += 1



        return eval_genomes