"""Управление процессом эволюции с 1v1 матчами и двумя популяциями.

Новая архитектура:
- Две независимые популяции NEAT (PopBLUE и PopRED)
- Оценка через 1v1 матчи (каждый геном против K случайных оппонентов)
- Potential-based reward shaping (Ng et al., 1999)
- Опциональная параллелизация через multiprocessing
"""
from __future__ import annotations

import random
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING, List, Tuple, Dict

import neat  # type: ignore

from .environment import Environment
from .agent import Agent
from .fitness import RewardConfig, PotentialShapingTracker, apply_base_rewards

if TYPE_CHECKING:
    from ..utils.settings import Settings


# ОПТИМИЗАЦИЯ: Глобальный кэш сетей для воркеров (локальный для каждого процесса)
# Ключ: genome.key, Значение: FeedForwardNetwork
_WORKER_NETWORK_CACHE: Dict[int, neat.nn.FeedForwardNetwork] = {}

# ОПТИМИЗАЦИЯ: Глобальный config для воркеров (инициализируется один раз)
_WORKER_CONFIG: Optional[neat.Config] = None


def _worker_init(config_path: str) -> None:
    """Инициализация воркера: загружаем config один раз для всех задач.

    ОПТИМИЗАЦИЯ: Вместо передачи огромного объекта config в каждую задачу,
    загружаем его один раз при старте воркера.

    Parameters
    ----------
    config_path : str
        Путь к файлу конфигурации NEAT.
    """
    global _WORKER_CONFIG, _WORKER_NETWORK_CACHE

    _WORKER_CONFIG = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Очищаем кэш сетей (на всякий случай)
    _WORKER_NETWORK_CACHE.clear()


def _get_or_create_network(genome: neat.DefaultGenome, config: neat.Config) -> neat.nn.FeedForwardNetwork:
    """Получить сеть из кэша или создать новую.

    КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: FeedForwardNetwork.create() - самая медленная операция
    в NEAT-Python. Кэшируем сети по genome.key, чтобы не пересоздавать их каждый раз.

    В однопоточном режиме (главный процесс) используется config напрямую.
    В многопоточном режиме (воркеры) используется глобальный _WORKER_CONFIG.

    Parameters
    ----------
    genome : neat.DefaultGenome
        Геном для которого нужна сеть.
    config : neat.Config
        Конфигурация NEAT (используется если _WORKER_CONFIG не инициализирован).

    Returns
    -------
    neat.nn.FeedForwardNetwork
        Нейросеть для данного генома.
    """
    # В воркере используем глобальный config
    actual_config = _WORKER_CONFIG if _WORKER_CONFIG is not None else config

    # Проверяем кэш
    genome_key = genome.key
    if genome_key not in _WORKER_NETWORK_CACHE:
        # Создаём сеть и кэшируем
        _WORKER_NETWORK_CACHE[genome_key] = neat.nn.FeedForwardNetwork.create(genome, actual_config)

    return _WORKER_NETWORK_CACHE[genome_key]


def _evaluate_genome_worker(task_data: Tuple) -> float:
    """Воркер для параллельной оценки одного генома (функция верхнего уровня для multiprocessing).

    ОПТИМИЗАЦИЯ: config больше не передаётся, используется глобальный _WORKER_CONFIG.

    Parameters
    ----------
    task_data : Tuple
        Кортеж с данными задачи: (gid, genome, opponents, team_main, team_opponent,
        base_seed, settings_dict, const_dict)

    Returns
    -------
    float
        Усреднённый фитнес генома по K матчам.
    """
    (
        gid,
        genome,
        opponents,
        team_main,
        team_opponent,
        base_seed,
        settings_dict,
        const_dict,
    ) = task_data

    # Играем K матчей
    fitness_sum = 0.0
    for match_idx, opponent_genome in enumerate(opponents):
        episode_seed = base_seed + gid * 100 + match_idx

        fitness_main, _ = _run_1v1_match_static(
            genome,
            opponent_genome,
            team_main,
            team_opponent,
            episode_seed,
            settings_dict,
            const_dict,
        )

        fitness_sum += fitness_main

    # Возвращаем усреднённый фитнес
    return fitness_sum / len(opponents)


def _evaluate_genome_batch_worker(batch_data: Tuple) -> List[float]:
    """Воркер для параллельной оценки batch-а геномов (ОПТИМИЗАЦИЯ для уменьшения overhead).

    КРИТИЧЕСКАЯ ОПТИМИЗАЦИЯ: Обрабатываем несколько геномов за раз, чтобы уменьшить
    overhead создания/уничтожения задач и лучше утилизировать кэш сетей.

    Parameters
    ----------
    batch_data : Tuple
        Кортеж с данными batch-а: (genome_tasks, team_main, team_opponent,
        base_seed, settings_dict, const_dict)
        где genome_tasks = [(gid, genome, opponents), ...]

    Returns
    -------
    List[float]
        Список усреднённых фитнесов для каждого генома в batch-е.
    """
    (
        genome_tasks,
        team_main,
        team_opponent,
        base_seed,
        settings_dict,
        const_dict,
    ) = batch_data

    results = []
    for gid, genome, opponents in genome_tasks:
        # Играем K матчей для этого генома
        fitness_sum = 0.0
        for match_idx, opponent_genome in enumerate(opponents):
            episode_seed = base_seed + gid * 100 + match_idx

            fitness_main, _ = _run_1v1_match_static(
                genome,
                opponent_genome,
                team_main,
                team_opponent,
                episode_seed,
                settings_dict,
                const_dict,
            )

            fitness_sum += fitness_main

        # Усредняем фитнес
        avg_fitness = fitness_sum / len(opponents)
        results.append(avg_fitness)

    return results


def _run_1v1_match_static(
    genome_a: neat.DefaultGenome,
    genome_b: neat.DefaultGenome,
    team_a: str,
    team_b: str,
    episode_seed: int,
    settings_dict: dict,
    const_dict: dict,
) -> Tuple[float, float]:
    """Статическая версия _run_1v1_match для использования в multiprocessing.

    ОПТИМИЗАЦИЯ: config убран из параметров, используется глобальный _WORKER_CONFIG.

    Parameters
    ----------
    genome_a, genome_b : neat.DefaultGenome
        Геномы агентов.
    team_a, team_b : str
        Команды агентов.
    episode_seed : int
        Seed эпизода.
    settings_dict : dict
        Словарь настроек (из Settings).
    const_dict : dict
        Словарь констант (из SimConstants).

    Returns
    -------
    Tuple[float, float]
        (fitness_a, fitness_b)
    """
    # Извлекаем параметры из словарей
    field_size = settings_dict["field_size"]
    food_quantity = settings_dict["food_quantity"]
    obstacles_percentage = settings_dict["obstacles_percentage"]
    teleporters_count = settings_dict["teleporters_count"]
    spawn_interval = settings_dict["respawn_interval"]
    spawn_batch = settings_dict["respawn_batch"]
    energy_max = const_dict["energy_max"]
    move_threshold = const_dict["move_threshold"]

    # Создаём окружение
    env = Environment(
        field_size=field_size,
        food_quantity=food_quantity,
        spawn_interval=spawn_interval,
        spawn_batch=spawn_batch,
        obstacles_percentage_str=obstacles_percentage,
        teleporters_count=teleporters_count,
        seed=episode_seed,
    )

    # ОПТИМИЗАЦИЯ: Используем кэшированные сети вместо пересоздания
    net_a = _get_or_create_network(genome_a, _WORKER_CONFIG)  # type: ignore
    net_b = _get_or_create_network(genome_b, _WORKER_CONFIG)  # type: ignore

    # Спавним агентов
    spawn_rng = random.Random(episode_seed + 1000)
    pos_a = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))
    pos_b = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))

    while pos_b == pos_a or env.is_obstacle(pos_a) or env.is_obstacle(pos_b):
        pos_a = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))
        pos_b = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))

    # Патчим ENERGY_MAX перед созданием агентов
    Agent.ENERGY_MAX = energy_max

    agent_a = Agent(id=1, team=team_a, position=pos_a, genome=genome_a, net=net_a, energy=energy_max)
    agent_b = Agent(id=2, team=team_b, position=pos_b, genome=genome_b, net=net_b, energy=energy_max)

    env.add_agent(agent_a)
    env.add_agent(agent_b)

    # Инициализируем reward shaping
    reward_config = RewardConfig.from_energy_max(energy_max)
    shaping_tracker = PotentialShapingTracker(reward_config)
    shaping_tracker.reset(env)

    # Инициализируем фитнес
    genome_a.fitness = 0.0
    genome_b.fitness = 0.0

    # Симуляция
    MAX_TICKS = int(energy_max * 1.5)

    for tick in range(MAX_TICKS):
        # Случайный порядок ходов
        agents_order = list(env.agents.values())
        random.shuffle(agents_order)

        # Действия агентов
        for agent in agents_order:
            old_pos = agent.position

            # Получаем действие
            obs = agent.get_observation(env)
            dx_raw, dy_raw = agent.net.activate(obs)

            # Дискретизация
            dx = -1 if dx_raw < -move_threshold else 1 if dx_raw > move_threshold else 0
            dy = -1 if dy_raw < -move_threshold else 1 if dy_raw > move_threshold else 0

            # Движение
            agent.move(dx, dy, field_size, env)

            # Базовые награды
            apply_base_rewards(
                agent,
                old_pos,
                (dx, dy),
                ate_food=False,  # Обновится ниже
                died=False,
                config=reward_config,
            )

        # Шаг среды
        eaters = env.step()

        # Награды за еду + инвалидация кэша при поедании
        if eaters:
            for eater in eaters:
                eater.genome.fitness += reward_config.eater_reward
            # ОПТИМИЗАЦИЯ: Инвалидируем кэш только когда еда съедена
            shaping_tracker.on_food_change(env)

        # Штраф за смерть
        if agent_a.id not in env.agents and not hasattr(agent_a, "_death_penalized"):
            agent_a.genome.fitness -= reward_config.death_penalty
            agent_a._death_penalized = True  # type: ignore
        if agent_b.id not in env.agents and not hasattr(agent_b, "_death_penalized"):
            agent_b.genome.fitness -= reward_config.death_penalty
            agent_b._death_penalized = True  # type: ignore

        # Potential shaping
        for agent in [agent_a, agent_b]:
            if agent.id in env.agents:
                shaping_reward = shaping_tracker.compute_shaping_reward(agent, env)
                agent.genome.fitness += shaping_reward

        # ОПТИМИЗАЦИЯ: Инвалидация кэша только при успешном спавне еды
        # (вместо каждого spawn_interval тика)

        # Условия завершения
        if not env.agents:
            break

    return (genome_a.fitness, genome_b.fitness)


class EvolutionManager:
    """Высокоуровневый интерфейс над NEAT-Python с 1v1 архитектурой."""

    def __init__(self, settings: "Settings", neat_config_path: str | Path) -> None:
        self._settings = settings
        self._neat_config_path = str(neat_config_path)  # Сохраняем для передачи воркерам
        self._config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self._neat_config_path,
        )

        # Обновляем размер популяции из settings.ini
        self._config.pop_size = settings.get_int("Simulation", "population_size")

        # Патчим размеры входов/выходов
        self._patch_io_sizes()

        # Вычисляем константы
        from ..utils.constants import compute_constants
        self._const = compute_constants(settings)
        Agent.ENERGY_MAX = self._const.energy_max

    def _patch_io_sizes(self) -> None:
        """Подстроить num_inputs/num_outputs под нашу среду."""
        # Создаём фиктивные объекты для определения размера
        dummy_env = Environment(field_size=10)
        dummy_agent = Agent(id=-1, team="DUMMY", position=(0, 0), genome=None, net=None)
        observation_size = len(dummy_agent.get_observation(dummy_env))

        genome_conf = self._config.genome_config
        genome_conf.num_inputs = observation_size
        genome_conf.num_outputs = 2

    def _get_adaptive_k(self, generation: int) -> int:
        """Получить адаптивное количество матчей для текущего поколения.

        ОПТИМИЗАЦИЯ: В ранних поколениях геномы случайные, K=5 матчей избыточно.
        Используем меньшее K для ускорения, увеличивая точность по мере эволюции.

        Parameters
        ----------
        generation : int
            Номер текущего поколения (0-indexed).

        Returns
        -------
        int
            Количество матчей для этого поколения.
        """
        if generation < 50:
            return 2  # Быстрая оценка для случайных геномов
        elif generation < 200:
            return 3  # Средняя точность
        else:
            # Полная точность для зрелых популяций
            return self._settings.get_int("Simulation", "matches_per_genome")

    def run_evolution(
        self,
        visualize: bool = False,
        continue_from: Optional[Tuple[neat.DefaultGenome, neat.DefaultGenome]] = None,
    ) -> None:
        """Запустить процесс эволюции с двумя популяциями.

        Parameters
        ----------
        visualize : bool
            Отрисовывать ли процесс обучения.
        continue_from : tuple, optional
            Пара геномов для дообучения (пока не реализовано).
        """
        if continue_from is not None:
            raise NotImplementedError("Дообучение с continue_from пока не реализовано в новой архитектуре")

        # Lazy импорты Rich - только для главного процесса
        from rich.console import Console
        from rich.live import Live
        from rich.table import Table
        from ..utils.console_utils import create_metrics_renderable

        console = Console()

        # Создаём две независимые популяции
        pop_blue = neat.Population(self._config)
        pop_red = neat.Population(self._config)

        # Добавляем репортёры
        pop_blue.add_reporter(neat.StatisticsReporter())
        pop_red.add_reporter(neat.StatisticsReporter())

        num_generations = self._settings.get_int("Simulation", "generations")

        console.print("\n[bold underline]Детали запуска эволюции (1v1 режим)[/]")
        info_table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        info_table.add_column(style="cyan")
        info_table.add_column()
        info_table.add_row("Размер популяции", str(self._config.pop_size))
        info_table.add_row("Количество поколений", str(num_generations))
        info_table.add_row("Входы нейросети", str(self._config.genome_config.num_inputs))
        info_table.add_row("Выходы нейросети", str(self._config.genome_config.num_outputs))
        info_table.add_row("Матчей на геном", str(self._settings.get_int("Simulation", "matches_per_genome")))
        console.print(info_table)
        console.print("[yellow]Запуск эволюции...[/]\n")

        # Базовый seed для эпизодов
        base_seed = random.randint(1, 1000000)
        console.print(f"[dim]Base seed: {base_seed}[/dim]\n")

        # Согласно PLAN.md: при visualize=True параллелизация отключается
        use_parallel = not visualize
        workers = self._settings.get_int("Simulation", "workers")

        # ОПТИМИЗАЦИЯ: Создаём Pool ОДИН РАЗ и переиспользуем между поколениями
        # Это экономит ~0.3-0.5 сек/поколение на создании/уничтожении процессов
        # ОПТИМИЗАЦИЯ 2: Используем initializer для загрузки config один раз в каждом воркере
        pool = None
        if use_parallel and workers > 1:
            # Передаём путь к конфигу вместо самого config (экономия на pickling)
            pool = mp.Pool(processes=workers, initializer=_worker_init, initargs=(self._neat_config_path,))
            console.print(f"[dim]Создан пул из {workers} воркеров для параллельных вычислений[/dim]")
            console.print(f"[dim]Конфиг воркеров: {self._neat_config_path}[/dim]")

        try:
            with Live(console=console, auto_refresh=False, transient=False) as live:
                for generation in range(num_generations):
                    # Оценка PopBLUE против PopRED
                    self._evaluate_population(
                        pop_blue,
                        pop_red,
                        "BLUE",
                        "RED",
                        base_seed + generation * 1000,
                        generation,
                        use_parallel=use_parallel,
                        visualize=visualize,
                        pool=pool,  # Передаём переиспользуемый pool
                    )

                    # Оценка PopRED против PopBLUE
                    self._evaluate_population(
                        pop_red,
                        pop_blue,
                        "RED",
                        "BLUE",
                        base_seed + generation * 1000 + 500,
                        generation,
                        use_parallel=use_parallel,
                        visualize=visualize,
                        pool=pool,  # Передаём переиспользуемый pool
                    )

                    # Метрики (ДО воспроизводства)
                    genomes_blue = list(pop_blue.population.values())
                    genomes_red = list(pop_red.population.values())

                    avg_blue = sum(g.fitness if g.fitness else 0.0 for g in genomes_blue) / len(genomes_blue)
                    avg_red = sum(g.fitness if g.fitness else 0.0 for g in genomes_red) / len(genomes_red)
                    best_blue = max((g.fitness if g.fitness else -float('inf') for g in genomes_blue), default=0.0)
                    best_red = max((g.fitness if g.fitness else -float('inf') for g in genomes_red), default=0.0)

                    avg_nodes_blue = sum(len(g.nodes) for g in genomes_blue) / len(genomes_blue)
                    avg_conns_blue = sum(len(g.connections) for g in genomes_blue) / len(genomes_blue)
                    avg_nodes_red = sum(len(g.nodes) for g in genomes_red) / len(genomes_red)
                    avg_conns_red = sum(len(g.connections) for g in genomes_red) / len(genomes_red)

                    # Выводим метрики (одинаково для обоих режимов)
                    renderable = create_metrics_renderable(
                        generation + 1,
                        avg_blue, avg_red,
                        best_blue, best_red,
                        avg_nodes_blue, avg_conns_blue,
                        avg_nodes_red, avg_conns_red
                    )
                    live.update(renderable, refresh=True)

                    # Обновляем species перед воспроизводством
                    pop_blue.species.speciate(self._config, pop_blue.population, generation)
                    pop_red.species.speciate(self._config, pop_red.population, generation)

                    # Воспроизводство
                    pop_blue.population = pop_blue.reproduction.reproduce(
                        self._config, pop_blue.species, self._config.pop_size, generation
                    )
                    pop_red.population = pop_red.reproduction.reproduce(
                        self._config, pop_red.species, self._config.pop_size, generation
                    )
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Тренировка прервана пользователем (Ctrl+C).[/bold yellow]")
            console.print("[dim]Сохранение лучших геномов текущего поколения...[/dim]")
        finally:
            # Закрываем pool при выходе (нормальном или через Ctrl+C)
            if pool:
                console.print("[dim]Закрытие пула воркеров...[/dim]")
                pool.close()
                pool.join()

        # Сохраняем лучшие геномы
        best_genome_blue = max(pop_blue.population.values(), key=lambda g: g.fitness if g.fitness else -float('inf'))
        best_genome_red = max(pop_red.population.values(), key=lambda g: g.fitness if g.fitness else -float('inf'))

        # Lazy импорт file_utils (содержит tkinter)
        from ..utils.file_utils import save_best_genomes
        output_path = save_best_genomes(best_genome_blue, best_genome_red)
        console.print(f"\n[bold green]Эволюция завершена.[/] Лучшие геномы сохранены в [cyan]{output_path}[/cyan]")

    def _evaluate_population(
        self,
        pop_main: neat.Population,
        pop_opponent: neat.Population,
        team_main: str,
        team_opponent: str,
        base_seed: int,
        generation: int,
        use_parallel: bool = True,
        visualize: bool = False,
        pool: Optional[mp.Pool] = None,
    ) -> None:
        """Оценить геномы одной популяции против другой через 1v1 матчи.

        Parameters
        ----------
        pop_main : neat.Population
            Популяция для оценки.
        pop_opponent : neat.Population
            Популяция оппонентов.
        team_main : str
            Название команды (BLUE или RED).
        team_opponent : str
            Название команды оппонентов.
        base_seed : int
            Базовый seed для эпизодов.
        generation : int
            Номер поколения.
        use_parallel : bool, optional
            Использовать ли параллелизацию (по умолчанию True).
        visualize : bool, optional
            Отрисовывать ли процесс обучения (по умолчанию False).
        pool : mp.Pool, optional
            Переиспользуемый пул воркеров (оптимизация).
        """
        genomes_main = list(pop_main.population.items())
        genomes_opponent = list(pop_opponent.population.values())

        # ОПТИМИЗАЦИЯ: Адаптивное K в зависимости от поколения
        K = self._get_adaptive_k(generation)

        # Решаем, использовать ли параллелизацию
        if use_parallel and pool is not None:
            self._evaluate_parallel(
                genomes_main, genomes_opponent, team_main, team_opponent, base_seed, K, pool
            )
        else:
            self._evaluate_sequential(
                genomes_main, genomes_opponent, team_main, team_opponent, base_seed, K, visualize, generation
            )

    def _evaluate_sequential(
        self,
        genomes_main: List[Tuple[int, neat.DefaultGenome]],
        genomes_opponent: List[neat.DefaultGenome],
        team_main: str,
        team_opponent: str,
        base_seed: int,
        K: int,
        visualize: bool = False,
        generation: int = 0,
    ) -> None:
        """Последовательная оценка геномов (однопоточный режим)."""
        # Инициализируем renderer только если нужна визуализация
        renderer = None
        if visualize:
            from ..game.renderer import Renderer
            field_size = self._settings.get_int("Field", "field_size")
            fps_setting = self._settings.get_int("Display", "fps")
            renderer = Renderer(field_size, cell_size=20, fps=fps_setting)

        total_genomes = len(genomes_main)
        for idx, (gid, genome) in enumerate(genomes_main):
            # Выбираем K случайных оппонентов
            opponents = random.sample(genomes_opponent, min(K, len(genomes_opponent)))

            if renderer:
                renderer.add_log(
                    f"Оценка генома {idx+1}/{total_genomes} команды {team_main}",
                    "GENERATION"
                )

            # Играем K матчей
            fitness_sum = 0.0
            for match_idx, opponent_genome in enumerate(opponents):
                episode_seed = base_seed + gid * 100 + match_idx

                fitness_main, fitness_opp = self._run_1v1_match(
                    genome, opponent_genome, team_main, team_opponent, episode_seed, renderer
                )

                fitness_sum += fitness_main

            # Усредняем фитнес
            genome.fitness = fitness_sum / len(opponents)

            if renderer:
                renderer.add_log(
                    f"Геном {idx+1}: средний фитнес = {genome.fitness:.2f}",
                    "INFO"
                )

    def _evaluate_parallel(
        self,
        genomes_main: List[Tuple[int, neat.DefaultGenome]],
        genomes_opponent: List[neat.DefaultGenome],
        team_main: str,
        team_opponent: str,
        base_seed: int,
        K: int,
        pool: mp.Pool,
    ) -> None:
        """Параллельная оценка геномов через multiprocessing.Pool.

        ОПТИМИЗАЦИЯ: Принимает переиспользуемый pool вместо создания нового.
        ОПТИМИЗАЦИЯ 2: config больше не передаётся (используется _WORKER_CONFIG).
        ОПТИМИЗАЦИЯ 3: Батчирование геномов для уменьшения overhead.
        """
        # Подготавливаем словари настроек для передачи воркерам
        settings_dict = {
            "field_size": self._settings.get_int("Field", "field_size"),
            "food_quantity": self._settings.get_int("Simulation", "food_quantity"),
            "obstacles_percentage": self._settings.get_str("Environment", "obstacles_percentage"),
            "teleporters_count": self._settings.get_int("Environment", "teleporters_count"),
            "respawn_interval": self._settings.get_int("Simulation", "respawn_interval"),
            "respawn_batch": self._settings.get_int("Simulation", "respawn_batch"),
        }

        const_dict = {
            "energy_max": self._const.energy_max,
            "move_threshold": self._const.move_threshold,
        }

        # ОПТИМИЗАЦИЯ 3: Батчируем геномы для уменьшения overhead
        # Оптимальный размер batch = количество геномов / (workers * 2)
        # Это даёт в 2 раза больше задач чем воркеров для лучшего баланса нагрузки
        workers = self._settings.get_int("Simulation", "workers")
        batch_size = max(1, len(genomes_main) // (workers * 2))

        # Группируем геномы в batch-и
        batches = []
        for i in range(0, len(genomes_main), batch_size):
            batch_genomes = genomes_main[i:i + batch_size]

            # Для каждого генома в batch-е подготавливаем данные
            genome_tasks = []
            for gid, genome in batch_genomes:
                # Выбираем K случайных оппонентов
                opponents = random.sample(genomes_opponent, min(K, len(genomes_opponent)))
                genome_tasks.append((gid, genome, opponents))

            # Создаём задачу для batch-а
            batches.append(
                (
                    genome_tasks,
                    team_main,
                    team_opponent,
                    base_seed,
                    settings_dict,
                    const_dict,
                )
            )

        # ОПТИМИЗАЦИЯ: Используем переданный pool вместо создания нового
        # Используем map_async для возможности прерывания
        async_result = pool.map_async(_evaluate_genome_batch_worker, batches)

        # Ждём результаты с короткими таймаутами для быстрой реакции на Ctrl+C
        while True:
            try:
                batch_results = async_result.get(timeout=1)  # Проверяем каждую секунду
                break
            except mp.TimeoutError:
                continue  # Продолжаем ждать

        # Собираем результаты из всех batch-ей и присваиваем фитнесы
        flat_results = []
        for batch_fitness_list in batch_results:
            flat_results.extend(batch_fitness_list)

        # Присваиваем вычисленные фитнесы геномам
        for (gid, genome), fitness in zip(genomes_main, flat_results):
            genome.fitness = fitness

    def _run_1v1_match(
        self,
        genome_a: neat.DefaultGenome,
        genome_b: neat.DefaultGenome,
        team_a: str,
        team_b: str,
        episode_seed: int,
        renderer=None,
    ) -> Tuple[float, float]:
        """Запустить один 1v1 матч между двумя геномами.

        Parameters
        ----------
        genome_a, genome_b : neat.DefaultGenome
            Геномы агентов.
        team_a, team_b : str
            Команды агентов.
        episode_seed : int
            Seed эпизода.
        renderer : Renderer, optional
            Объект для визуализации (если None, визуализация отключена).

        Returns
        -------
        Tuple[float, float]
            (fitness_a, fitness_b)
        """
        field_size = self._settings.get_int("Field", "field_size")
        food_quantity = self._settings.get_int("Simulation", "food_quantity")
        obstacles_percentage = self._settings.get_str("Environment", "obstacles_percentage")
        teleporters_count = self._settings.get_int("Environment", "teleporters_count")
        spawn_interval = self._settings.get_int("Simulation", "respawn_interval")
        spawn_batch = self._settings.get_int("Simulation", "respawn_batch")

        # Создаём окружение
        env = Environment(
            field_size=field_size,
            food_quantity=food_quantity,
            spawn_interval=spawn_interval,
            spawn_batch=spawn_batch,
            obstacles_percentage_str=obstacles_percentage,
            teleporters_count=teleporters_count,
            seed=episode_seed,
        )

        # ОПТИМИЗАЦИЯ: Используем кэшированные сети вместо пересоздания
        net_a = _get_or_create_network(genome_a, self._config)
        net_b = _get_or_create_network(genome_b, self._config)

        # Спавним агентов
        spawn_rng = random.Random(episode_seed + 1000)
        pos_a = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))
        pos_b = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))

        while pos_b == pos_a or env.is_obstacle(pos_a) or env.is_obstacle(pos_b):
            pos_a = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))
            pos_b = (spawn_rng.randint(0, field_size - 1), spawn_rng.randint(0, field_size - 1))

        agent_a = Agent(id=1, team=team_a, position=pos_a, genome=genome_a, net=net_a, energy=self._const.energy_max)
        agent_b = Agent(id=2, team=team_b, position=pos_b, genome=genome_b, net=net_b, energy=self._const.energy_max)

        env.add_agent(agent_a)
        env.add_agent(agent_b)

        # Логируем начальную информацию
        if renderer:
            renderer.add_log(f"Начало матча: {team_a} vs {team_b}", "GENERATION")
            renderer.add_log(f"Энергия агентов: {agent_a.energy} (max={self._const.energy_max})", "INFO")
            renderer.add_log(f"Порог движения: {self._const.move_threshold:.2f}", "INFO")

        # Инициализируем reward shaping
        reward_config = RewardConfig.from_energy_max(self._const.energy_max)
        shaping_tracker = PotentialShapingTracker(reward_config)
        shaping_tracker.reset(env)

        # Инициализируем фитнес
        genome_a.fitness = 0.0
        genome_b.fitness = 0.0

        # Симуляция
        MAX_TICKS = int(self._const.energy_max * 1.5)

        for tick in range(MAX_TICKS):
            # Случайный порядок ходов
            agents_order = list(env.agents.values())
            random.shuffle(agents_order)

            # Действия агентов
            for agent in agents_order:
                old_pos = agent.position

                # Получаем действие
                obs = agent.get_observation(env)
                dx_raw, dy_raw = agent.net.activate(obs)

                # Дискретизация
                threshold = self._const.move_threshold
                dx = -1 if dx_raw < -threshold else 1 if dx_raw > threshold else 0
                dy = -1 if dy_raw < -threshold else 1 if dy_raw > threshold else 0

                # Отладочное логирование для первых 5 тиков
                if renderer and tick < 5:
                    renderer.add_log(
                        f"Tick {tick}: {agent.team} NN out=({dx_raw:.2f}, {dy_raw:.2f}) -> dx={dx}, dy={dy}, E={agent.energy}",
                        "INFO"
                    )

                # Движение
                agent.move(dx, dy, field_size, env)

                # Логирование для визуализации
                if renderer:
                    if agent.position != old_pos:
                        renderer.add_log(f"Агент {agent.team} -> {agent.position}", "MOVE")
                    elif (dx, dy) != (0, 0):
                        renderer.add_log(f"Агент {agent.team} столкновение", "WARNING")

                # Базовые награды
                apply_base_rewards(
                    agent, old_pos, (dx, dy),
                    ate_food=False,  # Обновится ниже
                    died=False,
                    config=reward_config
                )

            # ОПТИМИЗАЦИЯ: Запоминаем количество еды до step для определения изменений
            food_count_before = len(env.food)

            # Шаг среды
            eaters = env.step()

            # Награды за еду + инвалидация кэша при изменении
            food_changed = False
            if eaters:
                for eater in eaters:
                    eater.genome.fitness += reward_config.eater_reward
                    if renderer:
                        renderer.add_log(f"Агент {eater.team} съел еду", "EAT")
                food_changed = True  # Еда съедена

            # Проверяем был ли респавн еды
            food_count_after = len(env.food)
            if food_count_after > food_count_before:
                food_changed = True  # Еда добавлена
                if renderer:
                    renderer.add_log(f"Новая еда (+{food_count_after - food_count_before} шт.)", "SPAWN")

            # ОПТИМИЗАЦИЯ: Инвалидируем кэш только когда еда реально изменилась
            if food_changed:
                shaping_tracker.on_food_change(env)

            # Штраф за смерть
            if agent_a.id not in env.agents and not hasattr(agent_a, "_death_penalized"):
                agent_a.genome.fitness -= reward_config.death_penalty
                agent_a._death_penalized = True  # type: ignore
                if renderer:
                    renderer.add_log(f"Агент {team_a} погиб", "DEATH")
            if agent_b.id not in env.agents and not hasattr(agent_b, "_death_penalized"):
                agent_b.genome.fitness -= reward_config.death_penalty
                agent_b._death_penalized = True  # type: ignore
                if renderer:
                    renderer.add_log(f"Агент {team_b} погиб", "DEATH")

            # Potential shaping
            for agent in [agent_a, agent_b]:
                if agent.id in env.agents:
                    shaping_reward = shaping_tracker.compute_shaping_reward(agent, env)
                    agent.genome.fitness += shaping_reward

            # Визуализация
            if renderer:
                renderer.draw_grid()
                renderer.draw_obstacles(env.obstacles)
                renderer.draw_teleporters(env.teleporters)
                renderer.draw_food(env.food)
                renderer.draw_agents(env.agents.values())
                if renderer.update(exit_on_esc=True):
                    # Пользователь нажал ESC, выходим из тренировки
                    from ..game.exceptions import UserQuitException
                    raise UserQuitException()

            # Условия завершения
            if not env.agents:
                break

        return (genome_a.fitness, genome_b.fitness)
