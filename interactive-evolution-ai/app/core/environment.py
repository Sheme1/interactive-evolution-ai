"""Grid-based simulation environment (logic-only, no rendering).

The environment keeps track of *agents* and *food* on a square integer grid of
size ``field_size x field_size``. It performs collision resolution (food
consumption) and ensures entities stay within bounds. No Pygame or rendering
logic is included here — the :pyclass:`app.game.renderer.Renderer` is
responsible for visualisation.

Обновлено: детерминированная генерация окружения по seed, телепорты с
критерием удалённости, периодический респавн еды.
"""
from __future__ import annotations

import random
import math
from typing import Dict, List, Set, Optional, Tuple

from .agent import Agent, GridPos


class Environment:
    """Pure-logic environment operating on grid coordinates."""

    def __init__(
        self,
        field_size: int,
        food_quantity: int = 100,
        spawn_interval: int = 10,
        spawn_batch: int = 5,
        obstacles_percentage_str: str = "0%",
        teleporters_count: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        """Создать среду симуляции.

        Parameters
        ----------
        field_size : int
            Размер поля (квадрат field_size × field_size).
        food_quantity : int
            Начальное количество еды на поле.
        spawn_interval : int
            Интервал (в тиках) между респавнами еды.
        spawn_batch : int
            Количество еды, добавляемой за каждый респавн.
        obstacles_percentage_str : str
            Процент клеток, занятых препятствиями (например, "5%").
        teleporters_count : int
            Количество телепортов (должно быть чётным, так как они парные).
        seed : int, optional
            Seed для детерминированной генерации окружения. Если None,
            используется случайная генерация.
        """
        self.field_size: int = field_size
        self.agents: Dict[int, Agent] = {}
        self.food: Set[GridPos] = set()
        self.obstacles: Set[GridPos] = set()
        self.teleporters: Dict[GridPos, GridPos] = {}  # from -> to
        self._food_quantity = food_quantity
        self._spawn_interval = spawn_interval
        self._spawn_batch = spawn_batch
        self._ticks = 0
        self._obstacles_percentage_str = obstacles_percentage_str
        self._teleporters_count = teleporters_count
        self._current_seed = seed

        # Генерируем окружение
        self.reset(seed)

    # ------------------------------------------------------------------
    # Environment generation
    # ------------------------------------------------------------------
    def _generate_obstacles(self, percentage_str: str, rng: random.Random) -> None:
        """Сгенерировать препятствия на основе процента от площади поля.

        Parameters
        ----------
        percentage_str : str
            Процент клеток (например, "5%").
        rng : random.Random
            Генератор случайных чисел для детерминированной генерации.
        """
        try:
            percentage = float(percentage_str.strip().replace("%", ""))
        except (ValueError, TypeError):
            percentage = 0.0

        num_obstacles = int((self.field_size * self.field_size) * (percentage / 100.0))

        self.obstacles.clear()
        while len(self.obstacles) < num_obstacles:
            pos = (
                rng.randint(0, self.field_size - 1),
                rng.randint(0, self.field_size - 1),
            )
            if pos not in self.obstacles:
                self.obstacles.add(pos)

    def _generate_teleporters(self, count: int, rng: random.Random) -> None:
        """Сгенерировать парные телепорты с критерием удалённости.

        Телепорты генерируются парами, при этом расстояние между клетками
        пары должно быть > 0.5 × diagonal (диагональ поля).

        Parameters
        ----------
        count : int
            Количество телепортов (должно быть чётным).
        rng : random.Random
            Генератор случайных чисел.
        """
        if count <= 0 or count % 2 != 0:
            return  # Должно быть чётным и положительным

        self.teleporters.clear()
        occupied_tiles = self.obstacles.copy()
        num_pairs = count // 2

        # Критерий удалённости: > 0.5 × diagonal
        diagonal = math.sqrt(2 * self.field_size ** 2)
        min_distance = 0.5 * diagonal

        for _ in range(num_pairs):
            pos1, pos2 = self._find_distant_pair(rng, occupied_tiles, min_distance)
            if pos1 and pos2:
                self.teleporters[pos1] = pos2
                self.teleporters[pos2] = pos1
                occupied_tiles.add(pos1)
                occupied_tiles.add(pos2)

    # ------------------------------------------------------------------
    # Food management
    # ------------------------------------------------------------------
    def spawn_food(self, rng: Optional[random.Random] = None) -> None:
        """Распределить начальное количество еды на свободных клетках.

        Parameters
        ----------
        rng : random.Random, optional
            Генератор случайных чисел. Если None, используется глобальный random.
        """
        if rng is None:
            rng = random.Random()

        self.food.clear()
        occupied_tiles = self.obstacles.union(set(self.teleporters.keys()))
        while len(self.food) < self._food_quantity:
            pos = (
                rng.randint(0, self.field_size - 1),
                rng.randint(0, self.field_size - 1),
            )
            # Ensure no duplicates and not on obstacles/teleporters.
            if pos not in self.food and pos not in occupied_tiles:
                self.food.add(pos)

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------
    def is_occupied(self, pos: GridPos, exclude_id: Optional[int] = None) -> bool:
        """Return True if *pos* is occupied by an agent (optionally excluding one)."""
        for aid, a in self.agents.items():
            if exclude_id is not None and aid == exclude_id:
                continue
            if a.position == pos:
                return True
        return False

    def is_obstacle(self, pos: GridPos) -> bool:
        """Return True if *pos* is an obstacle."""
        return pos in self.obstacles

    def add_agent(self, agent: Agent) -> None:
        """Register *agent* in the environment."""
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id: int) -> None:
        self.agents.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Сбросить среду и сгенерировать новое окружение.

        Parameters
        ----------
        seed : int, optional
            Seed для детерминированной генерации. Если None, используется
            текущий seed или случайная генерация.
        """
        if seed is not None:
            self._current_seed = seed

        # Создаём генератор с заданным seed
        if self._current_seed is not None:
            rng = random.Random(self._current_seed)
        else:
            rng = random.Random()

        # Очищаем агентов
        self.agents.clear()

        # Генерируем окружение
        self._generate_obstacles(self._obstacles_percentage_str, rng)
        self._generate_teleporters(self._teleporters_count, rng)

        # Спавним еду
        self.spawn_food(rng)

        # Сбрасываем счётчик тиков
        self._ticks = 0

    def step(self) -> List[Agent]:
        """Advance the environment by **one** logical tick.

        Возвращает список агентов, которые съели хотя бы одну единицу еды на
        этом шаге. Дополнительно уменьшает энергию агентов, удаляет тех, кто
        умер от голода, и спавнит новую еду пакетами каждые *spawn_interval*
        тиков.
        """
        self._ticks += 1
        eaters: List[Agent] = []
        dead_ids: List[int] = []

        # --- обработка агентов ---
        for agent in list(self.agents.values()):
            # Уменьшаем энергию; смерть — позже, чтобы учесть поедание в этот же тик.
            agent.energy -= 1

            # Проверяем телепортацию.
            if agent.position in self.teleporters:
                target_pos = self.teleporters[agent.position]
                # Чтобы избежать мгновенной телепортации туда-обратно, проверяем, свободна ли цель.
                if not self.is_occupied(target_pos, exclude_id=agent.id):
                    agent.position = target_pos

            # Проверяем поедание еды.
            if agent.position in self.food:
                self.food.remove(agent.position)
                eaters.append(agent)
                agent.energy = Agent.ENERGY_MAX  # пополняем энергию

            # Проверяем смерть от голода.
            if agent.energy <= 0:
                dead_ids.append(agent.id)

        # Удаляем мёртвых агентов.
        for aid in dead_ids:
            self.remove_agent(aid)

        # --- нерегулярный спавн еды ---
        if self._ticks % self._spawn_interval == 0:
            for _ in range(self._spawn_batch):
                self._spawn_single_food()

        return eaters

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _spawn_single_food(self) -> None:
        """Spawn exactly one food pellet on a random empty tile."""
        occupied_static = self.obstacles.union(set(self.teleporters.keys()))
        attempts = 0
        while attempts < 10:  # Avoid infinite loops on crowded boards
            pos = (
                random.randint(0, self.field_size - 1),
                random.randint(0, self.field_size - 1),
            )
            occupied = (
                pos in self.food
                or any(a.position == pos for a in self.agents.values())
                or pos in occupied_static
            )
            if not occupied:
                self.food.add(pos)
                return
            attempts += 1
        # BUG FIX: В качестве запасного варианта, если не удалось найти
        # свободную клетку за 10 попыток, добавляем еду в последнюю
        # сгенерированную позицию, но только если она не занята статическим
        # объектом, другим агентом или другой едой.
        if (
            "pos" in locals()
            and pos not in occupied_static
            and pos not in self.food
            and not any(a.position == pos for a in self.agents.values())
        ):
            self.food.add(pos)

    def _find_distant_pair(
        self,
        rng: random.Random,
        occupied_tiles: Set[GridPos],
        min_distance: float
    ) -> Tuple[Optional[GridPos], Optional[GridPos]]:
        """Найти пару удалённых друг от друга свободных клеток.

        Parameters
        ----------
        rng : random.Random
            Генератор случайных чисел.
        occupied_tiles : Set[GridPos]
            Набор уже занятых клеток.
        min_distance : float
            Минимальное Евклидово расстояние между клетками пары.

        Returns
        -------
        Tuple[Optional[GridPos], Optional[GridPos]]
            Пара позиций, или (None, None) если не найдено.
        """
        max_attempts = 200
        best_pair = (None, None)
        best_distance = 0.0

        for _ in range(max_attempts):
            p1 = (rng.randint(0, self.field_size - 1), rng.randint(0, self.field_size - 1))
            p2 = (rng.randint(0, self.field_size - 1), rng.randint(0, self.field_size - 1))

            if p1 == p2 or p1 in occupied_tiles or p2 in occupied_tiles:
                continue

            # Вычисляем Евклидово расстояние
            distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

            # Если нашли пару с достаточной дистанцией, возвращаем её
            if distance > min_distance:
                return p1, p2

            # Иначе запоминаем лучшую найденную пару
            if distance > best_distance:
                best_distance = distance
                best_pair = (p1, p2)

        # Fallback: возвращаем лучшую найденную пару (или (None, None))
        return best_pair