"""Grid-based simulation environment (logic-only, no rendering).

The environment keeps track of *agents* and *food* on a square integer grid of
size ``field_size x field_size``. It performs collision resolution (food
consumption) and ensures entities stay within bounds. No Pygame or rendering
logic is included here — the :pyclass:`app.game.renderer.Renderer` is
responsible for visualisation.
"""
from __future__ import annotations

import random
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
    ) -> None:
        self.field_size: int = field_size
        self.agents: Dict[int, Agent] = {}
        self.food: Set[GridPos] = set()
        self.obstacles: Set[GridPos] = set()
        self.teleporters: Dict[GridPos, GridPos] = {}  # from -> to
        self._food_quantity = food_quantity
        self._spawn_interval = spawn_interval
        self._spawn_batch = spawn_batch
        self._ticks = 0

        self._generate_obstacles(obstacles_percentage_str)
        self._generate_teleporters(teleporters_count)

        # Spawn initial food pellets.
        self.spawn_food()

    # ------------------------------------------------------------------
    # Environment generation
    # ------------------------------------------------------------------
    def _generate_obstacles(self, percentage_str: str) -> None:
        """Generate obstacle positions based on a percentage of the field area."""
        try:
            percentage = float(percentage_str.strip().replace("%", ""))
        except (ValueError, TypeError):
            percentage = 0.0

        num_obstacles = int((self.field_size * self.field_size) * (percentage / 100.0))

        # Use a seeded RNG for deterministic obstacle placement across runs
        placer_rng = random.Random(self.field_size)
        while len(self.obstacles) < num_obstacles:
            pos = (
                placer_rng.randint(0, self.field_size - 1),
                placer_rng.randint(0, self.field_size - 1),
            )
            if pos not in self.obstacles:
                self.obstacles.add(pos)

    def _generate_teleporters(self, count: int) -> None:
        """Generate fixed, paired teleporter locations."""
        if count <= 0 or count % 2 != 0:
            return  # Must be an even, positive number

        # Use a seeded RNG for deterministic placement
        placer_rng = random.Random(self.field_size + 1)  # Different seed from obstacles
        occupied_tiles = self.obstacles.copy()
        num_pairs = count // 2

        for _ in range(num_pairs):
            pos1, pos2 = self._find_unoccupied_pair(placer_rng, occupied_tiles)
            if pos1 and pos2:
                self.teleporters[pos1] = pos2
                self.teleporters[pos2] = pos1
                occupied_tiles.add(pos1)
                occupied_tiles.add(pos2)

    # ------------------------------------------------------------------
    # Food management
    # ------------------------------------------------------------------
    def spawn_food(self) -> None:
        """Randomly distribute *food_quantity* pellets on empty grid cells."""
        self.food.clear()
        occupied_tiles = self.obstacles.union(set(self.teleporters.keys()))
        while len(self.food) < self._food_quantity:
            pos = (
                random.randint(0, self.field_size - 1),
                random.randint(0, self.field_size - 1),
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
    def reset(self) -> None:
        """Remove all agents and respawn food."""
        self.agents.clear()
        self.spawn_food()

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
        # As a fallback, just append without check, but avoid static objects
        if "pos" in locals() and pos not in occupied_static:
            self.food.add(pos)

    def _find_unoccupied_pair(
        self, placer_rng: random.Random, occupied_tiles: Set[GridPos]
    ) -> Tuple[Optional[GridPos], Optional[GridPos]]:
        """Helper to find two distinct, unoccupied grid positions."""
        for _ in range(100):  # Limit attempts
            p1 = (placer_rng.randint(0, self.field_size - 1), placer_rng.randint(0, self.field_size - 1))
            p2 = (placer_rng.randint(0, self.field_size - 1), placer_rng.randint(0, self.field_size - 1))
            if p1 != p2 and p1 not in occupied_tiles and p2 not in occupied_tiles:
                return p1, p2
        return None, None