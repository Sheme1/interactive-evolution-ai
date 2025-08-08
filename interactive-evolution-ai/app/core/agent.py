"""Domain-level representation of a single agent in the simulation.

The class contains only **grid-based** logic (integer coordinates). Visual
rendering is delegated to ``app.game.renderer.Renderer`` and neural activity to
NEAT-Python networks provided externally.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .environment import Environment  # Forward reference to avoid cycles
    # ``neat`` is imported lazily in ``__post_init__`` to avoid mandatory
    # runtime dependency during static analysis / unit-testing.


GridPos = Tuple[int, int]

# Максимальный запас энергии агента. Пополняется при поедании еды.
ENERGY_MAX = 15  # will be patched dynamically by EvolutionManager


@dataclass
class Agent:  # pylint: disable=too-many-instance-attributes
    """Represents a single agent located on a 2-D integer grid."""

    id: int
    team: str  # Either "BLUE" or "RED"
    position: GridPos
    genome: "object"  # Stored for checkpointing / further evolution
    net: "object"  # neat.nn.FeedForwardNetwork | neat.nn.RecurrentNetwork etc.
    # Текущий запас энергии. Если достигает 0 — агент погибает.
    energy: int = ENERGY_MAX

    def move(self, dx: int, dy: int, field_size: int, env: "Environment | None" = None) -> None:
        """Translate the agent by *(dx, dy)* ensuring it remains in bounds.

        Parameters
        ----------
        dx, dy
            Discrete step values. Each should be in the range \[-1, 1\]. The
            calling code (e.g. evolution loop) is responsible for enforcing
            this range.
        field_size
            Edge length of the square simulation field.
        """
        if (dx, dy) == (0, 0):
            return  # Early-exit — saves boundary checks

        x, y = self.position
        new_x = max(0, min(field_size - 1, x + dx))
        new_y = max(0, min(field_size - 1, y + dy))

        # Проверяем коллизию с другим агентом или препятствием.
        if env is not None:
            if env.is_occupied((new_x, new_y), exclude_id=self.id) or env.is_obstacle((new_x, new_y)):
                # Клетка занята — остаёмся на месте.
                return

        self.position = (new_x, new_y)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def get_observation(self, env: "Environment") -> List[float]:
        """Compute a simple observation vector for the neural network.

        The current implementation encodes:

        1. *dx_food*, *dy_food* — Normalised vector (\[-1, 1\]) to the
           nearest food pellet.
        2. *dx_enemy*, *dy_enemy* — Normalised vector to the nearest enemy
           agent.
        3. *dx_obstacle*, *dy_obstacle* - Normalised vector to the nearest
           obstacle.
        4. *dx_teleporter*, *dy_teleporter* - Normalised vector to the nearest
           teleporter.

        5. 8 local obstacle sensors (N, NE, E, SE, S, SW, W, NW), where a
           value of 1.0 indicates a blocked cell (wall, obstacle, or
           another agent).

        The vector length is therefore *8 + 8 = 16*.
        """
        fx, fy = self._nearest_food_vector(env)
        ex, ey = self._nearest_enemy_vector(env)
        ox, oy = self._nearest_obstacle_vector(env)
        tx, ty = self._nearest_teleporter_vector(env)
        local_obstacles = self._get_local_obstacle_sensors(env)
        return [fx, fy, ex, ey, ox, oy, tx, ty] + local_obstacles

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_local_obstacle_sensors(self, env: "Environment") -> List[float]:
        """Return an 8-element list indicating obstacles in adjacent cells.

        Order is N, NE, E, SE, S, SW, W, NW (clockwise from North).
        Value is 1.0 if obstacle/wall/agent present, 0.0 otherwise.
        """
        x, y = self.position
        field_size = env.field_size
        # N, NE, E, SE, S, SW, W, NW (clockwise from North)
        directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        sensors = []
        for dx, dy in directions:
            check_x, check_y = x + dx, y + dy

            is_blocked = 0.0
            # Check for boundaries (walls)
            if not (0 <= check_x < field_size and 0 <= check_y < field_size):
                is_blocked = 1.0
            # Check for environment obstacles
            elif env.is_obstacle((check_x, check_y)):
                is_blocked = 1.0
            # Check for other agents
            elif env.is_occupied((check_x, check_y), exclude_id=self.id):
                is_blocked = 1.0

            sensors.append(is_blocked)
        return sensors

    def _nearest_food_vector(self, env: "Environment") -> Tuple[float, float]:
        if not env.food:
            return 0.0, 0.0

        x, y = self.position
        nearest = min(env.food, key=lambda p: hypot(p[0] - x, p[1] - y))
        dx = nearest[0] - x
        dy = nearest[1] - y
        norm = max(1.0, abs(dx) + abs(dy))  # Manhattan distance normaliser
        return dx / norm, dy / norm

    def _nearest_enemy_vector(self, env: "Environment") -> Tuple[float, float]:
        enemies = [a for a in env.agents.values() if a.team != self.team]
        if not enemies:
            return 0.0, 0.0

        x, y = self.position
        nearest = min(enemies, key=lambda a: hypot(a.position[0] - x, a.position[1] - y))
        dx = nearest.position[0] - x
        dy = nearest.position[1] - y
        norm = max(1.0, abs(dx) + abs(dy))
        return dx / norm, dy / norm

    def _nearest_obstacle_vector(self, env: "Environment") -> Tuple[float, float]:
        if not env.obstacles:
            return 0.0, 0.0

        x, y = self.position
        nearest = min(env.obstacles, key=lambda p: hypot(p[0] - x, p[1] - y))
        dx = nearest[0] - x
        dy = nearest[1] - y
        norm = max(1.0, abs(dx) + abs(dy))
        return dx / norm, dy / norm

    def _nearest_teleporter_vector(self, env: "Environment") -> Tuple[float, float]:
        if not env.teleporters:
            return 0.0, 0.0

        x, y = self.position
        # env.teleporters is a dict, we need to check keys
        nearest = min(env.teleporters.keys(), key=lambda p: hypot(p[0] - x, p[1] - y))
        dx = nearest[0] - x
        dy = nearest[1] - y
        norm = max(1.0, abs(dx) + abs(dy))
        return dx / norm, dy / norm
