"""Domain-level representation of a single agent in the simulation.

The class contains only **grid-based** logic (integer coordinates). Visual
rendering is delegated to ``app.game.renderer.Renderer`` and neural activity to
NEAT-Python networks provided externally.

Система восприятия агента теперь основана на эгоцентрическом окне 5x5
с 4 каналами информации (препятствия, еда, телепорты, враги).
"""
from __future__ import annotations

from dataclasses import dataclass
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
        """Построить эгоцентрическое наблюдение для нейросети.

        Использует новую систему эгоцентрических сенсоров: окно 5x5 вокруг
        агента с 4 каналами информации (препятствия, еда, телепорты, враги).

        Returns
        -------
        List[float]
            Плоский вектор длины 100 (5×5×4 канала).
        """
        from .sensors import get_egocentric_observation
        return get_egocentric_observation(self, env)
