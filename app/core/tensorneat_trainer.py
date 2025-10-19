"""Высокоуровневый интерфейс для обучения с TensorNEAT.

Этот модуль предоставляет упрощённый API для запуска обучения
с использованием TensorNEAT Pipeline и GPU-ускорения.
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import pickle

import jax
from jax import random

from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.common import ACT, AGG

from .jax_environment import EnvConfig
from .tensorneat_problem import GridBattleProblem

if TYPE_CHECKING:
    from ..utils.settings import Settings


class TensorNEATTrainer:
    """Менеджер обучения с TensorNEAT Pipeline.

    Преимущества над оригинальным EvolutionManager:
    - GPU-ускорение через JAX (до 500x быстрее)
    - Автоматическая векторизация оценки популяции
    - JIT-компиляция для максимальной производительности
    - Полная детерминированность через JAX random keys
    """

    def __init__(self, settings: "Settings", seed: int = 42):
        """Инициализация тренера.

        Parameters
        ----------
        settings : Settings
            Настройки из settings.ini.
        seed : int
            Seed для воспроизводимости результатов.
        """
        self.settings = settings
        self.seed = seed

        # Создаём конфигурацию окружения из settings
        self.env_config = EnvConfig(
            field_size=settings.get_int("Field", "field_size"),
            food_quantity=settings.get_int("Simulation", "food_quantity"),
            energy_max=self._compute_energy_max(settings),
            obstacles_percentage=self._parse_percentage(
                settings.get_str("Environment", "obstacles_percentage")
            ),
            teleporters_count=settings.get_int("Environment", "teleporters_count"),
        )

        # Параметры обучения
        self.population_size = settings.get_int("Simulation", "population_size")
        self.generations = settings.get_int("Simulation", "generations")
        self.max_steps = int(self.env_config.energy_max * 1.5)

        # Создаём TensorNEAT Pipeline
        self.pipeline = self._create_pipeline()

    def _compute_energy_max(self, settings: "Settings") -> int:
        """Вычислить ENERGY_MAX из настроек (копия логики из constants.py)."""
        field_size = settings.get_int("Field", "field_size")
        food_quantity = settings.get_int("Simulation", "food_quantity")

        avg_dist = field_size / 4.0
        avg_food_per_agent = food_quantity / 2.0

        if avg_food_per_agent > 0:
            energy = int(avg_dist / avg_food_per_agent)
        else:
            energy = 10

        return max(10, min(energy, 50))

    def _parse_percentage(self, percentage_str: str) -> float:
        """Конвертировать строку процента в float (например, "5%" -> 0.05)."""
        try:
            return float(percentage_str.strip().replace("%", "")) / 100.0
        except (ValueError, TypeError):
            return 0.0

    def _create_pipeline(self) -> Pipeline:
        """Создать TensorNEAT Pipeline с настройками из settings.ini."""
        # Определяем размеры входов/выходов
        num_inputs = 100  # 5x5x4 каналов эгоцентрического окна
        num_outputs = 2  # dx, dy для движения

        # Создаём алгоритм NEAT
        algorithm = NEAT(
            pop_size=self.population_size,
            species_size=20,  # Количество видов
            survival_threshold=0.2,  # 20% лучших выживают
            compatibility_threshold=4.0,  # Порог совместимости для видообразования
            genome=DefaultGenome(
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                init_hidden_layers=(),  # Начинаем без скрытых слоёв
                node_gene=BiasNode(
                    activation_options=ACT.tanh,  # Функция активации
                    aggregation_options=AGG.sum,  # Агрегация входов
                ),
                output_transform=ACT.tanh,  # Выходной слой: tanh для [-1, 1]
            ),
        )

        # Создаём проблему (окружение)
        problem = GridBattleProblem(
            env_config=self.env_config,
            max_steps=self.max_steps,
            num_matches=5,  # Количество матчей для оценки каждого генома
        )

        # Создаём Pipeline
        pipeline = Pipeline(
            algorithm=algorithm,
            problem=problem,
            seed=self.seed,
            generation_limit=self.generations,
            fitness_target=100.0,  # Целевой фитнес (можно настроить)
        )

        return pipeline

    def run_training(self, verbose: bool = True) -> tuple:
        """Запустить процесс обучения.

        Parameters
        ----------
        verbose : bool
            Выводить ли прогресс обучения в консоль.

        Returns
        -------
        tuple
            (final_state, best_genome) - финальное состояние и лучший геном.
        """
        if verbose:
            print("=" * 60)
            print("🚀 ЗАПУСК ОБУЧЕНИЯ С TENSORNEAT")
            print("=" * 60)
            print(f"Популяция: {self.population_size}")
            print(f"Поколения: {self.generations}")
            print(f"Seed: {self.seed}")
            print(f"Входы: {100}, Выходы: {2}")
            print(f"Размер поля: {self.env_config.field_size}")
            print(f"Энергия: {self.env_config.energy_max}")
            print(f"Max шагов: {self.max_steps}")

            # Информация о JAX устройствах
            devices = jax.devices()
            print(f"\nДоступные устройства JAX: {len(devices)}")
            for i, device in enumerate(devices):
                print(f"  [{i}] {device.device_kind}: {device}")

            print("=" * 60)
            print()

        # Инициализируем Pipeline
        state = self.pipeline.setup()

        if verbose:
            print("✅ Pipeline инициализирован")
            print("🏃 Запуск эволюции...\n")

        # Запускаем обучение
        try:
            state, best = self.pipeline.auto_run(state)

            if verbose:
                print("\n✅ Обучение завершено!")
                print(f"Лучший фитнес: {best}")

            return state, best

        except KeyboardInterrupt:
            if verbose:
                print("\n⚠️  Обучение прервано пользователем")
            return state, None

    def save_best_genome(self, state, output_dir: Path | str = "checkpoints") -> Path:
        """Сохранить лучший геном.

        Parameters
        ----------
        state
            Финальное состояние Pipeline.
        output_dir : Path | str
            Директория для сохранения.

        Returns
        -------
        Path
            Путь к сохранённому файлу.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Формируем имя файла
        filename = f"best_genome_tensorneat_gen{state.generation}.pkl"
        filepath = output_dir / filename

        # Сохраняем состояние целиком (содержит лучший геном)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        print(f"💾 Лучший геном сохранён: {filepath}")

        return filepath

    def load_genome(self, filepath: Path | str):
        """Загрузить сохранённый геном для дообучения или визуализации.

        Parameters
        ----------
        filepath : Path | str
            Путь к файлу с геномом.

        Returns
        -------
        Any
            Загруженное состояние Pipeline.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        print(f"📂 Геном загружен из: {filepath}")
        return state


def create_trainer_from_settings(settings: "Settings", seed: int = 42) -> TensorNEATTrainer:
    """Фабричная функция для создания тренера из Settings.

    Parameters
    ----------
    settings : Settings
        Настройки из settings.ini.
    seed : int
        Seed для воспроизводимости.

    Returns
    -------
    TensorNEATTrainer
        Готовый к использованию тренер.
    """
    return TensorNEATTrainer(settings, seed=seed)
