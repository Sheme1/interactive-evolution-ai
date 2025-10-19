"""CustomProblem для интеграции игрового окружения с TensorNEAT.

Этот модуль реализует интерфейс BaseProblem из TensorNEAT для обучения
агентов в 1v1 сражениях. Использует JAX для GPU-ускорения.
"""
from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jnp
from jax import random
from tensorneat.problem.base import BaseProblem

from .jax_environment import EnvConfig, EnvState, reset_env, step_env, get_observation, is_done


class GridBattleProblem(BaseProblem):
    """Проблема обучения агентов в 1v1 сражениях на сетке.

    Эта задача совместима с TensorNEAT Pipeline и поддерживает:
    - JIT-компиляцию для ускорения на GPU
    - Векторизацию через vmap для параллельной оценки популяции
    - Детерминированную воспроизводимость через JAX random keys
    """

    jitable = True  # Критически важно для GPU-ускорения!

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        max_steps: int = 100,
        num_matches: int = 5,
    ):
        """Инициализация проблемы.

        Parameters
        ----------
        env_config : EnvConfig, optional
            Конфигурация окружения. Если None, используются значения по умолчанию.
        max_steps : int
            Максимальное количество шагов в одном матче.
        num_matches : int
            Количество матчей для оценки каждого генома (усреднение).
        """
        super().__init__()
        self.env_config = env_config or EnvConfig()
        self.max_steps = max_steps
        self.num_matches = num_matches

    @property
    def input_shape(self) -> tuple:
        """Размерность входов нейросети (эгоцентрическое окно 5x5x4)."""
        return (100,)  # 5 * 5 * 4 каналов

    @property
    def output_shape(self) -> tuple:
        """Размерность выходов нейросети (dx, dy для движения)."""
        return (2,)

    def evaluate(self, state: Any, randkey: jnp.ndarray, act_func: Any, params: Any) -> float:
        """Оценить фитнес одного генома через несколько 1v1 матчей.

        Эта функция вызывается TensorNEAT Pipeline для каждого индивида в популяции.
        Pipeline автоматически векторизует её через vmap для параллельной оценки.

        Parameters
        ----------
        state : Any
            Состояние Pipeline (не используется в данной реализации).
        randkey : jnp.ndarray
            JAX random key для детерминированной генерации.
        act_func : callable
            Функция нейросети: act_func(state, params, observation) -> action
        params : Any
            Параметры нейросети (веса, структура).

        Returns
        -------
        float
            Усредненный фитнес по num_matches матчам.
        """
        # Создаём оппонента со случайными действиями (baseline)
        # В будущем можно заменить на co-evolution с двумя популяциями
        def random_opponent_policy(obs: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
            """Случайная политика оппонента."""
            return random.uniform(key, shape=(2,), minval=-1.0, maxval=1.0)

        # Играем num_matches матчей и усредняем фитнес
        match_keys = random.split(randkey, self.num_matches)

        def single_match(match_key: jnp.ndarray) -> float:
            """Провести один матч и вернуть фитнес главного агента."""
            return self._run_match(match_key, act_func, params, random_opponent_policy)

        # Векторизуем по матчам
        fitnesses = jax.vmap(single_match)(match_keys)

        return jnp.mean(fitnesses)

    @jax.jit
    def _run_match(
        self,
        randkey: jnp.ndarray,
        act_func: Any,
        params: Any,
        opponent_policy: Any,
    ) -> float:
        """Провести один 1v1 матч.

        Parameters
        ----------
        randkey : jnp.ndarray
            JAX random key.
        act_func : callable
            Функция нейросети главного агента.
        params : Any
            Параметры нейросети.
        opponent_policy : callable
            Политика оппонента.

        Returns
        -------
        float
            Финальный фитнес главного агента (индекс 0).
        """
        # Инициализируем окружение
        key_reset, key_sim = random.split(randkey)
        env_state = reset_env(key_reset, self.env_config)

        total_reward_main = 0.0
        total_reward_opp = 0.0

        # Симуляция матча
        def step_fn(carry, _):
            env_st, reward_main, reward_opp, rng_key = carry

            # Получаем наблюдения для обоих агентов
            obs_main = get_observation(env_st, agent_idx=0)
            obs_opp = get_observation(env_st, agent_idx=1)

            # Получаем действия
            key_opp, key_next = random.split(rng_key)
            action_main = act_func(None, params, obs_main)  # state=None для stateless networks
            action_opp = opponent_policy(obs_opp, key_opp)

            actions = jnp.stack([action_main, action_opp])

            # Выполняем шаг
            new_env_state, rewards = step_env(env_st, actions)

            # Накапливаем награды
            new_reward_main = reward_main + rewards[0]
            new_reward_opp = reward_opp + rewards[1]

            return (new_env_state, new_reward_main, new_reward_opp, key_next), None

        # Запускаем цикл симуляции через lax.scan (JIT-friendly)
        (final_state, total_reward_main, total_reward_opp, _), _ = jax.lax.scan(
            step_fn,
            (env_state, 0.0, 0.0, key_sim),
            None,
            length=self.max_steps
        )

        # Дополнительные бонусы/штрафы
        # Бонус за выживание
        survival_bonus = jnp.where(final_state.agent_alive[0], 5.0, 0.0)

        # Штраф если оппонент жив дольше
        opponent_survival_penalty = jnp.where(
            final_state.agent_alive[1] & ~final_state.agent_alive[0],
            -10.0,
            0.0
        )

        final_fitness = total_reward_main + survival_bonus + opponent_survival_penalty

        return final_fitness

    def show(self, state: Any, randkey: jnp.ndarray, act_func: Any, params: Any, *args, **kwargs):
        """Визуализация лучшего генома (пока не реализована).

        Для визуализации можно использовать оригинальный Renderer из app.game.renderer,
        но это потребует конвертации из JAX обратно в NumPy.
        """
        print("Визуализация пока не реализована для TensorNEAT версии.")
        print("Используйте оригинальный EvolutionManager для визуализации.")

    def show_details(self, state: Any, *args, **kwargs):
        """Вывод детальной информации о популяции (опционально).

        Можно использовать для анализа распределения фитнеса, сложности сетей и т.д.
        """
        pass  # Пока не реализовано


class CoEvolutionProblem(GridBattleProblem):
    """Расширенная версия с ко-эволюцией двух популяций.

    TODO: Реализовать поддержку двух популяций (BLUE vs RED),
    аналогично оригинальному EvolutionManager.

    Для этого потребуется:
    1. Хранить вторую популяцию в состоянии Pipeline
    2. Выбирать случайных оппонентов из второй популяции
    3. Обновлять обе популяции поочерёдно
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError(
            "Ко-эволюция пока не реализована. "
            "Используйте GridBattleProblem с random opponent или "
            "дождитесь следующей версии."
        )
