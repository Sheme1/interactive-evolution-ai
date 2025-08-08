"""Orchestrator: интерактивное меню и запуск режимов.

Полностью терминальное взаимодействие с Questionary.
"""
from __future__ import annotations

from pathlib import Path


import questionary
from rich.console import Console

from .utils.settings import Settings
from .core.evolution import EvolutionManager
from .utils.file_utils import pick_model_file
from .game.renderer import UserQuitException


class AppManager:  # pylint: disable=too-few-public-methods
    """Главный класс, управляющий циклом меню."""

    def __init__(self) -> None:
        self._settings = Settings()
        self._console = Console()
        self._neat_config_path = Path(__file__).resolve().parent.parent / "config" / "neat_config.txt"

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401
        """Запустить интерактивное меню (блокирующе)."""
        while True:
            choice = questionary.select(
                "Выберите действие:",
                choices=[
                    "1. Начать тренировку (с визуализацией)",
                    "2. Начать тренировку (без визуализации)",
                    "3. Начать игру",
                    "4. Дотренировать модель",
                    "5. Настройки",
                    "6. Выход",
                ],
            ).ask()

            if choice.startswith("1"):
                self._handle_train(visualize=True)
            elif choice.startswith("2"):
                self._handle_train(visualize=False)
            elif choice.startswith("3"):
                self._handle_play()
            elif choice.startswith("4"):
                self._handle_continue_training()
            elif choice.startswith("5"):
                self._handle_settings()
            elif choice.startswith("6") or choice is None:
                break

    # ------------------------------------------------------------------
    # Private handlers
    # ------------------------------------------------------------------
    def _handle_train(self, visualize: bool) -> None:
        self._console.print("[bold yellow]Запуск нового процесса эволюции...[/]")
        try:
            mgr = EvolutionManager(self._settings, self._neat_config_path)
            mgr.run_evolution(visualize=visualize)
        except UserQuitException:
            self._console.print("\n[yellow]Визуализация прервана пользователем. Возврат в главное меню.[/yellow]")

    def _handle_play(self) -> None:
        """Запуск режима игры с выбором двух моделей."""
        from .utils.file_utils import pick_model_file  # локальный импорт
        import pickle
        file_a = pick_model_file("Выберите модель для агента BLUE")
        file_b = pick_model_file("Выберите модель для агента RED")
        if not file_a or not file_b:
            self._console.print("[red]Файлы не выбраны, отмена.")
            return

        with open(file_a, "rb") as fp:
            genome_a = pickle.load(fp)
        with open(file_b, "rb") as fp:
            genome_b = pickle.load(fp)

        try:
            self._console.print(f"[bold yellow]Запуск игры с моделями:[/]\n- Агент BLUE: [cyan]{file_a.name}[/cyan]\n- Агент RED: [cyan]{file_b.name}[/cyan]")
            self._run_game(genome_a, genome_b)
        except UserQuitException:
            self._console.print("\n[yellow]Игра прервана пользователем. Возврат в главное меню.[/yellow]")

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    def _run_game(self, genome_a, genome_b):  # noqa: D401
        """Визуальная дуэль двух обученных агентов с подсчётом очков."""
        import neat  # локальный импорт
        from .core.environment import Environment
        from .core.agent import Agent
        from .utils.constants import compute_constants
        from .game.renderer import Renderer  # type: ignore[attr-defined]
        from .utils.console_utils import create_metrics_renderable
        import random

        field_size = self._settings.get_int("Field", "field_size")
        food_qty = self._settings.get_int("Simulation", "food_quantity")
        const = compute_constants(self._settings)
        Agent.ENERGY_MAX = const.energy_max

        # -------------------------
        # Подготовка NEAT сетей
        # -------------------------
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(self._neat_config_path),
        )
        # 4 вектора (x,y) до объектов + 8 сенсоров окружения
        config.genome_config.num_inputs = 16
        config.genome_config.num_outputs = 2
        net_a = neat.nn.FeedForwardNetwork.create(genome_a, config)
        net_b = neat.nn.FeedForwardNetwork.create(genome_b, config)

        # Счётчики очков (побед в раундах)
        score_a = 0
        score_b = 0
        round_idx = 0

        fps_setting = self._settings.get_int("Display", "fps")
        renderer = Renderer(field_size, cell_size=20, fps=fps_setting)

        obstacles_percentage = self._settings.get_str("Environment", "obstacles_percentage")
        teleporters_count = self._settings.get_int("Environment", "teleporters_count")

        # Бесконечный цикл раундов
        while True:
            renderer.add_log(f"Раунд {round_idx + 1}. Начали!", "GENERATION")
            round_idx += 1
            # --- (Re)создаём среду без респавна еды ---
            # Настраиваем поведение респавна в зависимости от конфигурации.
            food_respawn = self._settings.get_bool("Simulation", "food_respawn")
            if food_respawn:
                env = Environment(
                    field_size,
                    food_qty,
                    obstacles_percentage_str=obstacles_percentage,
                    teleporters_count=teleporters_count,
                )
            else:
                env = Environment(
                    field_size,
                    food_qty,
                    spawn_interval=999_999,  # по сути блокируем респавн
                    spawn_batch=0,
                    obstacles_percentage_str=obstacles_percentage,
                    teleporters_count=teleporters_count,
                )

            # Создаём агентов заново, чтобы сбросить энергию/позицию
            agent_a = Agent(
                0,
                "BLUE",
                (random.randint(0, field_size - 1), random.randint(0, field_size - 1)),
                genome_a,
                net_a,
            )
            agent_b = Agent(
                1,
                "RED",
                (random.randint(0, field_size - 1), random.randint(0, field_size - 1)),
                genome_b,
                net_b,
            )
            env.add_agent(agent_a)
            env.add_agent(agent_b)
            agent_a._logged_death = False  # type: ignore[attr-defined]
            agent_b._logged_death = False  # type: ignore[attr-defined]

            eaten_a = 0
            eaten_b = 0

            # --- цикл шагов в пределах одного раунда ---
            while True:
                # На каждом шаге перемешиваем агентов, чтобы порядок хода был
                # случайным. Это предотвращает ситуацию, когда агент BLUE
                # всегда ходит первым и получает преимущество.
                agent_list = [agent_a, agent_b]
                random.shuffle(agent_list)
                for agent in agent_list:
                    if agent.energy <= 0:
                        continue  # мёртвый — пропускаем
                    old_pos = agent.position
                    obs = agent.get_observation(env)
                    dx_raw, dy_raw = agent.net.activate(obs)
                    dx = max(-1, min(1, int(round(dx_raw))))
                    dy = max(-1, min(1, int(round(dy_raw))))
                    agent.move(dx, dy, field_size, env)
                    if agent.position != old_pos:
                        renderer.add_log(f"Агент {agent.team} -> {agent.position}", "MOVE")
                    elif (dx, dy) != (0, 0):
                        renderer.add_log(f"Агент {agent.team} столкновение", "WARNING")

                # Шаг среды
                eaters = env.step()
                for eater in eaters:
                    renderer.add_log(f"Агент {eater.team} съел еду", "EAT")
                    if eater.team == "BLUE":
                        eaten_a += 1
                    else:
                        eaten_b += 1

                # Логирование спавна еды
                if food_respawn and env._ticks % env._spawn_interval == 0 and env._spawn_batch > 0:
                    renderer.add_log(f"Новая еда ({env._spawn_batch} шт.)", "SPAWN")

                # --- Проверка условий завершения раунда ---
                a_dead = agent_a.energy <= 0
                b_dead = agent_b.energy <= 0
                food_empty = len(env.food) == 0

                if a_dead and not agent_a._logged_death:  # type: ignore[attr-defined]
                    renderer.add_log("Агент BLUE погиб", "DEATH")
                    agent_a._logged_death = True  # type: ignore[attr-defined]
                if b_dead and not agent_b._logged_death:  # type: ignore[attr-defined]
                    renderer.add_log("Агент RED погиб", "DEATH")
                    agent_b._logged_death = True  # type: ignore[attr-defined]

                if a_dead or b_dead or food_empty:
                    # Определяем победителя раунда
                    if a_dead and not b_dead:
                        score_b += 1
                    elif b_dead and not a_dead:
                        score_a += 1
                    elif food_empty:
                        if eaten_a > eaten_b:
                            score_a += 1
                        elif eaten_b > eaten_a:
                            score_b += 1

                    winner_msg = ""
                    if a_dead and not b_dead:
                        winner_msg = "RED победил!"
                    elif b_dead and not a_dead:
                        winner_msg = "BLUE победил!"
                    elif food_empty:
                        if eaten_a > eaten_b:
                            winner_msg = "BLUE победил по очкам!"
                        elif eaten_b > eaten_a:
                            winner_msg = "RED победил по очкам!"
                        else:
                            winner_msg = "Ничья!"
                    renderer.add_log(f"Раунд завершен. {winner_msg}", "GENERATION")
                    # Выводим счёт
                    renderable = create_metrics_renderable(round_idx, float(score_a), float(score_b))
                    # Восстанавливаем старое поведение (очистка + печать) для игрового режима.
                    self._console.clear()
                    self._console.print(renderable)
                    break  # выходим из цикла шагов — начинаем новый раунд

                # --- Рендеринг ---
                renderer.draw_grid()
                renderer.draw_obstacles(env.obstacles)
                renderer.draw_teleporters(env.teleporters)
                renderer.draw_food(env.food)
                renderer.draw_agents(env.agents.values())
                renderer.update()

    def _handle_continue_training(self) -> None:
        file_a = pick_model_file("Выберите модель для команды BLUE")
        file_b = pick_model_file("Выберите модель для команды RED")
        if not file_a or not file_b:
            self._console.print("[red]Файлы не выбраны, отмена.")
            return

        visualize = questionary.confirm("Запустить с визуализацией?").ask()
        if visualize is None:  # User pressed Ctrl+C
            return

        try:
            import pickle

            with open(file_a, "rb") as fp:
                genome_a = pickle.load(fp)
            with open(file_b, "rb") as fp:
                genome_b = pickle.load(fp)

            self._console.print(f"[bold yellow]Продолжение тренировки из файлов:[/]\n- Команда BLUE: [cyan]{file_a.name}[/cyan]\n- Команда RED: [cyan]{file_b.name}[/cyan]")
            mgr = EvolutionManager(self._settings, self._neat_config_path)
            mgr.run_evolution(visualize=visualize, continue_from=(genome_a, genome_b))
        except UserQuitException:
            self._console.print("\n[yellow]Визуализация прервана пользователем. Возврат в главное меню.[/yellow]")

    def _handle_settings(self) -> None:  # noqa: D401
        """Интерактивное редактирование `settings.ini`.

        1. Пользователь выбирает секцию.
        2. Затем параметр внутри секции.
        3. Вводит новое значение.
        Цикл продолжается, пока пользователь не выберет «Выход».
        """
        parser = self._settings._parser  # type: ignore[attr-defined]

        while True:
            section = questionary.select(
                "Выберите секцию настроек (Esc для выхода)",
                choices=[*parser.sections(), "<выход>"]
            ).ask()
            if section in (None, "<выход>"):
                break

            while True:
                items = list(parser.items(section))
                param_choice = questionary.select(
                    f"[{section}] Выберите параметр для изменения (Esc назад)",
                    choices=[f"{k} = {v}" for k, v in items] + ["<назад>"]
                ).ask()
                if param_choice in (None, "<назад>"):
                    break

                key = param_choice.split(" = ")[0]
                current_val = parser.get(section, key)
                new_val = questionary.text(
                    f"Новое значение для {key} (текущее: {current_val})", default=current_val
                ).ask()
                if new_val is not None and new_val != current_val:
                    self._settings.set_str(section, key, new_val)
                    self._console.print(f"[green]Обновлено[/] {section}.{key} → {new_val}")

        self._console.print("[bold green]Завершено редактирование настроек.[/]")