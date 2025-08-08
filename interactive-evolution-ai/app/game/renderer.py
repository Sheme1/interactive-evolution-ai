"""Pygame‐based визуализатор сеточного поля.

Переводит логические координаты **клеток** в пиксели и рисует:
1. Сетку поля
2. Агентов двух команд
3. Пищу

Все вычисления выполняются через integer‐grid, полученный из
``app.core.environment.Environment``.
"""
from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import pygame

# --- Платформонезависимая работа с буфером обмена (UTF-16) ---
# Pyperclip даёт корректный CF_UNICODETEXT на Windows, чего не умеет
# стандартный pygame.scrap. Используем его, если установлен.
try:
    import pyperclip  # type: ignore
except ImportError:  # pragma: no cover – модуль может отсутствовать
    pyperclip = None



from ..core.agent import Agent, GridPos


Color = Tuple[int, int, int]
BLUE: Color = (30, 144, 255)
RED: Color = (220, 20, 60)
GREEN: Color = (34, 139, 34)
GRAY: Color = (211, 211, 211)
BLACK: Color = (0, 0, 0)
DARK_GRAY: Color = (105, 105, 105)
PURPLE: Color = (148, 0, 211)


class UserQuitException(Exception):
    """Исключение, выбрасываемое при закрытии окна Pygame пользователем."""

    pass

class LogPanel:
    """Панель для отображения цветных логов в реальном времени с прокруткой."""

    def __init__(self, x: int, y: int, width: int, height: int, font_size: int = 16):
        self.rect = pygame.Rect(x, y, width, height)
        self.bg_color = (25, 25, 35)
        self.border_color = GRAY
        try:
            self.font = pygame.font.SysFont("consolas", font_size)
        except pygame.error:
            self.font = pygame.font.Font(None, font_size + 2)

        self.messages: List[Tuple[str, Color]] = []
        self.line_height = self.font.get_height()
        self.max_lines = (height - 10) // self.line_height if self.line_height > 0 else 0

        self.scroll_pos = 0
        self.is_auto_scrolling = True

        # --- Атрибуты для выделения текста ---
        self.selection_start: Optional[Tuple[int, int]] = None
        self.selection_end: Optional[Tuple[int, int]] = None
        self.is_selecting = False
        self.selection_color = (70, 90, 120)

        # --- Атрибуты слайдера ---
        self.slider_track_rect: pygame.Rect | None = None
        self.slider_thumb_rect: pygame.Rect | None = None
        self.is_dragging_slider = False
        self.slider_drag_offset_y = 0
        self.scroll_to_bottom_button_rect = None

        self.colors = {
            "INFO": (220, 220, 220),
            "MOVE": (135, 206, 250),
            "EAT": (60, 179, 113),
            "DEATH": (255, 99, 71),
            "SPAWN": (218, 112, 214),
            "WARNING": (255, 165, 0),
            "GENERATION": (255, 215, 0),
        }

    def add_log(self, text: str, level: str = "INFO") -> None:
        """Добавить сообщение в лог с указанным уровнем (для цвета)."""
        import textwrap

        char_width = self.font.size("a")[0]
        char_width = 1 if char_width == 0 else char_width
        # Резервируем место для слайдера (15px) и отступов
        text_area_width = self.rect.width - 10 - 15 - 5  # отступ слева, слайдер, отступ справа
        max_chars = text_area_width // char_width if text_area_width > 0 else 1
        wrapped_lines = textwrap.wrap(text, width=max_chars)
        color = self.colors.get(level.upper(), self.colors["INFO"])
        for line in wrapped_lines:
            self.messages.append((line, color))

        if self.is_auto_scrolling:
            self.scroll_to_bottom()

    def scroll_to_bottom(self) -> None:
        """Прокрутить лог до самого конца."""
        if self.max_lines > 0:
            self.scroll_pos = max(0, len(self.messages) - self.max_lines)
        self.is_auto_scrolling = True

    def _screen_to_text_coords(self, screen_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Преобразовать экранные координаты в (индекс_строки, индекс_символа) в логе."""
        if self.line_height <= 0:
            return -1, -1

        # Y -> Глобальный индекс строки
        relative_y = screen_pos[1] - (self.rect.top + 5)
        line_offset = relative_y // self.line_height
        if not (0 <= line_offset < self.max_lines):
            return -1, -1

        global_line_idx = self.scroll_pos + line_offset
        if not (0 <= global_line_idx < len(self.messages)):
            return -1, -1

        # X -> Индекс символа (ищем ближайшую границу символа)
        relative_x = screen_pos[0] - (self.rect.left + 10)
        line_text = self.messages[global_line_idx][0]

        best_match_idx = 0
        min_dist = float("inf")
        for i in range(len(line_text) + 1):
            substr_width = self.font.size(line_text[:i])[0]
            dist = abs(relative_x - substr_width)
            if dist < min_dist:
                min_dist = dist
                best_match_idx = i
            else:
                # Расстояние начало увеличиваться, значит мы прошли нужную точку
                break
        return global_line_idx, best_match_idx

    def get_selected_text(self) -> str:
        """Собрать выделенный текст в одну строку."""
        if not self.selection_start or not self.selection_end:
            return ""

        start_pos = min(self.selection_start, self.selection_end)
        end_pos = max(self.selection_start, self.selection_end)
        start_line, start_char = start_pos
        end_line, end_char = end_pos

        if start_line == end_line:
            return self.messages[start_line][0][start_char:end_char]

        lines = [self.messages[start_line][0][start_char:]]
        for i in range(start_line + 1, end_line):
            lines.append(self.messages[i][0])
        lines.append(self.messages[end_line][0][:end_char])
        return "\n".join(lines)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Обработать события мыши для прокрутки и нажатия кнопок."""
        if event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_pos -= event.y * 3
                max_scroll = max(0, len(self.messages) - self.max_lines)
                self.scroll_pos = max(0, min(self.scroll_pos, max_scroll))
                self.is_auto_scrolling = self.scroll_pos >= max_scroll

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Левая кнопка мыши
            # Кнопка прокрутки вниз
            if self.scroll_to_bottom_button_rect and self.scroll_to_bottom_button_rect.collidepoint(event.pos):
                self.scroll_to_bottom()
            # Захват ползунка слайдера
            elif self.slider_thumb_rect and self.slider_thumb_rect.collidepoint(event.pos):
                self.is_dragging_slider = True
                self.slider_drag_offset_y = event.pos[1] - self.slider_thumb_rect.y
                self.is_auto_scrolling = False
            # Клик по треку слайдера
            elif self.slider_track_rect and self.slider_track_rect.collidepoint(event.pos):
                scroll_range = max(0, len(self.messages) - self.max_lines)
                if scroll_range > 0 and self.slider_thumb_rect:
                    relative_y = event.pos[1] - self.slider_track_rect.y
                    thumb_h = self.slider_thumb_rect.height
                    track_h = self.slider_track_rect.height
                    thumb_y_range = max(1, track_h - thumb_h)

                    # Перемещаем центр ползунка к месту клика
                    pos_ratio = (relative_y - thumb_h / 2) / thumb_y_range
                    self.scroll_pos = int(pos_ratio * scroll_range)
                    self.scroll_pos = max(0, min(self.scroll_pos, scroll_range))
                    self.is_auto_scrolling = False
            # Клик по текстовой области для выделения
            elif self.rect.collidepoint(event.pos):
                coords = self._screen_to_text_coords(event.pos)
                if coords != (-1, -1):
                    self.is_selecting = True
                    self.selection_start = coords
                    self.selection_end = coords
                    self.is_auto_scrolling = False

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_dragging_slider = False
            self.is_selecting = False

        if event.type == pygame.MOUSEMOTION:
            # Обработка перемещения мыши: прокрутка лога при перетаскивании ползунка
            # или обновление выделения текста.
            if self.is_dragging_slider and self.slider_track_rect and self.slider_thumb_rect:
                scroll_range = max(0, len(self.messages) - self.max_lines)
                if scroll_range > 0:
                    track_h = self.slider_track_rect.height
                    thumb_h = self.slider_thumb_rect.height
                    thumb_y_range = max(1, track_h - thumb_h)

                    new_y = event.pos[1] - self.slider_drag_offset_y
                    clamped_y = max(self.slider_track_rect.y, min(new_y, self.slider_track_rect.bottom - thumb_h))

                    y_ratio = (clamped_y - self.slider_track_rect.y) / thumb_y_range
                    self.scroll_pos = int(y_ratio * scroll_range)

                    max_scroll = max(0, len(self.messages) - self.max_lines)
                    self.is_auto_scrolling = self.scroll_pos >= max_scroll
            elif self.is_selecting:
                coords = self._screen_to_text_coords(event.pos)
                if coords != (-1, -1):
                    self.selection_end = coords

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, self.border_color, self.rect, 1)

        total_lines = len(self.messages)
        has_slider = total_lines > self.max_lines
        slider_width = 15

        # --- Отрисовка слайдера ---
        if has_slider:
            self.slider_track_rect = pygame.Rect(
                self.rect.right - slider_width - 2, self.rect.top + 2, slider_width, self.rect.height - 4
            )
            pygame.draw.rect(screen, (40, 40, 50), self.slider_track_rect, border_radius=3)

            scroll_range = total_lines - self.max_lines
            track_h = self.slider_track_rect.height
            thumb_h = max(20, track_h * (self.max_lines / total_lines))
            thumb_y_range = max(1, track_h - thumb_h)

            thumb_y = self.slider_track_rect.y + (self.scroll_pos / scroll_range) * thumb_y_range

            self.slider_thumb_rect = pygame.Rect(self.slider_track_rect.x, thumb_y, self.slider_track_rect.width, thumb_h)
            thumb_color = (120, 120, 140) if self.is_dragging_slider else (100, 100, 120)
            pygame.draw.rect(screen, thumb_color, self.slider_thumb_rect, border_radius=3)
        else:
            self.slider_track_rect = None
            self.slider_thumb_rect = None

        # --- Отрисовка текста и выделения ---
        y = self.rect.top + 5
        if self.max_lines > 0:
            max_scroll = max(0, len(self.messages) - self.max_lines)
            self.scroll_pos = max(0, min(self.scroll_pos, max_scroll))

        selection_range = None
        if self.selection_start and self.selection_end:
            selection_range = (min(self.selection_start, self.selection_end), max(self.selection_start, self.selection_end))

        visible_messages = self.messages[self.scroll_pos : self.scroll_pos + self.max_lines]

        for i, (text, color) in enumerate(visible_messages):
            global_line_idx = self.scroll_pos + i

            # Отрисовка выделения для текущей строки
            if selection_range:
                start_pos, end_pos = selection_range
                start_line, start_char = start_pos
                end_line, end_char = end_pos

                if start_line <= global_line_idx <= end_line:
                    sel_start = start_char if global_line_idx == start_line else 0
                    sel_end = end_char if global_line_idx == end_line else len(text)

                    if sel_start < len(text) and sel_start < sel_end:
                        x_offset = self.font.size(text[:sel_start])[0]
                        sel_width = self.font.size(text[sel_start:sel_end])[0]
                        highlight_rect = pygame.Rect(self.rect.left + 10 + x_offset, y, sel_width, self.line_height)
                        pygame.draw.rect(screen, self.selection_color, highlight_rect)

            # Отрисовка текста
            text_surface = self.font.render(text, True, color)
            screen.blit(text_surface, (self.rect.left + 10, y))
            y += self.line_height

        # --- Отрисовка кнопки "вниз" ---
        if not self.is_auto_scrolling:
            button_h = 25
            button_w = self.rect.width - 10
            if has_slider:
                button_w -= slider_width + 5

            self.scroll_to_bottom_button_rect = pygame.Rect(
                self.rect.left + 5, self.rect.bottom - button_h - 5, button_w, button_h
            )
            pygame.draw.rect(screen, (80, 80, 100), self.scroll_to_bottom_button_rect, border_radius=5)
            btn_text = self.font.render("↓ К последним логам ↓", True, (220, 220, 220))
            text_rect = btn_text.get_rect(center=self.scroll_to_bottom_button_rect.center)
            screen.blit(btn_text, text_rect)
        else:
            self.scroll_to_bottom_button_rect = None


class Renderer:  # pylint: disable=too-few-public-methods,too-many-instance-attributes
    """Отрисовщик поля и сущностей с поддержкой масштабирования.

    Создаёт окно с возможностью изменения размера. Игровое поле и панель
    логов автоматически масштабируются, сохраняя пропорции. Переключение
    в полноэкранный режим — клавиша F11.
    """

    def __init__(self, field_size: int, cell_size: int = 20, fps: int = 60) -> None:
        self.field_size = field_size
        # Логический размер ячейки для отрисовки на game_surface. Не меняется.
        self.cell_size = cell_size
        self._game_fps = fps
        self._ui_fps = 60  # Логи и UI всегда должны быть отзывчивы

        if not pygame.get_init():
            pygame.init()

        try:
            pygame.scrap.init()
        except pygame.error:
            # Scrap may not be available (e.g. no display server).
            # Copy/paste will fail gracefully later with a log message.
            pass

        # Начальный размер окна
        self.window_width, self.window_height = 1600, 900

        # Логическая поверхность для игры, на которой всегда происходит отрисовка.
        # Её размер постоянен.
        self.game_surface_logical_size = self.field_size * self.cell_size
        self.game_surface = pygame.Surface(
            (self.game_surface_logical_size, self.game_surface_logical_size)
        )

        # Основная поверхность для отображения (окно)
        self._screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Interactive Evolution AI – Simulation (F11 for fullscreen)")

        # После создания окна пробуем инициализировать буфер обмена,
        # так как он требует активного окна SDL для корректной работы.
        if not pygame.scrap.get_init():
            try:
                pygame.scrap.init()
            except pygame.error as e:
                # Если инициализация не удалась, выводим предупреждение в консоль.
                print(f"[WARNING] Не удалось инициализировать pygame.scrap: {e}")

        self._clock = pygame.time.Clock()

        # Прямоугольник, куда будет вписана отмасштабированная игровая поверхность
        self.game_dest_rect = pygame.Rect(0, 0, 1, 1)  # Заполнится в _handle_resize

        self.log_panel = LogPanel(x=0, y=0, width=1, height=1)  # Заполнится в _handle_resize

        # Первоначальный расчет разметки
        self._handle_resize((self.window_width, self.window_height))

    def _handle_resize(self, size: Tuple[int, int]) -> None:
        """Обработать изменение размера окна и пересчитать всю геометрию."""
        self.window_width, self.window_height = size
        self._screen = pygame.display.set_mode(size, pygame.RESIZABLE)

        # Ширина панели логов: 30% от ширины окна, в пределах [300, 600] px
        log_panel_width = int(max(300, min(600, self.window_width * 0.3)))
        game_area_width = self.window_width - log_panel_width
        game_area_height = self.window_height

        # Игровое поле должно быть квадратным
        game_display_size = min(game_area_width, game_area_height)

        # Центрируем игровое поле в отведенном для него пространстве
        game_offset_x = (game_area_width - game_display_size) // 2
        game_offset_y = (game_area_height - game_display_size) // 2
        self.game_dest_rect = pygame.Rect(
            game_offset_x, game_offset_y, game_display_size, game_display_size
        )

        # Обновляем геометрию панели логов
        self.log_panel.rect.x = game_area_width
        self.log_panel.rect.y = 0
        self.log_panel.rect.width = self.window_width - game_area_width
        self.log_panel.rect.height = self.window_height

        # Обновляем внутренние параметры панели логов
        if self.log_panel.line_height > 0:
            self.log_panel.max_lines = (self.window_height - 10) // self.log_panel.line_height

    def add_log(self, message: str, level: str) -> None:
        """Добавить сообщение в панель логов."""
        self.log_panel.add_log(message, level)

    def draw_grid(self) -> None:
        """Нарисовать сетку на *логической* игровой поверхности."""
        self.game_surface.fill((255, 255, 255))
        for i in range(self.field_size):
            for j in range(self.field_size):
                rect = (i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size)
                # Рисуем рамку для каждой ячейки. Это гарантирует равномерность.
                pygame.draw.rect(self.game_surface, GRAY, rect, 1)

    def draw_agents(self, agents: Iterable[Agent]) -> None:
        """Отрисовать агентов на *логической* игровой поверхности."""
        for agent in agents:
            color = BLUE if agent.team == "BLUE" else RED
            x_pix, y_pix = self._to_pixel(agent.position)
            rect = pygame.Rect(x_pix, y_pix, self.cell_size, self.cell_size)
            pygame.draw.rect(self.game_surface, color, rect)
            pygame.draw.rect(self.game_surface, BLACK, rect, width=1)  # Обводка

    def draw_food(self, food: Iterable[GridPos]) -> None:
        """Отрисовать пищу на *логической* игровой поверхности."""
        radius = max(2, self.cell_size // 4)
        for pos in food:
            center = self._to_pixel_center(pos)
            pygame.draw.circle(self.game_surface, GREEN, center, radius)

    def draw_obstacles(self, obstacles: Iterable[GridPos]) -> None:
        """Отрисовать препятствия на *логической* игровой поверхности."""
        for pos in obstacles:
            x_pix, y_pix = self._to_pixel(pos)
            rect = pygame.Rect(x_pix, y_pix, self.cell_size, self.cell_size)
            pygame.draw.rect(self.game_surface, DARK_GRAY, rect)

    def draw_teleporters(self, teleporters: dict[GridPos, GridPos]) -> None:
        """Отрисовать телепорты на *логической* игровой поверхности."""
        drawn_pairs = set()
        for pos1, pos2 in teleporters.items():
            # Сортируем, чтобы избежать дублирования линий и кругов
            pair = tuple(sorted((pos1, pos2)))
            if pair in drawn_pairs:
                continue

            # Рисуем площадки телепортов
            for pos in [pos1, pos2]:
                center = self._to_pixel_center(pos)
                pygame.draw.circle(self.game_surface, PURPLE, center, self.cell_size // 2)
                pygame.draw.circle(self.game_surface, BLACK, center, self.cell_size // 2, width=2)

            start_pix = self._to_pixel_center(pos1)
            end_pix = self._to_pixel_center(pos2)
            pygame.draw.line(self.game_surface, PURPLE, start_pix, end_pix, width=2)
            drawn_pairs.add(pair)

    def update(self) -> None:
        """Финализировать кадр и обработать события, блокируясь на время игрового тика.

        Эта функция заменяет собой простой `pygame.time.Clock.tick()`. Она
        блокирует выполнение на время, соответствующее одному игровому кадру
        (1 / game_fps), но внутри этого времени запускает собственный цикл
        обработки событий и отрисовки с высокой частотой (UI_FPS), чтобы
        обеспечить плавность и отзывчивость интерфейса (например, прокрутки
        панели логов), даже если симуляция работает с низким FPS.

        Предполагается, что `game_surface` уже обновлена (на ней нарисовано
        актуальное состояние игрового поля) до вызова этого метода.
        """
        game_tick_duration_ms = 1000.0 / self._game_fps
        start_time_ms = pygame.time.get_ticks()

        # Этот цикл выполняется в течение одного игрового тика,
        # но с высокой частотой обновления UI.
        while pygame.time.get_ticks() - start_time_ms < game_tick_duration_ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise UserQuitException()
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.size)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        pygame.display.toggle_fullscreen()
                    elif event.key == pygame.K_c and (event.mod & pygame.KMOD_CTRL):
                        selected_text = self.log_panel.get_selected_text()
                        if selected_text:
                            if pyperclip is not None:
                                try:
                                    pyperclip.copy(selected_text)
                                    self.add_log("Текст скопирован", "INFO")
                                except Exception as e:  # noqa: BLE001
                                    self.add_log(f"Ошибка копирования: {e}", "WARNING")
                            else:
                                self.add_log("Для копирования установите пакет pyperclip", "WARNING")

                self.log_panel.handle_event(event)

            self._screen.fill(BLACK)

            # Игровая поверхность статична внутри этого цикла. Мы просто
            # перерисовываем её в каждом кадре UI.
            scaled_game_surface = pygame.transform.scale(self.game_surface, self.game_dest_rect.size)
            self._screen.blit(scaled_game_surface, self.game_dest_rect)

            self.log_panel.draw(self._screen)
            pygame.display.flip()
            self._clock.tick(self._ui_fps)

    # ------------------------------------------------------------------
    # Внутренние утилиты
    # ------------------------------------------------------------------
    def _to_pixel(self, grid_pos: GridPos) -> GridPos:
        """Преобразовать координату **ячейки** в левый-верхний пиксель на логической поверхности."""
        x, y = grid_pos
        return x * self.cell_size, y * self.cell_size

    def _to_pixel_center(self, grid_pos: GridPos) -> GridPos:
        """Преобразовать координату ячейки в центр пикселей на логической поверхности."""
        px, py = self._to_pixel(grid_pos)
        return px + self.cell_size // 2, py + self.cell_size // 2