from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pygame

from config.modes import DEFAULT_MODE_KEY, MODES, MODE_ORDER, ModeConfig, get_mode_config
from controllers.base_controller import BaseController, MovementState
from controllers.hand_controller import HandController
from controllers.pose_controller import PoseController
from core.calibration_store import CalibrationStore
from core.game_manager import GameManager as SessionGameManager
from core.level import FPS, HEIGHT, WIDTH, Level
from core.player_controller import PlayerController
from core.player import Player
from core.sound_manager import SoundManager
from core.ui_manager import UIManager
from screens.calibration_screen import CalibrationScreen
from screens.home_screen import HomeScreen
from screens.mode_select_screen import ModeSelectScreen


class Game:
    def __init__(self, mode_config: ModeConfig | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("GesturePlay AI Runner | Neon Fitness")
        display_info = pygame.display.Info()
        max_w = max(800, display_info.current_w - 80)
        max_h = max(450, display_info.current_h - 80)
        initial_scale = min(max_w / WIDTH, max_h / HEIGHT, 1.0)
        initial_size = (int(WIDTH * initial_scale), int(HEIGHT * initial_scale))
        self.window = pygame.display.set_mode(initial_size, pygame.RESIZABLE)
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self._viewport = pygame.Rect(0, 0, initial_size[0], initial_size[1])
        self._scale = 1.0
        self._update_viewport(initial_size)
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("segoe ui", 60, bold=True)
        self.font_prompt = pygame.font.SysFont("segoe ui", 50, bold=True)
        self.font_ui = pygame.font.SysFont("segoe ui", 30, bold=True)
        self.font_body = pygame.font.SysFont("segoe ui", 24)
        self.font_small = pygame.font.SysFont("segoe ui", 19)

        self.home_screen = HomeScreen()
        self.mode_select_screen = ModeSelectScreen(MODE_ORDER)
        self.calibration_screen = CalibrationScreen()
        self.sound_manager = SoundManager(enabled=True)
        self._load_sounds()
        self.calibration_store = CalibrationStore()
        self.session_manager = SessionGameManager(session_target_seconds=180.0)
        self.player_controller = PlayerController()
        self.ui_manager = UIManager()

        self.mode_config = mode_config or MODES[DEFAULT_MODE_KEY]
        if self.mode_config.key in MODE_ORDER:
            self.mode_select_screen.selected_index = MODE_ORDER.index(self.mode_config.key)

        self.controller: BaseController | None = None
        self.player: Player | None = None
        self.level: Level | None = None
        self.menu_level = Level(MODES[DEFAULT_MODE_KEY])

        self.score = 0
        self.coin_count = 0
        self.best_score = 0
        self.elapsed = 0.0
        self.current_speed = self.mode_config.speed
        self.next_prompt = "RUN"
        self.controls = MovementState()
        self.instructions_text = "Keep shoulders, wrists, and hips visible."
        self.camera_surface: pygame.Surface | None = None
        self.calibration_samples: list[dict[str, float]] = []
        self.calibration_target_samples = int(FPS * 3.0)
        self.calibration_progress = 0.0
        self.calibration_status = "Auto-calibrating height, arm length, dominant hand..."
        self.calibration_has_saved_profile = False
        self._frame_dt = 1.0 / FPS
        self._mouse_clicked = False

        self.state = "home" if mode_config is None else "playing"
        self._state_handlers = {
            "home": self._update_home,
            "mode_select": self._update_mode_select,
            "calibration": self._update_calibration,
            "playing": self._update_playing,
            "game_over": self._update_game_over,
            "stats": self._update_stats,
        }

        if self.state == "playing":
            self._activate_mode(self.mode_config.key)

    def run(self) -> None:
        self.running = True
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                self._frame_dt = dt
                events = pygame.event.get()
                self._mouse_clicked = any(
                    event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                    for event in events
                )

                if any(event.type == pygame.QUIT for event in events):
                    self.running = False
                    continue
                for event in events:
                    if event.type == pygame.VIDEORESIZE:
                        size = (max(640, event.w), max(360, event.h))
                        self.window = pygame.display.set_mode(size, pygame.RESIZABLE)
                        self._update_viewport(size)

                handler = self._state_handlers.get(self.state)
                if handler is not None:
                    handler(dt, events)

                self.ui_manager.update(dt)
                self._draw_frame()
                self._present_frame()
                pygame.display.flip()
        finally:
            self._release_controller()
            pygame.quit()

    def _update_viewport(self, window_size: tuple[int, int]) -> None:
        win_w, win_h = window_size
        self._scale = max(0.1, min(win_w / WIDTH, win_h / HEIGHT))
        scaled_w = int(WIDTH * self._scale)
        scaled_h = int(HEIGHT * self._scale)
        offset_x = (win_w - scaled_w) // 2
        offset_y = (win_h - scaled_h) // 2
        self._viewport = pygame.Rect(offset_x, offset_y, scaled_w, scaled_h)

    def _present_frame(self) -> None:
        if self._viewport.width <= 0 or self._viewport.height <= 0:
            return
        scaled = pygame.transform.smoothscale(self.screen, (self._viewport.width, self._viewport.height))
        self.window.fill((6, 10, 22))
        self.window.blit(scaled, self._viewport.topleft)

    def _map_mouse_pos(self) -> tuple[tuple[int, int], bool]:
        mx, my = pygame.mouse.get_pos()
        if not self._viewport.collidepoint(mx, my) or self._scale <= 0:
            return (-10000, -10000), False
        x = int((mx - self._viewport.x) / self._scale)
        y = int((my - self._viewport.y) / self._scale)
        return (x, y), True

    def _activate_mode(self, mode_key: str) -> None:
        self.mode_config = get_mode_config(mode_key)
        self._release_controller()
        calibration_data = self.calibration_store.get_mode(mode_key)
        self.controller = self._build_controller(self.mode_config, calibration_data)
        self._reset_run()
        self.state = "playing"

    def _load_sounds(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent / "assets" / "sfx"
        self.sound_manager.load("jump", base_dir / "jump.wav")
        self.sound_manager.load("hit", base_dir / "hit.wav")

    def _reset_run(self) -> None:
        self.player = Player()
        self.level = Level(self.mode_config)
        self.score = 0
        self.coin_count = 0
        self.elapsed = 0.0
        self.current_speed = self.mode_config.speed
        self.next_prompt = "RUN"
        self.controls = MovementState(message=f"{self.mode_config.label} active.")
        self.camera_surface = None
        self.session_manager.reset_session()
        self.ui_manager.reset_summary_view()

    def _build_controller(self, mode_config: ModeConfig, calibration_data: dict[str, float] | None = None) -> BaseController:
        if mode_config.control_type == "hand":
            return HandController(mode_config, calibration_data=calibration_data)
        return PoseController(mode_config, calibration_data=calibration_data)

    def _release_controller(self) -> None:
        if self.controller is not None:
            self.controller.release_resources()
            self.controller = None

    def _start_calibration_session(self, mode_key: str) -> None:
        self.mode_config = get_mode_config(mode_key)
        self._release_controller()
        saved_profile = self.calibration_store.get_mode(mode_key)
        self.calibration_has_saved_profile = bool(saved_profile)
        self.controller = self._build_controller(self.mode_config, saved_profile)
        self._reset_calibration_capture()
        self.state = "calibration"

    def _reset_calibration_capture(self) -> None:
        self.calibration_samples = []
        self.calibration_progress = 0.0
        self.calibration_status = "Auto-calibrating height, arm length, dominant hand..."
        self.camera_surface = None

    def _finalize_calibration(self) -> None:
        if not self.calibration_samples:
            return

        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for sample in self.calibration_samples:
            for key, value in sample.items():
                totals[key] += float(value)
                counts[key] += 1

        averaged = {
            key: totals[key] / counts[key]
            for key in totals
            if counts[key] > 0
        }
        self.calibration_store.save_mode(self.mode_config.key, averaged)
        self.calibration_has_saved_profile = True
        if self.controller is not None:
            self.controller.apply_calibration(averaged)

        self._reset_run()
        self.state = "playing"

    def _speed_for_time(self) -> float:
        profile_speed_bonus = {
            "kids": 4.0,
            "elderly": 1.7,
            "disabled_leg": 2.8,
            "disabled_hand": 2.3,
        }
        profile_speed_ramp = {
            "kids": 0.36,
            "elderly": 0.24,
            "disabled_leg": 0.30,
            "disabled_hand": 0.27,
        }
        cap = profile_speed_bonus.get(self.mode_config.gesture_profile, 4.0)
        ramp = profile_speed_ramp.get(self.mode_config.gesture_profile, 0.30)
        return self.mode_config.speed + min(cap, self.elapsed * ramp)

    def _update_home(self, dt: float, events: list[pygame.event.Event]) -> None:
        self.menu_level.world_scroll += dt * 3.0
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.state = "mode_select"
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def _update_mode_select(self, dt: float, events: list[pygame.event.Event]) -> None:
        self.menu_level.world_scroll += dt * 3.0
        self.mode_select_screen.update(dt, events)

        for event in events:
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_UP, pygame.K_w):
                self.mode_select_screen.move_selection(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.mode_select_screen.move_selection(1)
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                key_to_number = {
                    pygame.K_1: 1,
                    pygame.K_2: 2,
                    pygame.K_3: 3,
                    pygame.K_4: 4,
                }
                self.mode_select_screen.select_by_number(key_to_number[event.key])
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.ui_manager.trigger_fade()
                self._start_calibration_session(self.mode_select_screen.current_mode_key())
            elif event.key == pygame.K_ESCAPE:
                self.ui_manager.trigger_fade()
                self.state = "home"

    def _update_calibration(self, _dt: float, events: list[pygame.event.Event]) -> None:
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                self._release_controller()
                self.state = "mode_select"
                return
            if event.key == pygame.K_r:
                self._reset_calibration_capture()
            if event.key == pygame.K_s and self.calibration_has_saved_profile:
                self._reset_run()
                self.state = "playing"
                return

        if self.controller is None:
            self._start_calibration_session(self.mode_config.key)
            return

        sample, status_message, preview = self.controller.get_calibration_sample()
        self.camera_surface = preview
        self.calibration_status = status_message

        if sample is not None:
            self.calibration_samples.append(sample)

        self.calibration_progress = min(1.0, len(self.calibration_samples) / float(self.calibration_target_samples))
        if len(self.calibration_samples) >= self.calibration_target_samples:
            self._finalize_calibration()

    def _update_playing(self, dt: float, events: list[pygame.event.Event]) -> None:
        if self.controller is None or self.player is None or self.level is None:
            self._activate_mode(self.mode_config.key)
            return

        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._release_controller()
                self.state = "mode_select"
                return

        self.controls, self.camera_surface = self.controller.get_movement()
        keys = pygame.key.get_pressed()

        self.elapsed += dt
        self.current_speed = self._speed_for_time()
        self.score += int(dt * (58 + (self.current_speed * 5.8)))

        player_input = self.player_controller.apply_input(self.player, self.controls, keys)
        if player_input.jumped:
            self.sound_manager.play("jump")

        self.player.update(dt, player_input.duck_hold, self.current_speed)
        self.level.update(dt, self.current_speed)

        if self.level.check_collision(self.player):
            self.session_manager.reset_combo()
            self.best_score = max(self.best_score, self.score)
            self.sound_manager.play("hit")
            self.ui_manager.trigger_fade(120)
            self.state = "game_over"
            return

        coins_gained = self.level.collect_coins(self.player)
        if coins_gained:
            self.coin_count += coins_gained
            combo_bonus = max(0, (self.session_manager.metrics.combo + coins_gained) * 4)
            self.score += (coins_gained * 25) + combo_bonus

        self.next_prompt = self.level.next_prompt()
        self.session_manager.update_metrics(
            dt=dt,
            speed=self.current_speed,
            tracked=self.controls.tracked,
            lane_changed=player_input.lane_changed,
            jumped=player_input.jumped,
            duck_hold=player_input.duck_hold,
            coins_gained=coins_gained,
        )

    def _update_game_over(self, _dt: float, events: list[pygame.event.Event]) -> None:
        restart_requested = False
        menu_requested = False
        stats_requested = False

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_r, pygame.K_RETURN, pygame.K_SPACE):
                    restart_requested = True
                elif event.key in (pygame.K_m, pygame.K_ESCAPE):
                    menu_requested = True
                elif event.key == pygame.K_v:
                    stats_requested = True

        if self.controller is not None:
            self.controls, self.camera_surface = self.controller.get_movement()
            if self.controls.jump:
                restart_requested = True

        if restart_requested:
            self.ui_manager.trigger_fade()
            self._reset_run()
            self.state = "playing"
            return
        if menu_requested:
            self.ui_manager.trigger_fade()
            self._release_controller()
            self.state = "mode_select"
            return
        if stats_requested:
            self.ui_manager.trigger_fade(96)
            self.state = "stats"

    def _update_stats(self, _dt: float, events: list[pygame.event.Event]) -> None:
        replay_requested = False
        menu_requested = False
        back_requested = False

        for event in events:
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_r, pygame.K_RETURN, pygame.K_SPACE):
                replay_requested = True
            elif event.key == pygame.K_m:
                menu_requested = True
            elif event.key in (pygame.K_ESCAPE, pygame.K_b, pygame.K_BACKSPACE):
                back_requested = True

        if replay_requested:
            self.ui_manager.trigger_fade()
            self._reset_run()
            self.state = "playing"
            return
        if menu_requested:
            self.ui_manager.trigger_fade()
            self._release_controller()
            self.state = "mode_select"
            return
        if back_requested:
            self.ui_manager.trigger_fade(96)
            self.state = "game_over"

    def _draw_frame(self) -> None:
        if self.state in ("playing", "game_over", "stats") and self.level is not None and self.player is not None:
            self.level.draw(self.screen)
            self.player.draw(self.screen)
            if self.state == "playing":
                self._draw_hud()
            if self.state == "game_over":
                self._draw_game_over_overlay()
            if self.state == "stats":
                self._draw_stats_overlay()
        elif self.state == "calibration":
            self.menu_level.draw(self.screen)
            self.calibration_screen.draw(
                self.screen,
                self.font_title,
                self.font_ui,
                self.font_body,
                self.mode_config.key,
                self.mode_config.label,
                self.calibration_progress,
                self.calibration_status,
                self.camera_surface,
                self.calibration_has_saved_profile,
            )
        else:
            self.menu_level.draw(self.screen)
            if self.state == "home":
                self.home_screen.draw(self.screen, self.font_title, self.font_body, self.best_score)
            else:
                self.mode_select_screen.draw(self.screen, self.font_title, self.font_ui, self.font_body)

        pygame.draw.rect(
            self.screen,
            (90, 244, 255),
            pygame.Rect(8, 8, WIDTH - 16, HEIGHT - 16),
            2,
            border_radius=24,
        )

    def _draw_hud(self) -> None:
        self.ui_manager.draw_hud(
            screen=self.screen,
            font_prompt=self.font_prompt,
            font_ui=self.font_ui,
            font_body=self.font_body,
            font_small=self.font_small,
            mode_label=self.mode_config.label,
            score=self.score,
            coin_count=self.coin_count,
            next_prompt=self.next_prompt,
            instruction=self.instructions_text,
            status_message=self.controls.message,
            gesture_label=self.controls.gesture,
            gesture_confidence=self.controls.confidence,
            failure_reason=self.controls.confidence_reason,
            metrics=self.session_manager.metrics,
            timer_text=self.session_manager.formatted_timer(),
            current_speed=self.current_speed,
            camera_surface=self.camera_surface,
        )

    def _draw_game_over_overlay(self) -> None:
        mapped_mouse, inside = self._map_mouse_pos()
        action = self.ui_manager.draw_summary(
            screen=self.screen,
            dt=self._frame_dt,
            mouse_pos=mapped_mouse,
            click=self._mouse_clicked and inside,
            font_title=self.font_title,
            font_ui=self.font_ui,
            font_body=self.font_body,
            font_small=self.font_small,
            mode_label=self.mode_config.label,
            score=self.score,
            best_score=self.best_score,
            coin_count=self.coin_count,
            metrics=self.session_manager.metrics,
            timer_text=self.session_manager.formatted_timer(),
            session_history=self.session_manager.history_points(),
            session_target_seconds=self.session_manager.session_target_seconds,
        )
        if action == "replay":
            self.ui_manager.trigger_fade()
            self._reset_run()
            self.state = "playing"
        elif action == "mode":
            self.ui_manager.trigger_fade()
            self._release_controller()
            self.state = "mode_select"
        elif action == "stats":
            self.ui_manager.trigger_fade(96)
            self.state = "stats"

    def _draw_stats_overlay(self) -> None:
        mapped_mouse, inside = self._map_mouse_pos()
        action = self.ui_manager.draw_stats_page(
            screen=self.screen,
            dt=self._frame_dt,
            mouse_pos=mapped_mouse,
            click=self._mouse_clicked and inside,
            font_title=self.font_title,
            font_ui=self.font_ui,
            font_body=self.font_body,
            font_small=self.font_small,
            mode_label=self.mode_config.label,
            score=self.score,
            best_score=self.best_score,
            coin_count=self.coin_count,
            metrics=self.session_manager.metrics,
            timer_text=self.session_manager.formatted_timer(),
            session_history=self.session_manager.history_points(),
            session_target_seconds=self.session_manager.session_target_seconds,
        )
        if action == "replay":
            self.ui_manager.trigger_fade()
            self._reset_run()
            self.state = "playing"
        elif action == "mode":
            self.ui_manager.trigger_fade()
            self._release_controller()
            self.state = "mode_select"
        elif action == "back":
            self.ui_manager.trigger_fade(96)
            self.state = "game_over"
