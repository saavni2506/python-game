from __future__ import annotations

import math
from typing import Callable, Sequence

import pygame

from core.game_manager import SessionHistorySample, SessionMetrics


class UIManager:
    def __init__(self) -> None:
        self._intensity_display = 0.0
        self._calories_display = 0.0
        self._progress_display = 0.0
        self._button_pop: dict[str, float] = {}
        self._fade_alpha = 0.0

    def trigger_fade(self, alpha: float = 180.0) -> None:
        self._fade_alpha = max(self._fade_alpha, alpha)

    def reset_summary_view(self) -> None:
        self._button_pop.clear()

    def update(self, dt: float) -> None:
        self._fade_alpha = max(0.0, self._fade_alpha - (dt * 420.0))
        for key in list(self._button_pop.keys()):
            self._button_pop[key] = max(0.0, self._button_pop[key] - dt)
            if self._button_pop[key] <= 0.0:
                del self._button_pop[key]

    def draw_hud(
        self,
        screen: pygame.Surface,
        font_prompt: pygame.font.Font,
        font_ui: pygame.font.Font,
        font_body: pygame.font.Font,
        font_small: pygame.font.Font,
        mode_label: str,
        score: int,
        coin_count: int,
        next_prompt: str,
        instruction: str,
        status_message: str,
        gesture_label: str,
        gesture_confidence: float,
        failure_reason: str,
        metrics: SessionMetrics,
        timer_text: str,
        current_speed: float,
        camera_surface: pygame.Surface | None,
    ) -> None:
        self._intensity_display += (metrics.intensity - self._intensity_display) * 0.15
        self._calories_display += (metrics.calories - self._calories_display) * 0.12
        self._progress_display += (metrics.progress - self._progress_display) * 0.18

        prompt_panel = pygame.Rect((screen.get_width() // 2) - 170, 18, 340, 74)
        self._draw_glass_panel(screen, prompt_panel, (80, 244, 255), (20, 36, 58), 164, 18)
        prompt_text = font_prompt.render(f"{next_prompt}!", True, (232, 255, 255))
        screen.blit(prompt_text, prompt_text.get_rect(center=(prompt_panel.centerx, prompt_panel.centery + 2)))

        top_right = pygame.Rect(screen.get_width() - 372, 20, 352, 208)
        self._draw_glass_panel(screen, top_right, (82, 247, 255), (10, 24, 42), 168, 20)

        combo_text = font_ui.render(f"Combo: {metrics.combo}", True, (228, 255, 255))
        coins_text = font_body.render(f"Coins: {coin_count}", True, (236, 249, 255))
        calories_text = font_body.render(f"Calories Burned: {self._calories_display:0.0f} kcal", True, (241, 249, 255))
        speed_text = font_body.render(f"Speed: {current_speed:0.1f} m/s", True, (225, 248, 255))
        timer_caption = font_small.render("Session Timer", True, (176, 224, 255))
        timer_value = font_ui.render(timer_text, True, (216, 248, 255))

        screen.blit(combo_text, (top_right.x + 20, top_right.y + 18))
        screen.blit(coins_text, (top_right.right - 150, top_right.y + 24))
        screen.blit(calories_text, (top_right.x + 20, top_right.y + 62))
        screen.blit(speed_text, (top_right.x + 20, top_right.y + 84))
        screen.blit(timer_caption, (top_right.x + 20, top_right.y + 162))
        screen.blit(timer_value, (top_right.right - 120, top_right.y + 152))

        intensity_label = font_small.render("Movement Intensity", True, (172, 227, 255))
        screen.blit(intensity_label, (top_right.x + 20, top_right.y + 112))
        bar_bg = pygame.Rect(top_right.x + 20, top_right.y + 136, top_right.width - 40, 14)
        pygame.draw.rect(screen, (16, 38, 58), bar_bg, border_radius=8)
        fill_width = int((bar_bg.width - 4) * max(0.0, min(1.0, self._intensity_display)))
        fill = pygame.Rect(bar_bg.x + 2, bar_bg.y + 2, fill_width, bar_bg.height - 4)
        pygame.draw.rect(screen, (72, 244, 255), fill, border_radius=7)
        pygame.draw.rect(screen, (198, 239, 255), bar_bg, 1, border_radius=8)

        bottom_panel = pygame.Rect(20, screen.get_height() - 140, screen.get_width() - 40, 120)
        self._draw_glass_panel(screen, bottom_panel, (109, 92, 255), (8, 18, 36), 172, 18)

        mode_text = font_ui.render(mode_label, True, (178, 223, 255))
        score_text = font_ui.render(f"Score: {score}", True, (236, 251, 255))
        screen.blit(mode_text, (bottom_panel.x + 18, bottom_panel.y + 10))
        screen.blit(score_text, (bottom_panel.right - 210, bottom_panel.y + 10))

        progress_bg = pygame.Rect(bottom_panel.x + 18, bottom_panel.y + 50, bottom_panel.width - 36, 10)
        pygame.draw.rect(screen, (18, 42, 66), progress_bg, border_radius=6)
        progress_fill_w = int((progress_bg.width - 4) * max(0.0, min(1.0, self._progress_display)))
        progress_fill = pygame.Rect(progress_bg.x + 2, progress_bg.y + 2, progress_fill_w, progress_bg.height - 4)
        pygame.draw.rect(screen, (118, 84, 255), progress_fill, border_radius=6)

        instruction_text = font_body.render(instruction, True, (222, 238, 255))
        screen.blit(instruction_text, (bottom_panel.x + 18, bottom_panel.y + 62))

        status_line = failure_reason or status_message
        status_line = self._truncate_to_width(status_line, font_small, bottom_panel.width - 36)
        status_text = font_small.render(status_line, True, (186, 229, 255))
        screen.blit(status_text, (bottom_panel.x + 18, bottom_panel.y + 82))

        safe_label = gesture_label or "NONE"
        conf_pct = int(max(0.0, min(1.0, gesture_confidence)) * 100)
        gesture_line = self._truncate_to_width(
            f"Gesture: {safe_label} ({conf_pct}%)",
            font_small,
            bottom_panel.width - 36,
        )
        gesture_text = font_small.render(gesture_line, True, (186, 229, 255))
        screen.blit(gesture_text, (bottom_panel.x + 18, bottom_panel.y + 100))

        if camera_surface is not None:
            preview = pygame.transform.smoothscale(camera_surface, (220, 164))
            preview_rect = preview.get_rect(topleft=(20, 20))
            screen.blit(preview, preview_rect)
            pygame.draw.rect(screen, (82, 247, 255), preview_rect, 2, border_radius=10)

        self._draw_fade(screen)

    def draw_summary(
        self,
        screen: pygame.Surface,
        dt: float,
        mouse_pos: tuple[int, int],
        click: bool,
        font_title: pygame.font.Font,
        font_ui: pygame.font.Font,
        font_body: pygame.font.Font,
        font_small: pygame.font.Font,
        mode_label: str,
        score: int,
        best_score: int,
        coin_count: int,
        metrics: SessionMetrics,
        timer_text: str,
        session_history: Sequence[SessionHistorySample],
        session_target_seconds: float,
    ) -> str | None:
        _ = dt
        _ = session_history
        _ = session_target_seconds

        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((5, 10, 30, 220))
        screen.blit(overlay, (0, 0))

        panel_w = min(760, screen.get_width() - 120)
        panel_h = min(520, screen.get_height() - 120)
        panel = pygame.Rect(
            (screen.get_width() - panel_w) // 2,
            (screen.get_height() - panel_h) // 2,
            panel_w,
            panel_h,
        )
        self._draw_glass_panel(screen, panel, (87, 238, 255), (10, 22, 42), 186, 24)

        left_col_w = max(180, int(panel.width * 0.26))
        right_col_x = panel.x + left_col_w + 24
        right_col_w = panel.right - right_col_x - 24

        title = font_title.render("Session Summary", True, (214, 249, 255))
        title_rect = title.get_rect(midtop=(right_col_x + (right_col_w // 2), panel.y + 22))
        screen.blit(title, title_rect)

        mode_text = self._truncate_to_width(mode_label, font_small, right_col_w)
        mode_caption = font_small.render(mode_text, True, (154, 224, 255))
        screen.blit(mode_caption, mode_caption.get_rect(midtop=(title_rect.centerx, title_rect.bottom + 4)))

        trophy_panel = pygame.Rect(panel.x + 22, panel.y + 22, left_col_w - 44, 176)
        self._draw_glass_panel(screen, trophy_panel, (120, 226, 255), (8, 18, 34), 168, 16)
        trophy_x = trophy_panel.centerx - 50
        trophy_y = trophy_panel.y + 12
        self._draw_trophy(screen, trophy_x, trophy_y)

        divider_x = panel.x + left_col_w + 12
        pygame.draw.line(
            screen,
            (96, 196, 255),
            (divider_x, panel.y + 24),
            (divider_x, panel.bottom - 124),
            1,
        )

        content_top = title_rect.bottom + 28
        line_gap = font_body.get_height() + 10
        lines = [
            f"Mode: {mode_label}",
            f"Score: {score}",
            f"Best Score: {best_score}",
            f"Coins Collected: {coin_count}",
            f"Calories Burned: {metrics.calories:0.0f} kcal",
            f"Session Time: {timer_text}",
        ]
        for idx, text in enumerate(lines):
            label = font_body.render(self._truncate_to_width(text, font_body, right_col_w), True, (228, 243, 255))
            screen.blit(label, (right_col_x, content_top + (idx * line_gap)))

        actions = [
            ("replay", "Replay", (84, 255, 136)),
            ("mode", "Change Mode", (88, 184, 255)),
            ("stats", "View Stats", (202, 110, 255)),
        ]
        clicked_action = self._draw_action_buttons(
            screen=screen,
            mouse_pos=mouse_pos,
            click=click,
            font_ui=font_ui,
            actions=actions,
            start_x=panel.centerx - 304,
            y=panel.bottom - 104,
            button_w=196,
            button_h=48,
            gap=16,
        )

        hint = font_small.render("Enter/R: replay   V: stats   M/Esc: mode select   Click buttons", True, (194, 227, 255))
        screen.blit(hint, hint.get_rect(center=(panel.centerx, panel.bottom - 16)))

        self._draw_fade(screen)
        return clicked_action

    def draw_stats_page(
        self,
        screen: pygame.Surface,
        dt: float,
        mouse_pos: tuple[int, int],
        click: bool,
        font_title: pygame.font.Font,
        font_ui: pygame.font.Font,
        font_body: pygame.font.Font,
        font_small: pygame.font.Font,
        mode_label: str,
        score: int,
        best_score: int,
        coin_count: int,
        metrics: SessionMetrics,
        timer_text: str,
        session_history: Sequence[SessionHistorySample],
        session_target_seconds: float,
    ) -> str | None:
        _ = dt

        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((4, 8, 24, 236))
        screen.blit(overlay, (0, 0))
        self._draw_stats_backdrop(screen)

        samples = list(session_history)
        target_label = self._format_seconds(session_target_seconds)
        elapsed_minutes = max(metrics.elapsed_seconds / 60.0, 1.0 / 60.0)
        avg_intensity = 0.0
        peak_intensity = 0.0
        if samples:
            avg_intensity = sum(sample.intensity for sample in samples) / len(samples)
            peak_intensity = max(sample.intensity for sample in samples)
        score_rate = score / elapsed_minutes
        burn_rate = metrics.calories / elapsed_minutes
        x_max = max(6.0, metrics.elapsed_seconds, session_target_seconds * 0.33)
        calories_max = max(1.0, max((sample.calories for sample in samples), default=metrics.calories) * 1.15)

        header = pygame.Rect(34, 24, screen.get_width() - 68, 92)
        self._draw_glass_panel(screen, header, (82, 247, 255), (8, 18, 34), 156, 20)

        title = font_title.render("Session Intelligence", True, (233, 250, 255))
        subtitle = font_body.render(f"{timer_text} / {target_label}", True, (176, 224, 255))
        screen.blit(title, (header.x + 20, header.y + 12))
        screen.blit(subtitle, (header.x + 24, header.y + 60))

        top_badge = pygame.Rect(header.right - 206, header.y + 24, 182, 42)
        self._draw_glass_panel(screen, top_badge, (255, 198, 84), (22, 24, 18), 170, 14)
        badge_text = font_small.render("Session Complete", True, (255, 241, 204))
        screen.blit(badge_text, badge_text.get_rect(center=top_badge.center))

        content_y = header.bottom + 20
        content_h = screen.get_height() - content_y - 76
        left_panel = pygame.Rect(34, content_y, 312, content_h)
        self._draw_glass_panel(screen, left_panel, (92, 224, 255), (8, 18, 34), 150, 24)

        hero = pygame.Rect(left_panel.x + 18, left_panel.y + 18, left_panel.width - 36, 146)
        self._draw_glass_panel(screen, hero, (255, 206, 92), (18, 18, 28), 182, 18)
        self._draw_trophy(screen, hero.x + 18, hero.y + 16)
        hero_text_x = hero.x + 120
        hero_text_w = hero.right - hero_text_x - 16
        hero_gap = 6
        hero_score = font_ui.render(
            self._truncate_to_width(f"Score {score}", font_ui, hero_text_w),
            True,
            (255, 249, 236),
        )
        hero_meta = font_body.render(
            self._truncate_to_width(f"Calories {metrics.calories:0.1f} kcal", font_body, hero_text_w),
            True,
            (241, 230, 198),
        )
        hero_time = font_body.render(
            self._truncate_to_width(f"Time {timer_text}", font_body, hero_text_w),
            True,
            (229, 221, 196),
        )
        hero_block_h = hero_score.get_height() + hero_meta.get_height() + hero_time.get_height() + (hero_gap * 2)
        hero_block_y = hero.centery - (hero_block_h // 2) - 2
        screen.blit(hero_score, (hero_text_x, hero_block_y))
        screen.blit(hero_meta, (hero_text_x, hero_block_y + hero_score.get_height() + hero_gap))
        screen.blit(
            hero_time,
            (
                hero_text_x,
                hero_block_y + hero_score.get_height() + hero_meta.get_height() + (hero_gap * 2),
            ),
        )

        metric_rects = self._build_metric_grid(
            pygame.Rect(left_panel.x + 18, hero.bottom + 16, left_panel.width - 36, left_panel.bottom - hero.bottom - 30),
            columns=2,
            rows=3,
            gap=12,
        )
        metric_cards = [
            ("Best Score", f"{best_score}", (88, 184, 255)),
            ("Coins", f"{coin_count}", (255, 206, 92)),
            ("Avg Intensity", f"{avg_intensity * 100:0.0f}%", (84, 244, 255)),
            ("Peak Intensity", f"{peak_intensity * 100:0.0f}%", (202, 110, 255)),
            ("Score / Min", f"{score_rate:0.0f}", (120, 255, 170)),
            ("Burn Rate", f"{burn_rate:0.2f} kcal/m", (255, 138, 118)),
        ]
        for rect, (label, value, accent) in zip(metric_rects, metric_cards):
            self._draw_metric_card(screen, rect, font_small, font_ui, label, value, accent)

        charts_area = pygame.Rect(366, content_y, screen.get_width() - 400, content_h)
        chart_gap = 18
        top_h = min(260, max(210, int(charts_area.height * 0.46)))
        bottom_h = charts_area.height - top_h - chart_gap
        half_w = (charts_area.width - chart_gap) // 2

        calories_rect = pygame.Rect(charts_area.x, charts_area.y, half_w, top_h)
        intensity_rect = pygame.Rect(charts_area.x + half_w + chart_gap, charts_area.y, charts_area.width - half_w - chart_gap, top_h)
        time_rect = pygame.Rect(charts_area.x, charts_area.y + top_h + chart_gap, charts_area.width, bottom_h)

        self._draw_line_chart(
            screen=screen,
            rect=calories_rect,
            font_body=font_body,
            font_small=font_small,
            title="Calories Burn Curve",
            value_label=f"{metrics.calories:0.1f} kcal",
            accent_rgb=(255, 198, 84),
            samples=samples,
            value_getter=lambda sample: sample.calories,
            max_value=calories_max,
            x_max=x_max,
            left_footer="Warm-up",
            right_footer="Current",
            y_label_formatter=lambda value: f"{value:0.1f}" if value < 10.0 else self._default_y_label(value),
        )
        self._draw_line_chart(
            screen=screen,
            rect=intensity_rect,
            font_body=font_body,
            font_small=font_small,
            title="Movement Intensity Signal",
            value_label=f"{metrics.intensity * 100:0.0f}% live",
            accent_rgb=(84, 244, 255),
            samples=samples,
            value_getter=lambda sample: sample.intensity,
            max_value=1.0,
            x_max=x_max,
            left_footer="Calm",
            right_footer="Peak",
            reference_value=0.65,
            y_label_formatter=lambda value: f"{int(round(value * 100.0))}%",
        )
        self._draw_line_chart(
            screen=screen,
            rect=time_rect,
            font_body=font_body,
            font_small=font_small,
            title="Session Time Progress",
            value_label=f"{timer_text} of {target_label}",
            accent_rgb=(202, 110, 255),
            samples=samples,
            value_getter=lambda sample: sample.progress,
            max_value=1.0,
            x_max=x_max,
            left_footer="Start",
            right_footer="Target",
            reference_value=1.0,
            y_label_formatter=lambda value: f"{int(round(value * 100.0))}%",
        )

        actions = [
            ("back", "Back", (255, 198, 84)),
            ("replay", "Replay", (84, 255, 136)),
            ("mode", "Change Mode", (88, 184, 255)),
        ]
        clicked_action = self._draw_action_buttons(
            screen=screen,
            mouse_pos=mouse_pos,
            click=click,
            font_ui=font_ui,
            actions=actions,
            start_x=screen.get_width() - 634,
            y=screen.get_height() - 64,
            button_w=184,
            button_h=44,
            gap=12,
        )

        footer = font_small.render("ESC/B: back   R/Enter: replay   M: mode select", True, (182, 223, 255))
        screen.blit(footer, (42, screen.get_height() - 50))

        self._draw_fade(screen)
        return clicked_action

    def _draw_stats_backdrop(self, screen: pygame.Surface) -> None:
        backdrop = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        width = screen.get_width()
        height = screen.get_height()

        pygame.draw.circle(backdrop, (84, 244, 255, 34), (width - 220, 120), 240)
        pygame.draw.circle(backdrop, (202, 110, 255, 26), (width - 120, height - 140), 280)
        pygame.draw.circle(backdrop, (255, 198, 84, 18), (170, 190), 160)

        for x in range(0, width, 80):
            pygame.draw.line(backdrop, (82, 247, 255, 10), (x, 112), (x, height - 36), 1)
        for y in range(112, height, 52):
            pygame.draw.line(backdrop, (92, 126, 255, 10), (34, y), (width - 34, y), 1)

        pygame.draw.line(backdrop, (82, 247, 255, 20), (34, height - 104), (width - 34, 118), 2)
        pygame.draw.line(backdrop, (202, 110, 255, 16), (240, height - 64), (width - 80, height - 190), 2)
        screen.blit(backdrop, (0, 0))

    def _draw_action_buttons(
        self,
        screen: pygame.Surface,
        mouse_pos: tuple[int, int],
        click: bool,
        font_ui: pygame.font.Font,
        actions: Sequence[tuple[str, str, tuple[int, int, int]]],
        start_x: int,
        y: int,
        button_w: int,
        button_h: int,
        gap: int,
    ) -> str | None:
        clicked_action: str | None = None
        for idx, (key, label, color) in enumerate(actions):
            base = pygame.Rect(start_x + (idx * (button_w + gap)), y, button_w, button_h)
            hovered = base.collidepoint(mouse_pos)
            pop = self._button_pop.get(key, 0.0)
            hover_scale = 1.05 if hovered else 1.0
            pop_scale = 1.0 + (0.08 * math.sin((0.16 - pop) * 18.0)) if pop > 0.0 else 1.0
            scale = hover_scale * pop_scale
            button_rect = self._scaled_rect(base, scale)
            pressed = click and button_rect.collidepoint(mouse_pos)
            if pressed:
                self._button_pop[key] = 0.16
                clicked_action = key

            self._draw_glass_panel(screen, button_rect, color, (12, 26, 44), 192, 12)
            inner = pygame.Rect(button_rect.x + 10, button_rect.y + 10, button_rect.width - 20, max(8, button_rect.height // 4))
            pygame.draw.rect(screen, (color[0], color[1], color[2], 26), inner, border_radius=10)
            text = font_ui.render(label, True, (237, 249, 255))
            screen.blit(text, text.get_rect(center=button_rect.center))
        return clicked_action

    def _draw_glass_panel(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        border_rgb: tuple[int, int, int],
        fill_rgb: tuple[int, int, int],
        fill_alpha: int,
        radius: int,
    ) -> None:
        glow_pad = 14
        glow = pygame.Surface((rect.width + (glow_pad * 2), rect.height + (glow_pad * 2)), pygame.SRCALPHA)
        pygame.draw.rect(
            glow,
            (border_rgb[0], border_rgb[1], border_rgb[2], 44),
            glow.get_rect(),
            width=3,
            border_radius=radius + 8,
        )
        screen.blit(glow, (rect.x - glow_pad, rect.y - glow_pad))

        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel, (fill_rgb[0], fill_rgb[1], fill_rgb[2], fill_alpha), panel.get_rect(), border_radius=radius)
        pygame.draw.rect(panel, (border_rgb[0], border_rgb[1], border_rgb[2], 200), panel.get_rect(), 2, border_radius=radius)
        screen.blit(panel, rect.topleft)

    def _draw_trophy(self, screen: pygame.Surface, x: int, y: int) -> None:
        cup = pygame.Rect(x + 20, y + 18, 72, 54)
        pygame.draw.rect(screen, (255, 210, 72), cup, border_radius=18)
        pygame.draw.rect(screen, (255, 239, 136), cup, 2, border_radius=18)
        pygame.draw.circle(screen, (255, 196, 64), (x + 20, y + 45), 14, 4)
        pygame.draw.circle(screen, (255, 196, 64), (x + 92, y + 45), 14, 4)
        pygame.draw.rect(screen, (255, 196, 64), pygame.Rect(x + 49, y + 72, 14, 18), border_radius=4)
        pygame.draw.rect(screen, (182, 204, 255), pygame.Rect(x + 36, y + 90, 40, 12), border_radius=6)

    def _draw_metric_card(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        font_small: pygame.font.Font,
        font_ui: pygame.font.Font,
        label: str,
        value: str,
        accent_rgb: tuple[int, int, int],
    ) -> None:
        self._draw_glass_panel(screen, rect, accent_rgb, (8, 18, 34), 152, 14)
        accent_bar = pygame.Rect(rect.x + 10, rect.y + 10, rect.width - 20, 6)
        pygame.draw.rect(screen, accent_rgb, accent_bar, border_radius=4)
        label_text = font_small.render(label, True, (190, 229, 255))
        value_text = font_ui.render(self._truncate_to_width(value, font_ui, rect.width - 24), True, (239, 248, 255))
        screen.blit(label_text, (rect.x + 12, rect.y + 24))
        screen.blit(value_text, (rect.x + 12, rect.y + 48))

    def _draw_line_chart(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        font_body: pygame.font.Font,
        font_small: pygame.font.Font,
        title: str,
        value_label: str,
        accent_rgb: tuple[int, int, int],
        samples: Sequence[SessionHistorySample],
        value_getter: Callable[[SessionHistorySample], float],
        max_value: float,
        x_max: float,
        left_footer: str,
        right_footer: str,
        reference_value: float | None = None,
        y_label_formatter: Callable[[float], str] | None = None,
    ) -> None:
        self._draw_glass_panel(screen, rect, accent_rgb, (8, 18, 34), 148, 16)

        title_text = font_small.render(title, True, (190, 229, 255))
        value_text = font_body.render(self._truncate_to_width(value_label, font_body, rect.width - 160), True, (240, 248, 255))
        screen.blit(title_text, (rect.x + 16, rect.y + 12))
        screen.blit(value_text, value_text.get_rect(topright=(rect.right - 16, rect.y + 10)))

        chart_area = pygame.Rect(rect.x + 16, rect.y + 42, rect.width - 32, rect.height - 74)
        y_label_w = max(42, min(58, chart_area.width // 6))
        x_label_h = 28
        plot_rect = pygame.Rect(
            chart_area.x + y_label_w + 10,
            chart_area.y + 2,
            chart_area.width - y_label_w - 14,
            chart_area.height - x_label_h - 6,
        )
        if plot_rect.width <= 4 or plot_rect.height <= 4:
            return

        plot_bg = pygame.Surface(plot_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(plot_bg, (2, 5, 12, 235), plot_bg.get_rect(), border_radius=10)
        screen.blit(plot_bg, plot_rect.topleft)

        grid = pygame.Surface(plot_rect.size, pygame.SRCALPHA)
        y_ticks = 6
        x_ticks = 8
        for idx in range(y_ticks):
            y = int((plot_rect.height - 1) * (idx / max(1, y_ticks - 1)))
            alpha = 44 if idx in (0, y_ticks - 1) else 26
            pygame.draw.line(grid, (255, 255, 255, alpha), (0, y), (plot_rect.width, y), 1)
        for idx in range(x_ticks):
            x = int((plot_rect.width - 1) * (idx / max(1, x_ticks - 1)))
            pygame.draw.line(grid, (255, 255, 255, 10), (x, 0), (x, plot_rect.height), 1)
        screen.blit(grid, plot_rect.topleft)

        if reference_value is not None:
            ref_ratio = max(0.0, min(1.0, reference_value / max(0.001, max_value)))
            ref_y = plot_rect.bottom - 1 - int(ref_ratio * (plot_rect.height - 1))
            pygame.draw.line(screen, (accent_rgb[0], accent_rgb[1], accent_rgb[2], 80), (plot_rect.x, ref_y), (plot_rect.right, ref_y), 1)

        points = self._build_chart_points(plot_rect, samples, value_getter, max_value=max_value, x_max=x_max)
        if len(points) == 1:
            points = [(plot_rect.x, points[0][1]), points[0]]

        if len(points) >= 2:
            smooth_points = self._smooth_chart_points(points, plot_rect)
            self._draw_glow_path(screen, smooth_points, accent_rgb)
            pygame.draw.circle(screen, accent_rgb, smooth_points[-1], 5)
            pygame.draw.circle(screen, (236, 248, 255), smooth_points[-1], 2)

        pygame.draw.rect(screen, (162, 184, 214), plot_rect, 1, border_radius=8)

        formatter = y_label_formatter or self._default_y_label
        for idx in range(y_ticks):
            ratio = 1.0 - (idx / max(1, y_ticks - 1))
            value = max_value * ratio
            y = plot_rect.y + int((plot_rect.height - 1) * (idx / max(1, y_ticks - 1)))
            label = font_small.render(formatter(value), True, (126, 137, 154))
            screen.blit(label, label.get_rect(midright=(plot_rect.x - 10, y)))

        for idx in range(x_ticks):
            ratio = idx / max(1, x_ticks - 1)
            x = plot_rect.x + int((plot_rect.width - 1) * ratio)
            pygame.draw.line(screen, (74, 84, 102), (x, plot_rect.bottom), (x, plot_rect.bottom + 6), 1)
            tick_seconds = x_max * ratio
            tick_label = font_small.render(self._format_axis_time(tick_seconds), True, (112, 124, 144))
            screen.blit(tick_label, tick_label.get_rect(midtop=(x, plot_rect.bottom + 8)))

        left_text = font_small.render(left_footer, True, (150, 159, 176))
        right_text = font_small.render(right_footer, True, (150, 159, 176))
        screen.blit(left_text, (chart_area.x, rect.bottom - 22))
        screen.blit(right_text, right_text.get_rect(topright=(chart_area.right, rect.bottom - 22)))

    def _build_chart_points(
        self,
        plot_rect: pygame.Rect,
        samples: Sequence[SessionHistorySample],
        value_getter: Callable[[SessionHistorySample], float],
        max_value: float,
        x_max: float,
    ) -> list[tuple[int, int]]:
        if not samples:
            return []

        safe_x_max = max(1.0, x_max)
        safe_y_max = max(0.001, max_value)
        points: list[tuple[int, int]] = []
        for sample in samples:
            x_ratio = max(0.0, min(1.0, sample.elapsed_seconds / safe_x_max))
            y_ratio = max(0.0, min(1.0, value_getter(sample) / safe_y_max))
            x = plot_rect.x + int(x_ratio * (plot_rect.width - 1))
            y = plot_rect.bottom - 1 - int(y_ratio * (plot_rect.height - 1))
            points.append((x, y))
        return points

    def _draw_glow_path(
        self,
        screen: pygame.Surface,
        points: Sequence[tuple[int, int]],
        color: tuple[int, int, int],
    ) -> None:
        if len(points) < 2:
            return

        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        pad = 14
        glow_rect = pygame.Rect(min_x - pad, min_y - pad, (max_x - min_x) + (pad * 2) + 1, (max_y - min_y) + (pad * 2) + 1)
        glow = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        local_points = [(x - glow_rect.x, y - glow_rect.y) for x, y in points]

        pygame.draw.lines(glow, (color[0], color[1], color[2], 32), False, local_points, 10)
        pygame.draw.lines(glow, (color[0], color[1], color[2], 54), False, local_points, 6)
        pygame.draw.lines(glow, color, False, local_points, 2)
        screen.blit(glow, glow_rect.topleft)

    def _smooth_chart_points(
        self,
        points: Sequence[tuple[int, int]],
        bounds: pygame.Rect,
        samples_per_segment: int = 10,
    ) -> list[tuple[int, int]]:
        if len(points) < 3:
            return list(points)

        def clamp_point(x: float, y: float) -> tuple[int, int]:
            cx = max(bounds.left, min(bounds.right - 1, int(round(x))))
            cy = max(bounds.top, min(bounds.bottom - 1, int(round(y))))
            return (cx, cy)

        extended = [points[0], *points, points[-1]]
        smooth: list[tuple[int, int]] = [points[0]]
        for idx in range(1, len(extended) - 2):
            p0 = extended[idx - 1]
            p1 = extended[idx]
            p2 = extended[idx + 1]
            p3 = extended[idx + 2]
            for step in range(1, samples_per_segment + 1):
                t = step / float(samples_per_segment)
                t2 = t * t
                t3 = t2 * t
                x = 0.5 * (
                    (2 * p1[0])
                    + (-p0[0] + p2[0]) * t
                    + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2 * p1[1])
                    + (-p0[1] + p2[1]) * t
                    + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                point = clamp_point(x, y)
                if point != smooth[-1]:
                    smooth.append(point)
        return smooth

    @staticmethod
    def _default_y_label(value: float) -> str:
        if value >= 1000.0:
            if value >= 10000.0:
                return f"{value / 1000.0:0.0f}K"
            return f"{value / 1000.0:0.1f}K"
        if value >= 100.0:
            return f"{value:0.0f}"
        if value >= 10.0:
            return f"{value:0.0f}"
        if value >= 1.0:
            return f"{value:0.1f}"
        return f"{value:0.2f}".rstrip("0").rstrip(".") or "0"

    @staticmethod
    def _format_axis_time(total_seconds: float) -> str:
        total = max(0, int(round(total_seconds)))
        minutes = total // 60
        seconds = total % 60
        return f"{minutes}:{seconds:02}"

    @staticmethod
    def _build_metric_grid(area: pygame.Rect, columns: int, rows: int, gap: int) -> list[pygame.Rect]:
        if columns <= 0 or rows <= 0:
            return []
        card_w = (area.width - (gap * (columns - 1))) // columns
        card_h = (area.height - (gap * (rows - 1))) // rows
        rects: list[pygame.Rect] = []
        for row in range(rows):
            for col in range(columns):
                x = area.x + (col * (card_w + gap))
                y = area.y + (row * (card_h + gap))
                rects.append(pygame.Rect(x, y, card_w, card_h))
        return rects

    def _draw_fade(self, screen: pygame.Surface) -> None:
        if self._fade_alpha <= 0.5:
            return
        fade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        fade.fill((0, 0, 0, int(self._fade_alpha)))
        screen.blit(fade, (0, 0))

    @staticmethod
    def _scaled_rect(rect: pygame.Rect, scale: float) -> pygame.Rect:
        width = max(8, int(rect.width * scale))
        height = max(8, int(rect.height * scale))
        return pygame.Rect(rect.centerx - (width // 2), rect.centery - (height // 2), width, height)

    @staticmethod
    def _truncate_to_width(text: str, font: pygame.font.Font, max_width: int) -> str:
        if font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        clipped = text
        while clipped and font.size(clipped + ellipsis)[0] > max_width:
            clipped = clipped[:-1]
        return (clipped + ellipsis) if clipped else ellipsis

    @staticmethod
    def _format_seconds(total_seconds: float) -> str:
        total = max(0, int(total_seconds))
        minutes = total // 60
        seconds = total % 60
        return f"{minutes:02}:{seconds:02}"
