from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class ModeCardStyle:
    mode_key: str
    title: str
    subtitle: str
    detail: str
    accent: tuple[int, int, int]


CARD_STYLES: dict[str, ModeCardStyle] = {
    "kids": ModeCardStyle(
        mode_key="kids",
        title="Kids Mode",
        subtitle="Fun energetic pose icon",
        detail="High engagement rating",
        accent=(74, 255, 138),
    ),
    "elderly": ModeCardStyle(
        mode_key="elderly",
        title="Elderly Mode",
        subtitle="Yoga pose icon",
        detail="Medium intensity",
        accent=(88, 192, 255),
    ),
    "disabled_leg": ModeCardStyle(
        mode_key="disabled_leg",
        title="Leg-Free Mode",
        subtitle="Wheelchair icon",
        detail="Upper-body control | Moderate difficulty",
        accent=(78, 232, 255),
    ),
    "disabled_hand": ModeCardStyle(
        mode_key="disabled_hand",
        title="Hand-Free Mode",
        subtitle="Pose-only control",
        detail="Red theme | Low sensitivity",
        accent=(255, 86, 124),
    ),
}


class ModeManager:
    def __init__(self, mode_keys: list[str]) -> None:
        self.mode_keys = list(mode_keys)
        self.selected_index = 0
        self.hover_index = -1
        self.float_time = 0.0

    def current_mode_key(self) -> str:
        return self.mode_keys[self.selected_index]

    def move_selection(self, delta: int) -> None:
        self.selected_index = (self.selected_index + delta) % len(self.mode_keys)

    def select_by_number(self, number: int) -> None:
        index = number - 1
        if 0 <= index < len(self.mode_keys):
            self.selected_index = index

    def style_for_index(self, index: int) -> ModeCardStyle:
        mode_key = self.mode_keys[index]
        return CARD_STYLES.get(mode_key, CARD_STYLES["kids"])

    def style_for_mode(self, mode_key: str) -> ModeCardStyle:
        return CARD_STYLES.get(mode_key, CARD_STYLES["kids"])

    def update(self, dt: float, mouse_pos: tuple[int, int], card_rects: list[pygame.Rect]) -> None:
        self.float_time += dt
        self.hover_index = -1
        for idx, rect in enumerate(card_rects):
            if rect.collidepoint(mouse_pos):
                self.hover_index = idx
                break

    def click_select(self, mouse_pos: tuple[int, int], card_rects: list[pygame.Rect]) -> bool:
        for idx, rect in enumerate(card_rects):
            if rect.collidepoint(mouse_pos):
                self.selected_index = idx
                return True
        return False
