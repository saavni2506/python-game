from __future__ import annotations

from dataclasses import dataclass

import pygame

from controllers.base_controller import MovementState
from core.player import Player


@dataclass(slots=True)
class PlayerInputResult:
    lane_changed: bool = False
    jumped: bool = False
    duck_hold: bool = False


class PlayerController:
    """Merges keyboard and gesture input into one player command stream."""

    def apply_input(
        self,
        player: Player,
        controls: MovementState,
        keys: pygame.key.ScancodeWrapper,
    ) -> PlayerInputResult:
        previous_lane = player.target_lane

        if keys[pygame.K_LEFT]:
            player.set_lane(0)
        elif keys[pygame.K_RIGHT]:
            player.set_lane(2)
        elif controls.tracked:
            player.set_lane(controls.lane)

        lane_changed = player.target_lane != previous_lane

        jump_requested = bool(controls.jump or keys[pygame.K_UP])
        jumped = False
        if jump_requested and player.on_ground() and not player.ducking:
            player.jump()
            jumped = True

        duck_hold = bool(controls.duck or keys[pygame.K_DOWN])
        return PlayerInputResult(lane_changed=lane_changed, jumped=jumped, duck_hold=duck_hold)
