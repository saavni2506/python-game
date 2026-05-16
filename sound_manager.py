from __future__ import annotations

from pathlib import Path

import pygame


class SoundManager:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = False
        self.sounds: dict[str, pygame.mixer.Sound] = {}

        if not enabled:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.enabled = True
        except pygame.error:
            self.enabled = False

    def load(self, sound_id: str, path: str | Path) -> None:
        if not self.enabled:
            return
        sound_path = Path(path)
        if not sound_path.exists():
            return
        try:
            self.sounds[sound_id] = pygame.mixer.Sound(str(sound_path))
        except pygame.error:
            pass

    def play(self, sound_id: str) -> None:
        if not self.enabled:
            return
        sound = self.sounds.get(sound_id)
        if sound is not None:
            sound.play()

    def stop_all(self) -> None:
        if self.enabled and pygame.mixer.get_init():
            pygame.mixer.stop()

