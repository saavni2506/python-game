from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygame

from config.modes import ModeConfig

if TYPE_CHECKING:
    from core.player import Player


WIDTH = 1280
HEIGHT = 720
FPS = 30
HORIZON_Y = int(HEIGHT * 0.27)
GROUND_Y = HEIGHT - 68

CAMERA_HEIGHT = 1.45
PROJECTION_SCALE = 760.0
PLAYER_Z = 3.2
ROAD_HALF_WIDTH = 2.8
LANE_X = [-1.2, 0.0, 1.2]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def lane_x(lane: int) -> float:
    return LANE_X[int(clamp(lane, 0, 2))]


def project_world(x: float, y: float, z: float) -> tuple[float, float, float] | None:
    if z <= 0.12:
        return None
    scale = PROJECTION_SCALE / z
    screen_x = (WIDTH * 0.5) + (x * scale)
    screen_y = HORIZON_Y + ((CAMERA_HEIGHT - y) * scale)
    return screen_x, screen_y, scale


@dataclass
class Obstacle:
    lane: int
    kind: str
    z: float
    prev_z: float = 0.0

    def __post_init__(self) -> None:
        self.prev_z = self.z

    def advance(self, distance: float) -> None:
        self.prev_z = self.z
        self.z -= distance

    def draw(self, screen: pygame.Surface) -> None:
        x_world = lane_x(self.lane)
        proj_ground = project_world(x_world, 0.0, self.z)
        if proj_ground is None:
            return
        sx, sy, scale = proj_ground

        if self.kind == "jump":
            width = max(24, int(scale * 1.12))
            height = max(18, int(scale * 0.62))
            rect = pygame.Rect(int(sx - (width * 0.5)), int(sy - height), width, height)

            glow = pygame.Surface((rect.width + 34, rect.height + 34), pygame.SRCALPHA)
            pygame.draw.rect(glow, (255, 130, 92, 110), glow.get_rect(), border_radius=12, width=4)
            screen.blit(glow, (rect.x - 17, rect.y - 17))

            shadow = pygame.Rect(rect.x + 2, rect.bottom - max(6, height // 5), rect.width - 4, max(4, height // 6))
            pygame.draw.rect(screen, (150, 52, 34), shadow, border_radius=5)

            pygame.draw.rect(screen, (255, 132, 90), rect, border_radius=10)
            pygame.draw.rect(screen, (255, 238, 198), rect, 2, border_radius=10)
            pygame.draw.rect(
                screen,
                (232, 92, 58),
                pygame.Rect(rect.x + 4, rect.y + 4, rect.width - 8, max(5, height // 4)),
                border_radius=6,
            )
            self._draw_marker(screen, int(sx), rect.y - max(10, int(scale * 0.22)), (255, 164, 112), scale * 1.4)
        else:
            bar_y = project_world(x_world, 1.03, self.z)
            if bar_y is None:
                return
            _, bar_sy, _ = bar_y
            width = max(34, int(scale * 1.28))
            height = max(12, int(scale * 0.26))
            rect = pygame.Rect(int(sx - (width * 0.5)), int(bar_sy - (height * 0.5)), width, height)

            glow = pygame.Surface((rect.width + 36, rect.height + 36), pygame.SRCALPHA)
            pygame.draw.rect(glow, (108, 238, 255, 120), glow.get_rect(), border_radius=12, width=4)
            screen.blit(glow, (rect.x - 18, rect.y - 18))

            pygame.draw.rect(screen, (102, 208, 255), rect, border_radius=7)
            pygame.draw.rect(screen, (232, 248, 255), rect, 2, border_radius=7)
            highlight = pygame.Rect(rect.x + 4, rect.y + 3, rect.width - 8, max(3, height // 3))
            pygame.draw.rect(screen, (182, 238, 255), highlight, border_radius=4)

            # Side posts make overhead bars easier to read at a glance.
            post_h = max(14, int(scale * 0.52))
            post_w = max(4, int(scale * 0.08))
            left_post = pygame.Rect(rect.left + 2, rect.bottom - 2, post_w, post_h)
            right_post = pygame.Rect(rect.right - post_w - 2, rect.bottom - 2, post_w, post_h)
            pygame.draw.rect(screen, (84, 176, 244), left_post, border_radius=4)
            pygame.draw.rect(screen, (84, 176, 244), right_post, border_radius=4)
            pygame.draw.rect(screen, (206, 242, 255), left_post, 1, border_radius=4)
            pygame.draw.rect(screen, (206, 242, 255), right_post, 1, border_radius=4)
            self._draw_marker(screen, int(sx), rect.y - max(12, int(scale * 0.26)), (140, 244, 255), scale * 1.4)

    @staticmethod
    def _draw_marker(
        screen: pygame.Surface,
        x: int,
        y: int,
        color: tuple[int, int, int],
        scale: float,
    ) -> None:
        marker_size = max(5, int(scale * 0.12))
        glow = pygame.Surface((marker_size * 6, marker_size * 6), pygame.SRCALPHA)
        c = glow.get_width() // 2
        pygame.draw.circle(glow, (color[0], color[1], color[2], 110), (c, c), int(marker_size * 1.8))
        pygame.draw.circle(glow, (color[0], color[1], color[2], 58), (c, c), int(marker_size * 2.6))
        screen.blit(glow, (x - c, y - c))
        pygame.draw.circle(screen, color, (x, y), marker_size)


@dataclass
class Coin:
    lane: int
    z: float
    height: float = 0.75
    phase: float = 0.0

    def advance(self, distance: float) -> None:
        self.z -= distance

    def draw(self, screen: pygame.Surface) -> None:
        y = self.height + (math.sin(self.phase + (self.z * 0.35)) * 0.08)
        proj = project_world(lane_x(self.lane), y, self.z)
        if proj is None:
            return
        sx, sy, scale = proj
        radius = max(4, int(scale * 0.16))

        glow = pygame.Surface((radius * 5, radius * 5), pygame.SRCALPHA)
        center = glow.get_width() // 2
        pygame.draw.circle(glow, (255, 204, 58, 116), (center, center), int(radius * 1.6))
        pygame.draw.circle(glow, (255, 236, 133, 62), (center, center), int(radius * 2.0))
        screen.blit(glow, (int(sx - center), int(sy - center)))

        pygame.draw.circle(screen, (250, 198, 40), (int(sx), int(sy)), radius)
        pygame.draw.circle(screen, (255, 235, 128), (int(sx - (radius * 0.25)), int(sy - (radius * 0.25))), max(2, radius // 3))
        pygame.draw.circle(screen, (235, 151, 24), (int(sx), int(sy)), radius, max(1, radius // 6))


class Level:
    def __init__(self, mode_config: ModeConfig) -> None:
        self.mode_config = mode_config
        self.background = self._build_background()
        self.reset()

    def reset(self) -> None:
        self.obstacles: list[Obstacle] = []
        self.coins: list[Coin] = []
        self.elapsed = 0.0
        self.world_scroll = 0.0
        self.spawn_timer = self.mode_config.obstacle_spawn_rate
        self.coin_timer = 0.45

    def update(self, dt: float, speed: float) -> None:
        self.elapsed += dt
        self.world_scroll += speed * dt

        self.spawn_timer -= dt
        if self.spawn_timer <= 0.0:
            self._spawn_obstacle()
            jitter = random.uniform(-0.10, 0.18)
            min_gap = max(0.6, self.mode_config.obstacle_spawn_rate * 0.68)
            self.spawn_timer = max(min_gap, self.mode_config.obstacle_spawn_rate + jitter)

        self.coin_timer -= dt
        if self.coin_timer <= 0.0:
            self._spawn_coin()
            self.coin_timer = random.uniform(0.22, 0.48)

        distance = speed * dt
        for obstacle in self.obstacles:
            obstacle.advance(distance)
        for coin in self.coins:
            coin.advance(distance)

        self.obstacles = [obstacle for obstacle in self.obstacles if obstacle.z > 0.55]
        self.coins = [coin for coin in self.coins if coin.z > 0.55]

    def _spawn_obstacle(self) -> None:
        jump_bias = 0.65 if self.mode_config.gesture_profile == "kids" else 0.55
        kind = "jump" if random.random() < jump_bias else "duck"
        lane = random.randint(0, 2)
        # Spawn slightly farther so obstacle silhouettes are readable before reaction window.
        spawn_z = random.uniform(45.0, 62.0)
        self.obstacles.append(Obstacle(lane, kind, spawn_z))

    def _spawn_coin(self) -> None:
        spawn_z = random.uniform(33.0, 57.0)
        self.coins.append(
            Coin(
                random.randint(0, 2),
                spawn_z,
                phase=random.uniform(0.0, 6.0),
            )
        )

    def hits_obstacle(self, player: Player, obstacle: Obstacle) -> bool:
        z_min = min(obstacle.z, obstacle.prev_z)
        z_max = max(obstacle.z, obstacle.prev_z)
        z_pad = 0.55
        if z_max < PLAYER_Z - z_pad or z_min > PLAYER_Z + z_pad:
            return False
        if abs(player.x - lane_x(obstacle.lane)) > 0.62:
            return False
        if obstacle.kind == "jump":
            return player.y < 0.60
        return not player.ducking

    def takes_coin(self, player: Player, coin: Coin) -> bool:
        if abs(coin.z - PLAYER_Z) > 0.48:
            return False
        if abs(player.x - lane_x(coin.lane)) > 0.62:
            return False
        return True

    def check_collision(self, player: Player) -> bool:
        return any(self.hits_obstacle(player, obstacle) for obstacle in self.obstacles)

    def collect_coins(self, player: Player) -> int:
        collected = 0
        remaining: list[Coin] = []
        for coin in self.coins:
            if self.takes_coin(player, coin):
                collected += 1
            else:
                remaining.append(coin)
        self.coins = remaining
        return collected

    def next_prompt(self) -> str:
        ahead = [item for item in self.obstacles if item.z > PLAYER_Z]
        if not ahead:
            return "RUN"
        nearest = min(ahead, key=lambda item: item.z)
        return "JUMP" if nearest.kind == "jump" else "DUCK"

    def draw(self, screen: pygame.Surface) -> None:
        screen.blit(self.background, (0, 0))
        self.draw_road(screen)
        self.draw_scenery(screen)

        for coin in sorted(self.coins, key=lambda item: item.z, reverse=True):
            coin.draw(screen)
        for obstacle in sorted(self.obstacles, key=lambda item: item.z, reverse=True):
            obstacle.draw(screen)

    def _build_background(self) -> pygame.Surface:
        surface = pygame.Surface((WIDTH, HEIGHT))
        for y in range(HEIGHT):
            if y < HORIZON_Y:
                k = y / max(1, HORIZON_Y)
                color = (
                    int(8 + (22 * k)),
                    int(10 + (18 * k)),
                    int(30 + (56 * k)),
                )
            else:
                k = (y - HORIZON_Y) / max(1, HEIGHT - HORIZON_Y)
                color = (
                    int(14 + (6 * k)),
                    int(18 + (12 * k)),
                    int(38 + (10 * k)),
                )
            pygame.draw.line(surface, color, (0, y), (WIDTH, y))

        # Distant skyline blocks with subtle neon windows.
        for idx in range(28):
            block_w = 30 + ((idx * 17) % 76)
            x = (idx * 47) % WIDTH
            h = 62 + ((idx * 23) % 130)
            y = HORIZON_Y - h + 26
            body = pygame.Rect(x, y, block_w, h)
            pygame.draw.rect(surface, (12, 18, 34), body, border_radius=4)
            pygame.draw.rect(surface, (20, 34, 60), body, 1, border_radius=4)
            for row in range(3, h - 6, 12):
                for col in range(4, block_w - 6, 10):
                    if (row + col + idx) % 3 == 0:
                        continue
                    win_color = (78, 234, 255) if (row + idx) % 2 == 0 else (154, 104, 255)
                    pygame.draw.rect(surface, win_color, pygame.Rect(x + col, y + row, 4, 6), border_radius=1)

        star_layer = pygame.Surface((WIDTH, HORIZON_Y + 30), pygame.SRCALPHA)
        for i in range(120):
            sx = (i * 97) % WIDTH
            sy = (i * 53) % max(1, HORIZON_Y - 8)
            twinkle = 140 + ((i * 37) % 96)
            star_layer.set_at((sx, sy), (186, 218, 255, twinkle))
        surface.blit(star_layer, (0, 0))

        haze = pygame.Surface((WIDTH, 170), pygame.SRCALPHA)
        for i in range(170):
            alpha = max(0, 112 - (i // 2))
            pygame.draw.line(haze, (58, 88, 140, alpha), (0, i), (WIDTH, i))
        surface.blit(haze, (0, HORIZON_Y - 34))
        return surface

    def draw_road(self, screen: pygame.Surface) -> None:
        near_z = 2.35
        far_z = 74.0

        left_near = project_world(-ROAD_HALF_WIDTH, 0.0, near_z)
        right_near = project_world(ROAD_HALF_WIDTH, 0.0, near_z)
        left_far = project_world(-ROAD_HALF_WIDTH, 0.0, far_z)
        right_far = project_world(ROAD_HALF_WIDTH, 0.0, far_z)

        shoulder_width = ROAD_HALF_WIDTH + 1.45
        shoulder_left_near = project_world(-shoulder_width, 0.0, near_z)
        shoulder_right_near = project_world(shoulder_width, 0.0, near_z)
        shoulder_left_far = project_world(-shoulder_width, 0.0, far_z)
        shoulder_right_far = project_world(shoulder_width, 0.0, far_z)

        curb_drop = -0.16
        left_near_low = project_world(-ROAD_HALF_WIDTH, curb_drop, near_z)
        right_near_low = project_world(ROAD_HALF_WIDTH, curb_drop, near_z)
        left_far_low = project_world(-ROAD_HALF_WIDTH, curb_drop, far_z)
        right_far_low = project_world(ROAD_HALF_WIDTH, curb_drop, far_z)

        if (
            left_near is None
            or right_near is None
            or left_far is None
            or right_far is None
            or shoulder_left_near is None
            or shoulder_right_near is None
            or shoulder_left_far is None
            or shoulder_right_far is None
            or left_near_low is None
            or right_near_low is None
            or left_far_low is None
            or right_far_low is None
        ):
            return

        ln_x, ln_y, _ = left_near
        rn_x, rn_y, _ = right_near
        lf_x, lf_y, _ = left_far
        rf_x, rf_y, _ = right_far
        sln_x, sln_y, _ = shoulder_left_near
        srn_x, srn_y, _ = shoulder_right_near
        slf_x, slf_y, _ = shoulder_left_far
        srf_x, srf_y, _ = shoulder_right_far
        ln_low_x, ln_low_y, _ = left_near_low
        rn_low_x, rn_low_y, _ = right_near_low
        lf_low_x, lf_low_y, _ = left_far_low
        rf_low_x, rf_low_y, _ = right_far_low

        dirt_left = [
            (0, HEIGHT),
            (int(sln_x), int(sln_y)),
            (int(slf_x), int(slf_y)),
            (0, int(slf_y)),
        ]
        dirt_right = [
            (WIDTH, HEIGHT),
            (int(srn_x), int(srn_y)),
            (int(srf_x), int(srf_y)),
            (WIDTH, int(srf_y)),
        ]
        shoulder_left = [
            (int(sln_x), int(sln_y)),
            (int(ln_x), int(ln_y)),
            (int(lf_x), int(lf_y)),
            (int(slf_x), int(slf_y)),
        ]
        shoulder_right = [
            (int(rn_x), int(rn_y)),
            (int(srn_x), int(srn_y)),
            (int(srf_x), int(srf_y)),
            (int(rf_x), int(rf_y)),
        ]
        road_left_side = [
            (int(ln_x), int(ln_y)),
            (int(lf_x), int(lf_y)),
            (int(lf_low_x), int(lf_low_y)),
            (int(ln_low_x), int(ln_low_y)),
        ]
        road_right_side = [
            (int(rn_x), int(rn_y)),
            (int(rf_x), int(rf_y)),
            (int(rf_low_x), int(rf_low_y)),
            (int(rn_low_x), int(rn_low_y)),
        ]
        road_poly = [
            (int(ln_x), int(ln_y)),
            (int(rn_x), int(rn_y)),
            (int(rf_x), int(rf_y)),
            (int(lf_x), int(lf_y)),
        ]

        pygame.draw.polygon(screen, (16, 18, 30), dirt_left)
        pygame.draw.polygon(screen, (16, 18, 30), dirt_right)
        pygame.draw.polygon(screen, (18, 26, 42), shoulder_left)
        pygame.draw.polygon(screen, (18, 26, 42), shoulder_right)
        pygame.draw.polygon(screen, (20, 24, 36), road_left_side)
        pygame.draw.polygon(screen, (20, 24, 36), road_right_side)
        pygame.draw.polygon(screen, (27, 34, 56), road_poly)
        pygame.draw.polygon(screen, (80, 238, 255), road_poly, 1)

        pygame.draw.line(screen, (102, 250, 255), (int(ln_x), int(ln_y)), (int(lf_x), int(lf_y)), 3)
        pygame.draw.line(screen, (102, 250, 255), (int(rn_x), int(rn_y)), (int(rf_x), int(rf_y)), 3)

        center_strip_near_left = project_world(-0.25, 0.0, near_z)
        center_strip_near_right = project_world(0.25, 0.0, near_z)
        center_strip_far_left = project_world(-0.25, 0.0, far_z)
        center_strip_far_right = project_world(0.25, 0.0, far_z)
        if (
            center_strip_near_left is not None
            and center_strip_near_right is not None
            and center_strip_far_left is not None
            and center_strip_far_right is not None
        ):
            strip_poly = [
                (int(center_strip_near_left[0]), int(center_strip_near_left[1])),
                (int(center_strip_near_right[0]), int(center_strip_near_right[1])),
                (int(center_strip_far_right[0]), int(center_strip_far_right[1])),
                (int(center_strip_far_left[0]), int(center_strip_far_left[1])),
            ]
            pygame.draw.polygon(screen, (36, 48, 82), strip_poly)

        dash_spacing = 2.3
        dash_len = 1.1
        phase = self.world_scroll % dash_spacing
        for lane_divider in (-0.93, 0.93):
            z = near_z + phase
            while z < far_z:
                p0 = project_world(lane_divider, 0.0, z)
                p1 = project_world(lane_divider, 0.0, z + dash_len)
                if p0 is not None and p1 is not None:
                    x0, y0, s0 = p0
                    x1, y1, s1 = p1
                    width0 = max(2, int(s0 * 0.012))
                    width1 = max(1, int(s1 * 0.009))
                    poly = [
                        (int(x0 - width0), int(y0)),
                        (int(x0 + width0), int(y0)),
                        (int(x1 + width1), int(y1)),
                        (int(x1 - width1), int(y1)),
                    ]
                    pygame.draw.polygon(screen, (190, 228, 255), poly)
                z += dash_spacing

        rail_spacing = 3.2
        rail_phase = self.world_scroll % rail_spacing
        for side in (-1, 1):
            prev_top: tuple[float, float, float] | None = None
            prev_mid: tuple[float, float, float] | None = None
            z = near_z + rail_phase
            while z < far_z:
                x = side * (ROAD_HALF_WIDTH + 1.06)
                post_base = project_world(x, 0.0, z)
                post_mid = project_world(x, 0.42, z)
                post_top = project_world(x, 0.72, z)
                if post_base is not None and post_mid is not None and post_top is not None:
                    post_w = max(1, int(post_base[2] * 0.0048))
                    pygame.draw.line(screen, (118, 246, 255), (int(post_base[0]), int(post_base[1])), (int(post_top[0]), int(post_top[1])), post_w)
                    if prev_top is not None and prev_mid is not None:
                        rail_w = max(1, int(((post_top[2] + prev_top[2]) * 0.5) * 0.0036))
                        pygame.draw.line(screen, (176, 128, 255), (int(prev_top[0]), int(prev_top[1])), (int(post_top[0]), int(post_top[1])), rail_w)
                        pygame.draw.line(screen, (98, 228, 255), (int(prev_mid[0]), int(prev_mid[1])), (int(post_mid[0]), int(post_mid[1])), rail_w)
                    prev_top = post_top
                    prev_mid = post_mid
                z += rail_spacing

    def draw_scenery(self, screen: pygame.Surface) -> None:
        near_z = 5.0
        far_z = 70.0
        spacing = 3.8
        phase = self.world_scroll % spacing

        items: list[tuple[float, int, str]] = []
        z = near_z + phase
        idx = 0
        while z < far_z:
            items.append((z, idx, "building"))
            if idx % 2 == 0:
                items.append((z + 1.4, idx, "street_light"))
            z += spacing
            idx += 1

        items.sort(reverse=True, key=lambda item: item[0])
        for z, idx, kind in items:
            for side in (-1, 1):
                x = side * (ROAD_HALF_WIDTH + 1.3 + ((idx % 3) * 0.55))
                if kind == "building":
                    self._draw_building(screen, x, z, idx)
                else:
                    self._draw_street_light(screen, x, z, idx)

    def _draw_building(self, screen: pygame.Surface, x: float, z: float, idx: int) -> None:
        half_w = 0.42 + ((idx % 3) * 0.08)
        height = 1.85 + ((idx % 4) * 0.34)
        depth = 0.9 + ((idx % 2) * 0.2)

        flb = project_world(x - half_w, 0.0, z)
        frb = project_world(x + half_w, 0.0, z)
        flt = project_world(x - half_w, height, z)
        frt = project_world(x + half_w, height, z)
        blb = project_world(x - half_w, 0.0, z + depth)
        brb = project_world(x + half_w, 0.0, z + depth)
        blt = project_world(x - half_w, height, z + depth)
        brt = project_world(x + half_w, height, z + depth)

        if (
            flb is None
            or frb is None
            or flt is None
            or frt is None
            or blb is None
            or brb is None
            or blt is None
            or brt is None
        ):
            return

        def pt(vertex: tuple[float, float, float]) -> tuple[int, int]:
            return int(vertex[0]), int(vertex[1])

        front_face = [pt(flb), pt(frb), pt(frt), pt(flt)]
        roof_face = [pt(flt), pt(frt), pt(brt), pt(blt)]
        if x < 0:
            side_face = [pt(frb), pt(brb), pt(brt), pt(frt)]
        else:
            side_face = [pt(flb), pt(blb), pt(blt), pt(flt)]

        base_color = (
            24 + ((idx % 4) * 10),
            36 + ((idx % 3) * 12),
            62 + ((idx % 2) * 16),
        )
        side_color = (
            max(0, base_color[0] - 18),
            max(0, base_color[1] - 16),
            max(0, base_color[2] - 10),
        )
        roof_color = (
            min(255, base_color[0] + 22),
            min(255, base_color[1] + 24),
            min(255, base_color[2] + 34),
        )

        pygame.draw.polygon(screen, roof_color, roof_face)
        pygame.draw.polygon(screen, side_color, side_face)
        pygame.draw.polygon(screen, base_color, front_face)
        pygame.draw.polygon(screen, (110, 220, 255), front_face, 1)

        front_min_x = int(min(flb[0], frb[0], flt[0], frt[0]))
        front_max_x = int(max(flb[0], frb[0], flt[0], frt[0]))
        front_top = int(min(flt[1], frt[1]))
        front_bottom = int(max(flb[1], frb[1]))
        face_w = front_max_x - front_min_x
        face_h = front_bottom - front_top
        if face_w > 10 and face_h > 20:
            win_w = max(2, face_w // 6)
            win_h = max(3, face_h // 10)
            for r in range(1, 7):
                for c in range(1, 4):
                    wx = front_min_x + int((c / 4) * face_w) - (win_w // 2)
                    wy = front_top + int((r / 8) * face_h)
                    if wy + win_h >= front_bottom - 2:
                        continue
                    win_color = (126, 242, 255) if (r + c + idx) % 2 == 0 else (178, 118, 255)
                    pygame.draw.rect(screen, win_color, pygame.Rect(wx, wy, win_w, win_h), border_radius=2)

    def _draw_street_light(self, screen: pygame.Surface, x: float, z: float, idx: int) -> None:
        trunk_w = 0.08
        trunk_h = 1.18
        trunk_depth = 0.20

        tflb = project_world(x - trunk_w, 0.0, z)
        tfrb = project_world(x + trunk_w, 0.0, z)
        tflt = project_world(x - trunk_w, trunk_h, z)
        tfrt = project_world(x + trunk_w, trunk_h, z)
        tblb = project_world(x - trunk_w, 0.0, z + trunk_depth)
        tbrb = project_world(x + trunk_w, 0.0, z + trunk_depth)
        tblt = project_world(x - trunk_w, trunk_h, z + trunk_depth)
        tbrt = project_world(x + trunk_w, trunk_h, z + trunk_depth)

        if (
            tflb is None
            or tfrb is None
            or tflt is None
            or tfrt is None
            or tblb is None
            or tbrb is None
            or tblt is None
            or tbrt is None
        ):
            return

        def pt(vertex: tuple[float, float, float]) -> tuple[int, int]:
            return int(vertex[0]), int(vertex[1])

        trunk_front = [pt(tflb), pt(tfrb), pt(tfrt), pt(tflt)]
        if x < 0:
            trunk_side = [pt(tfrb), pt(tbrb), pt(tbrt), pt(tfrt)]
        else:
            trunk_side = [pt(tflb), pt(tblb), pt(tblt), pt(tflt)]
        pygame.draw.polygon(screen, (38, 46, 66), trunk_side)
        pygame.draw.polygon(screen, (56, 66, 94), trunk_front)

        lamp = project_world(x, 1.16, z + 0.06)
        if lamp is None:
            return

        lamp_color = (114, 244, 255) if idx % 2 == 0 else (188, 122, 255)
        glow_radius = max(5, int(lamp[2] * 0.16))
        glow = pygame.Surface((glow_radius * 6, glow_radius * 6), pygame.SRCALPHA)
        center = glow.get_width() // 2
        pygame.draw.circle(glow, (lamp_color[0], lamp_color[1], lamp_color[2], 90), (center, center), int(glow_radius * 1.9))
        pygame.draw.circle(glow, (lamp_color[0], lamp_color[1], lamp_color[2], 56), (center, center), int(glow_radius * 2.5))
        screen.blit(glow, (int(lamp[0] - center), int(lamp[1] - center)))
        pygame.draw.circle(screen, lamp_color, (int(lamp[0]), int(lamp[1])), glow_radius)
