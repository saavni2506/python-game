from __future__ import annotations

import math

import pygame

from core.level import PLAYER_Z, clamp, lane_x, project_world


class Player:
    def __init__(self) -> None:
        self.target_lane = 1
        self.x = lane_x(self.target_lane)
        self.y = 0.0
        self.velocity_y = 0.0
        self.ducking = False
        self.run_phase = 0.0

    def set_lane(self, lane: int) -> None:
        self.target_lane = int(clamp(lane, 0, 2))

    def jump(self) -> None:
        if self.on_ground() and not self.ducking:
            self.velocity_y = 8.1

    def on_ground(self) -> bool:
        return self.y <= 0.001 and self.velocity_y <= 0.0

    def is_airborne(self) -> bool:
        return self.y > 0.08

    def state(self) -> str:
        if self.ducking:
            return "duck"
        if self.is_airborne():
            return "jump"
        return "run"

    def update(self, dt: float, duck_hold: bool, speed: float) -> None:
        target_x = lane_x(self.target_lane)
        self.x += (target_x - self.x) * min(1.0, dt * 10.0)

        self.velocity_y -= 21.5 * dt
        self.y += self.velocity_y * dt
        if self.y < 0.0:
            self.y = 0.0
            self.velocity_y = 0.0

        self.ducking = duck_hold and self.on_ground()
        if not self.is_airborne() and not self.ducking:
            self.run_phase += dt * (6.8 + (speed * 0.08))
        else:
            self.run_phase += dt * 2.0

    def _project_joint(
        self,
        x_offset: float,
        y_offset: float,
        z_offset: float = 0.0,
        model_scale: float = 1.0,
    ) -> tuple[float, float, float, float] | None:
        world_z = PLAYER_Z + (z_offset * model_scale)
        proj = project_world(
            self.x + (x_offset * model_scale),
            self.y + (y_offset * model_scale),
            world_z,
        )
        if proj is None:
            return None
        sx, sy, scale = proj
        return sx, sy, scale, world_z

    def _draw_segment(
        self,
        screen: pygame.Surface,
        start: tuple[float, float, float, float],
        end: tuple[float, float, float, float],
        color: tuple[int, int, int],
        thickness: float,
    ) -> None:
        avg_scale = (start[2] + end[2]) * 0.5
        width = max(2, int(avg_scale * thickness))
        start_pt = (int(start[0]), int(start[1]))
        end_pt = (int(end[0]), int(end[1]))
        pygame.draw.line(screen, color, start_pt, end_pt, width)

        rim = (
            min(255, color[0] + 32),
            min(255, color[1] + 32),
            min(255, color[2] + 32),
        )
        shift = max(1, width // 5)
        pygame.draw.line(
            screen,
            rim,
            (start_pt[0] - shift, start_pt[1] - shift),
            (end_pt[0] - shift, end_pt[1] - shift),
            max(1, width // 3),
        )

    def _draw_joint(
        self,
        screen: pygame.Surface,
        joint: tuple[float, float, float, float],
        color: tuple[int, int, int],
        radius_scale: float,
    ) -> None:
        radius = max(2, int(joint[2] * radius_scale))
        center = (int(joint[0]), int(joint[1]))
        pygame.draw.circle(screen, color, center, radius)
        highlight = (
            min(255, color[0] + 40),
            min(255, color[1] + 40),
            min(255, color[2] + 40),
        )
        pygame.draw.circle(screen, highlight, center, max(1, radius // 2))

    def _draw_limb(
        self,
        screen: pygame.Surface,
        upper: tuple[float, float, float, float],
        mid: tuple[float, float, float, float],
        lower: tuple[float, float, float, float],
        color: tuple[int, int, int],
        thickness: float,
    ) -> None:
        self._draw_segment(screen, upper, mid, color, thickness)
        self._draw_segment(screen, mid, lower, color, thickness)
        self._draw_joint(screen, mid, color, thickness * 0.48)
        self._draw_joint(screen, lower, color, thickness * 0.36)

    def draw(self, screen: pygame.Surface) -> None:
        current_state = self.state()
        model_scale = 0.84

        def joint(x_offset: float, y_offset: float, z_offset: float = 0.0) -> tuple[float, float, float, float] | None:
            return self._project_joint(x_offset, y_offset, z_offset, model_scale)

        if current_state == "duck":
            hip_y = 0.56
            shoulder_y = 0.96
            head_lift = 0.15
            stride_scale = 0.38
            arm_scale = 0.28
            body_tilt = 0.08
        elif current_state == "jump":
            hip_y = 0.84
            shoulder_y = 1.34
            head_lift = 0.18
            stride_scale = 0.34
            arm_scale = 0.72
            body_tilt = -0.04
        else:
            hip_y = 0.82
            shoulder_y = 1.36
            head_lift = 0.17
            stride_scale = 1.0
            arm_scale = 1.0
            body_tilt = 0.0

        gait_wave = math.sin(self.run_phase * 2.5)
        bounce = abs(gait_wave) * (0.032 if current_state == "run" else 0.012)
        twist = gait_wave * 0.06 * stride_scale

        root = joint(0.0, 0.0, 0.0)
        left_hip = joint(-0.14, hip_y - bounce, 0.02 + twist)
        right_hip = joint(0.14, hip_y - bounce, -0.02 - twist)
        left_shoulder = joint(-0.22, shoulder_y - bounce + body_tilt, -0.02 - twist)
        right_shoulder = joint(0.22, shoulder_y - bounce + body_tilt, 0.02 + twist)
        neck = joint(0.0, shoulder_y + 0.02 - bounce + body_tilt, -twist * 0.3)
        head = joint(0.0, shoulder_y + head_lift - bounce + body_tilt, -twist * 0.25)
        chest_front = joint(0.0, ((hip_y + shoulder_y) * 0.5) - bounce, 0.0)
        chest_back = joint(0.0, ((hip_y + shoulder_y) * 0.5) - bounce, 0.24)

        if (
            root is None
            or left_hip is None
            or right_hip is None
            or left_shoulder is None
            or right_shoulder is None
            or neck is None
            or head is None
            or chest_front is None
            or chest_back is None
        ):
            return

        left_leg_phase = gait_wave * stride_scale
        right_leg_phase = -gait_wave * stride_scale
        if current_state == "jump":
            left_leg_phase = math.sin(self.run_phase * 1.6) * 0.28
            right_leg_phase = -left_leg_phase

        def build_leg(side: float, phase: float) -> tuple[tuple[float, float, float, float] | None, tuple[float, float, float, float] | None]:
            foot_lift = 0.02 + (max(0.0, phase) * 0.24)
            if current_state == "jump":
                foot_lift += 0.12
            elif current_state == "duck":
                foot_lift = 0.01 + (max(0.0, phase) * 0.12)

            foot_x = (side * 0.16) + (phase * 0.13)
            foot_z = -phase * 0.18
            knee_x = (side * 0.14) + (phase * 0.07)
            knee_y = (0.37 + (foot_lift * 0.5)) + (0.06 if current_state == "duck" else 0.0)
            knee_z = foot_z * 0.58
            knee = joint(knee_x, knee_y - bounce, knee_z)
            foot = joint(foot_x, foot_lift, foot_z)
            return knee, foot

        left_knee, left_foot = build_leg(-1.0, left_leg_phase)
        right_knee, right_foot = build_leg(1.0, right_leg_phase)
        if left_knee is None or left_foot is None or right_knee is None or right_foot is None:
            return

        left_arm_phase = -left_leg_phase * arm_scale
        right_arm_phase = -right_leg_phase * arm_scale

        def build_arm(side: float, phase: float) -> tuple[tuple[float, float, float, float] | None, tuple[float, float, float, float] | None]:
            if current_state == "duck":
                elbow_x = (side * 0.25) + (phase * 0.03)
                hand_x = (side * 0.2) + (phase * 0.02)
                elbow_y = shoulder_y - 0.24 - bounce
                hand_y = shoulder_y - 0.39 - bounce
            else:
                elbow_x = (side * 0.26) + (phase * 0.08)
                hand_x = (side * 0.32) + (phase * 0.14)
                elbow_y = shoulder_y - 0.25 - bounce + (max(0.0, -phase) * 0.08)
                hand_y = shoulder_y - 0.58 - bounce + (max(0.0, -phase) * 0.18)
            elbow_z = -phase * 0.08
            hand_z = -phase * 0.14
            elbow = joint(elbow_x, elbow_y, elbow_z)
            hand = joint(hand_x, hand_y, hand_z)
            return elbow, hand

        left_elbow, left_hand = build_arm(-1.0, left_arm_phase)
        right_elbow, right_hand = build_arm(1.0, right_arm_phase)
        if left_elbow is None or left_hand is None or right_elbow is None or right_hand is None:
            return

        shadow_scale = max(0.36, 1.0 - (self.y * 0.34))
        shadow_w = max(24, int(root[2] * 0.38 * shadow_scale))
        shadow_h = max(8, int(root[2] * 0.11 * shadow_scale))
        shadow_rect = pygame.Rect(
            int(root[0] - (shadow_w * 0.5)),
            int(root[1] - (shadow_h * 0.45)),
            shadow_w,
            shadow_h,
        )
        pygame.draw.ellipse(screen, (25, 37, 61), shadow_rect)

        left_leg_front = left_foot[3] < right_foot[3]
        if left_leg_front:
            back_leg = (right_hip, right_knee, right_foot)
            front_leg = (left_hip, left_knee, left_foot)
            back_arm = (left_shoulder, left_elbow, left_hand)
            front_arm = (right_shoulder, right_elbow, right_hand)
            side_anchor_top = right_shoulder
            side_anchor_bottom = right_hip
        else:
            back_leg = (left_hip, left_knee, left_foot)
            front_leg = (right_hip, right_knee, right_foot)
            back_arm = (right_shoulder, right_elbow, right_hand)
            front_arm = (left_shoulder, left_elbow, left_hand)
            side_anchor_top = left_shoulder
            side_anchor_bottom = left_hip

        side_shift_x = chest_back[0] - chest_front[0]
        side_shift_y = chest_back[1] - chest_front[1]

        self._draw_limb(screen, back_leg[0], back_leg[1], back_leg[2], (46, 66, 125), 0.010)
        self._draw_limb(screen, back_arm[0], back_arm[1], back_arm[2], (206, 128, 94), 0.009)

        torso_front = [
            (int(left_hip[0]), int(left_hip[1])),
            (int(right_hip[0]), int(right_hip[1])),
            (int(right_shoulder[0]), int(right_shoulder[1])),
            (int(left_shoulder[0]), int(left_shoulder[1])),
        ]
        torso_side = [
            (int(side_anchor_bottom[0]), int(side_anchor_bottom[1])),
            (int(side_anchor_top[0]), int(side_anchor_top[1])),
            (int(side_anchor_top[0] + side_shift_x), int(side_anchor_top[1] + side_shift_y)),
            (int(side_anchor_bottom[0] + side_shift_x), int(side_anchor_bottom[1] + side_shift_y)),
        ]
        pygame.draw.polygon(screen, (166, 36, 44), torso_side)
        pygame.draw.polygon(screen, (226, 61, 69), torso_front)
        pygame.draw.polygon(screen, (255, 166, 130), torso_front, 2)

        self._draw_segment(screen, neck, head, (228, 170, 139), 0.007)
        head_radius = max(8, int(head[2] * 0.085))
        head_center = (int(head[0]), int(head[1]))
        pygame.draw.circle(screen, (250, 198, 161), head_center, head_radius)
        pygame.draw.circle(
            screen,
            (255, 224, 196),
            (int(head[0] - (head_radius * 0.28)), int(head[1] - (head_radius * 0.28))),
            max(2, head_radius // 3),
        )

        self._draw_limb(screen, front_arm[0], front_arm[1], front_arm[2], (245, 169, 132), 0.009)
        self._draw_limb(screen, front_leg[0], front_leg[1], front_leg[2], (74, 104, 182), 0.010)
        self._draw_joint(screen, left_hip, (96, 126, 204), 0.005)
        self._draw_joint(screen, right_hip, (96, 126, 204), 0.005)
