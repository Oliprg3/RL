import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

class Car:
    def __init__(self, x=0, y=0, heading=0):
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0
        self.max_steering = np.radians(35)
        self.max_speed = 30.0
        self.accel_force = 12.0
        self.drag = 0.02
        self.length = 2.5
        self.width = 1.8
        self.slip = 0.0

    def update(self, throttle, steering, dt=0.05):
        throttle = np.clip(throttle, -1, 1)
        steering = np.clip(steering, -1, 1)
        steer_angle = steering * self.max_steering
        accel = throttle * self.accel_force
        self.speed += accel * dt
        self.speed *= (1 - self.drag * dt)
        self.speed = np.clip(self.speed, -self.max_speed*0.5, self.max_speed)
        beta = math.atan(0.5 * math.tan(steer_angle))
        slip_rate = (self.speed / self.length) * math.sin(beta)
        self.slip = np.clip(self.slip + slip_rate*dt, -0.3, 0.3)
        heading_rate = (self.speed / self.length) * math.cos(beta) * math.tan(steer_angle)
        self.heading = (self.heading + heading_rate*dt) % (2*math.pi)
        vx = self.speed * math.cos(self.heading + self.slip)
        vy = self.speed * math.sin(self.heading + self.slip)
        self.x += vx * dt
        self.y += vy * dt

class TrackEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode="rgb_array", track_type="circular", width=90):
        super().__init__()
        self.render_mode = render_mode
        self.screen_width = 1000
        self.screen_height = 700
        self.track_type = track_type
        self.track_width = width
        pygame.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("RL Racer - Training")
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = None
        self.num_rays = 15
        self.ray_angles = np.linspace(-80, 80, self.num_rays)
        self.max_ray_len = 1200.0
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        obs_dim = self.num_rays + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self._build_track()
        self.reset()

    def _build_track(self):
        self.track_mask = pygame.Surface((self.screen_width, self.screen_height))
        self.track_mask.fill((0,0,0))
        cx, cy = 500, 350
        r = 280
        pts = []
        n = 100
        if self.track_type == "circular":
            for i in range(n):
                theta = 2*math.pi*i/n
                pts.append((cx + r*math.cos(theta), cy + r*math.sin(theta)))
        elif self.track_type == "oval":
            a, b = r*1.2, r*0.7
            for i in range(n):
                theta = 2*math.pi*i/n
                pts.append((cx + a*math.cos(theta), cy + b*math.sin(theta)))
        elif self.track_type == "figure8":
            for i in range(n):
                theta = 2*math.pi*i/n
                R = r * math.cos(theta) / (1 + math.sin(theta)**2)
                pts.append((cx + R*math.cos(theta), cy + R*math.sin(theta)))
        else:
            raise ValueError("Unknown track type")
        self.center_line = np.array(pts)
        w = int(self.track_width)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i+1)%n]
            pygame.draw.line(self.track_mask, (255,255,255), p1, p2, w)
        for pt in pts:
            pygame.draw.circle(self.track_mask, (255,255,255), pt, w//2)
        bottom_y = cy + (r if self.track_type!="oval" else r*0.7)
        self.start_idx = np.argmin(np.abs(self.center_line[:,1] - bottom_y))

    def _point_on_track(self, x, y):
        if x<0 or x>=self.screen_width or y<0 or y>=self.screen_height:
            return False
        try:
            return self.track_mask.get_at((int(x), int(y)))[0] > 128
        except:
            return False

    def _is_on_track(self, x, y):
        if not self._point_on_track(x, y):
            return False
        for dx, dy in [(4,4), (-4,4), (4,-4), (-4,-4)]:
            if not self._point_on_track(x+dx, y+dy):
                return False
        return True

    def _cast_ray(self, angle):
        rad = self.car.heading + math.radians(angle)
        dx = math.cos(rad)
        dy = math.sin(rad)
        for d in range(0, int(self.max_ray_len), 2):
            tx = self.car.x + dx*d
            ty = self.car.y + dy*d
            if not self._point_on_track(tx, ty):
                return float(d)
        return self.max_ray_len

    def _get_obs(self):
        rays = [self._cast_ray(a)/self.max_ray_len for a in self.ray_angles]
        speed_norm = self.car.speed / 30.0
        heading_sin = math.sin(self.car.heading)
        heading_cos = math.cos(self.car.heading)
        progress = min(1.0, self.total_distance / self._track_length())
        return np.array(rays + [speed_norm, heading_sin, heading_cos, progress], dtype=np.float32)

    def _track_length(self):
        length = 0.0
        n = len(self.center_line)
        for i in range(n):
            p1 = self.center_line[i]
            p2 = self.center_line[(i+1)%n]
            length += math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        return length

    def _distance_along(self, x, y):
        n = len(self.center_line)
        idx = min(range(n), key=lambda i: math.hypot(x-self.center_line[i][0], y-self.center_line[i][1]))
        cum = 0.0
        for i in range(idx):
            p1 = self.center_line[i]
            p2 = self.center_line[(i+1)%n]
            cum += math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        p = self.center_line[idx]
        cum += math.hypot(x-p[0], y-p[1])
        return cum

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.start_idx
        start = self.center_line[idx]
        next_idx = (idx+1)%len(self.center_line)
        heading = math.atan2(self.center_line[next_idx][1]-start[1],
                             self.center_line[next_idx][0]-start[0])
        self.car = Car(start[0], start[1], heading)
        self.car.speed = 0.5
        self.total_distance = 0.0
        self.episode_reward = 0.0
        self.lap_count = 0
        self.stuck_counter = 0
        self.steps = 0
        self.off_track_counter = 0
        return self._get_obs(), {}

    def step(self, action):
        throttle = np.clip(action[0], -1, 1)
        steering = np.clip(action[1], -1, 1)
        self.car.update(throttle, steering)
        self.steps += 1
        reward = 0.0
        done = False
        truncated = False

        on = self._is_on_track(self.car.x, self.car.y)
        if on:
            reward += 10.0
            self.off_track_counter = 0
        else:
            reward -= 200.0
            self.off_track_counter += 1
            if self.off_track_counter >= 5:
                done = True
                reward -= 50.0

        dist = self._distance_along(self.car.x, self.car.y)
        forward = dist - self.total_distance
        if forward < -self._track_length()*0.5:
            forward += self._track_length()
        self.total_distance += forward

        reward += 0.5 * max(0, forward)
        reward += 0.1 * abs(self.car.speed)

        centre_dist = min(math.hypot(self.car.x-p[0], self.car.y-p[1]) for p in self.center_line)
        reward -= 0.02 * centre_dist

        n = len(self.center_line)
        idx = min(range(n), key=lambda i: math.hypot(self.car.x-self.center_line[i][0], self.car.y-self.center_line[i][1]))
        nxt = (idx+1)%n
        target_heading = math.atan2(self.center_line[nxt][1]-self.center_line[idx][1],
                                    self.center_line[nxt][0]-self.center_line[idx][0])
        error = target_heading - self.car.heading
        error = (error + math.pi) % (2*math.pi) - math.pi
        reward -= 0.02 * abs(error)

        if abs(forward) < 0.1:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 300:
            truncated = True
            reward -= 30.0

        if self.steps >= 5000:
            truncated = True

        if self.total_distance >= self._track_length():
            self.lap_count += 1
            reward += 400.0
            done = True

        self.episode_reward += reward
        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        if self.render_mode is None:
            return
        self.screen.fill((30,30,30))
        self.screen.blit(self.track_mask, (0,0))
        for i, pt in enumerate(self.center_line):
            col = (0,200,0) if i%10==0 else (50,50,50)
            pygame.draw.circle(self.screen, col, (int(pt[0]), int(pt[1])), 3)
        for angle in self.ray_angles:
            d = self._cast_ray(angle)
            rad = self.car.heading + math.radians(angle)
            ex = self.car.x + math.cos(rad)*d
            ey = self.car.y + math.sin(rad)*d
            pygame.draw.line(self.screen, (0,200,255), (int(self.car.x), int(self.car.y)), (int(ex), int(ey)), 1)
            pygame.draw.circle(self.screen, (255,0,0), (int(ex), int(ey)), 3)
        cx, cy = int(self.car.x), int(self.car.y)
        car_body = pygame.Surface((int(self.car.width*8), int(self.car.length*8)), pygame.SRCALPHA)
        car_body.fill((0,0,0,0))
        pygame.draw.rect(car_body, (255,50,50), (0,0,*car_body.get_size()), border_radius=6)
        pygame.draw.rect(car_body, (100,100,200,180), (0,0,car_body.get_width(), car_body.get_height()//3))
        rotated = pygame.transform.rotate(car_body, -math.degrees(self.car.heading))
        rect = rotated.get_rect(center=(cx,cy))
        self.screen.blit(rotated, rect.topleft)
        fx, fy = self.car.x + 25*math.cos(self.car.heading), self.car.y + 25*math.sin(self.car.heading)
        pygame.draw.line(self.screen, (255,255,0), (cx, cy), (int(fx), int(fy)), 3)
        for w in [(self.car.x + self.car.width*3*math.cos(self.car.heading+math.pi/2) + self.car.length*3*math.cos(self.car.heading),
                   self.car.y + self.car.width*3*math.sin(self.car.heading+math.pi/2) + self.car.length*3*math.sin(self.car.heading)),
                  (self.car.x - self.car.width*3*math.cos(self.car.heading+math.pi/2) + self.car.length*3*math.cos(self.car.heading),
                   self.car.y - self.car.width*3*math.sin(self.car.heading+math.pi/2) + self.car.length*3*math.sin(self.car.heading)),
                  (self.car.x + self.car.width*3*math.cos(self.car.heading+math.pi/2) - self.car.length*3*math.cos(self.car.heading),
                   self.car.y + self.car.width*3*math.sin(self.car.heading+math.pi/2) - self.car.length*3*math.sin(self.car.heading)),
                  (self.car.x - self.car.width*3*math.cos(self.car.heading+math.pi/2) - self.car.length*3*math.cos(self.car.heading),
                   self.car.y - self.car.width*3*math.sin(self.car.heading+math.pi/2) - self.car.length*3*math.sin(self.car.heading))]:
            pygame.draw.circle(self.screen, (50,50,50), (int(w[0]), int(w[1])), 4)
        font = pygame.font.Font(None, 28)
        texts = [f"Speed: {abs(self.car.speed):.1f}", f"Lap: {self.lap_count}", f"Dist: {self.total_distance:.0f}", f"Reward: {self.episode_reward:.1f}"]
        for i, txt in enumerate(texts):
            surf = font.render(txt, True, (255,255,255))
            self.screen.blit(surf, (10, 10+i*32))
        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(60)

    def close(self):
        pygame.quit()
