import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random

CUBE_SIZE = 10
GRID_RES = 5
PARTICLE_RADIUS = 0.15
MAX_PARTICLES = 150


class Particle:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([random.uniform(-CUBE_SIZE / 2 + 0.5, CUBE_SIZE / 2 - 0.5) for _ in range(3)])
        self.vel = np.array([random.uniform(-0.1, 0.1) for _ in range(3)])
        self.mass = 1.0

    def update(self):
        self.pos += self.vel
        for i in range(3):
            if abs(self.pos[i]) + PARTICLE_RADIUS > CUBE_SIZE / 2:
                self.vel[i] *= -1
                self.pos[i] = np.sign(self.pos[i]) * (CUBE_SIZE / 2 - PARTICLE_RADIUS)


def handle_particle_collisions(particles):
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            p1, p2 = particles[i], particles[j]
            dist_vec = p1.pos - p2.pos
            distance = np.linalg.norm(dist_vec)
            if distance < (PARTICLE_RADIUS * 2):
                overlap = (PARTICLE_RADIUS * 2) - distance
                res_vec = dist_vec / distance
                p1.pos += res_vec * (overlap / 2)
                p2.pos -= res_vec * (overlap / 2)
                normal = res_vec
                relative_vel = p1.vel - p2.vel
                vel_along_normal = np.dot(relative_vel, normal)
                if vel_along_normal < 0:
                    impulse = (2 * vel_along_normal) / (p1.mass + p2.mass)
                    p1.vel -= impulse * p2.mass * normal
                    p2.vel += impulse * p1.mass * normal


def calculate_shannon_entropy(particles):
    """
    Returns entropy_map, total Shannon entropy, and density_map (probability per cell).
    """
    counts = np.zeros((GRID_RES, GRID_RES, GRID_RES))
    step = CUBE_SIZE / GRID_RES
    for p in particles:
        ix = int((p.pos[0] + CUBE_SIZE / 2) / step)
        iy = int((p.pos[1] + CUBE_SIZE / 2) / step)
        iz = int((p.pos[2] + CUBE_SIZE / 2) / step)
        ix, iy, iz = map(lambda x: min(max(x, 0), GRID_RES - 1), (ix, iy, iz))
        counts[ix, iy, iz] += 1

    total = len(particles)
    if total == 0:
        return counts, 0, counts  # entropy_map, total_h, density_map

    probs = counts / total  # ← density: fraction of particles in each cell
    entropy_map = np.zeros_like(probs)
    mask = probs > 0
    entropy_map[mask] = -probs[mask] * np.log2(probs[mask])

    return entropy_map, np.sum(entropy_map), probs  # ← return probs


def draw_sphere(pos, radius):
    glPushMatrix()
    glTranslatef(*pos)
    quad = gluNewQuadric()
    gluSphere(quad, radius, 10, 10)
    glPopMatrix()


def draw_grid_cell(x, y, z, density, max_h):
    """
    Colors each voxel by local particle density (particles_in_cell / total_particles).
    Low density → blue, high density → red.
    """
    step = CUBE_SIZE / GRID_RES

    # Scale up so that uniform distribution (≈1/125) maps to a visible mid-tone.
    # Dividing GRID_RES³ by 5 gives a soft ceiling so even modest clusters look red.
    norm_d = min(density * (GRID_RES ** 3) / 5.0, 1.0)

    # RGBA: red channel rises with density, blue channel falls
    glColor4f(norm_d, 0.2, 1.0 - norm_d, 0.15)

    glPushMatrix()
    glTranslatef(
        -CUBE_SIZE / 2 + x * step + step / 2,
        -CUBE_SIZE / 2 + y * step + step / 2,
        -CUBE_SIZE / 2 + z * step + step / 2,
    )
    s = step / 2 * 0.95

    glBegin(GL_QUADS)
    for v in [(s, s, -s), (-s, s, -s), (-s, s, s), (s, s, s),
              (s, -s, s), (-s, -s, s), (-s, -s, -s), (s, -s, -s)]:
        glVertex3fv(v)
    glEnd()
    glPopMatrix()


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.dragging = False
        self.handle_radius = h + 4

    @property
    def handle_x(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return int(self.rect.x + ratio * self.rect.w)

    @property
    def handle_rect(self):
        cx = self.handle_x
        cy = self.rect.centery
        r = self.handle_radius
        return pygame.Rect(cx - r, cy - r, r * 2, r * 2)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x
            ratio = max(0.0, min(1.0, rel_x / self.rect.w))
            self.value = int(self.min_val + ratio * (self.max_val - self.min_val))

    def draw(self, surface, font):
        pygame.draw.rect(surface, (60, 60, 60), self.rect, border_radius=4)
        filled_w = self.handle_x - self.rect.x
        if filled_w > 0:
            filled_rect = pygame.Rect(self.rect.x, self.rect.y, filled_w, self.rect.h)
            pygame.draw.rect(surface, (80, 140, 220), filled_rect, border_radius=4)
        pygame.draw.circle(surface, (200, 220, 255), (self.handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(surface, (255, 255, 255), (self.handle_x, self.rect.centery), self.handle_radius, 2)
        label = font.render(f"Particles: {self.value}", True, (255, 255, 255))
        surface.blit(label, (self.rect.x, self.rect.y - 22))


def draw_overlay(screen_size, overlay_surface, slider, font, total_h):
    overlay_surface.fill((0, 0, 0, 0))
    slider.draw(overlay_surface, font)
    entropy_text = font.render(f"Shannon Entropy: {total_h:.3f}", True, (200, 255, 200))
    overlay_surface.blit(entropy_text, (10, screen_size[1] - 30))

    tex_data = pygame.image.tostring(overlay_surface, "RGBA", True)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 screen_size[0], screen_size[1], 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, screen_size[0], 0, screen_size[1], -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_TEXTURE_2D)
    glDisable(GL_DEPTH_TEST)
    glColor4f(1, 1, 1, 1)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(screen_size[0], 0)
    glTexCoord2f(1, 1); glVertex2f(screen_size[0], screen_size[1])
    glTexCoord2f(0, 1); glVertex2f(0, screen_size[1])
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glDeleteTextures([tex_id])


def main():
    pygame.init()
    display = (800, 600)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Particle Entropy Sim")

    gluPerspective(45, display[0] / display[1], 0.1, 70.0)
    glTranslatef(0.0, 0.0, -20)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    particles = [Particle() for _ in range(MAX_PARTICLES)]
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    slider = Slider(x=20, y=40, w=220, h=8, min_val=1, max_val=MAX_PARTICLES, initial=50)
    overlay = pygame.Surface(display, pygame.SRCALPHA)

    angle_x, angle_y = 0, 0
    total_h = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            slider.handle_event(event)

        if pygame.mouse.get_pressed()[0] and not slider.dragging:
            rel_x, rel_y = pygame.mouse.get_rel()
            angle_x += rel_y
            angle_y += rel_x
        else:
            pygame.mouse.get_rel()

        num_active = slider.value
        active_list = particles[:num_active]
        handle_particle_collisions(active_list)

        # ← unpack the new third return value
        entropy_map, total_h, density_map = calculate_shannon_entropy(active_list)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glRotatef(angle_x, 1, 0, 0)
        glRotatef(angle_y, 0, 1, 0)

        for x in range(GRID_RES):
            for y in range(GRID_RES):
                for z in range(GRID_RES):
                    # ← pass density instead of entropy
                    draw_grid_cell(x, y, z, density_map[x, y, z], 1.0)

        glColor3f(1, 1, 1)
        for p in active_list:
            p.update()
            draw_sphere(p.pos, PARTICLE_RADIUS)

        glPopMatrix()
        draw_overlay(display, overlay, slider, font, total_h)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()