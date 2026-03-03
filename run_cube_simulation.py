import pygame  # Main framework for windowing, input, and 2D rendering
from pygame.locals import *  # Imports Pygame constants like DOUBLEBUF and OPENGL
from OpenGL.GL import *  # Standard OpenGL functions (drawing primitives, colors)
from OpenGL.GLU import *  # OpenGL Utility library (camera and perspective setup)
import numpy as np  # Numerical library for vector math and array operations
import random  # Python's random number generator for initial states

# --- Configuration & Constants ---
CUBE_SIZE = 10  # Width, height, and depth of the 3D bounding box
GRID_RES = 5  # Number of voxels per axis (5x5x5 = 125 total cells)
PARTICLE_RADIUS = 0.15  # Visual and physical size of each particle
MAX_PARTICLES = 150  # Absolute maximum particles initialized in memory


class Particle:
    """Represents a single point-mass within the 3D simulation."""

    def __init__(self):
        self.reset()  # Initialize state immediately upon creation

    def reset(self):
        """Initializes particle at a random position and velocity within the cube."""
        # Create a random (x, y, z) position within the cube, offset to avoid starting inside walls
        self.pos = np.array([random.uniform(-CUBE_SIZE / 2 + 0.5, CUBE_SIZE / 2 - 0.5) for _ in range(3)])
        # Create a random (vx, vy, vz) velocity vector for movement
        self.vel = np.array([random.uniform(-0.1, 0.1) for _ in range(3)])
        self.mass = 1.0  # Assigned mass used for elastic collision math

    def update(self):
        """Updates position based on velocity and handles elastic wall collisions."""
        self.pos += self.vel  # Basic Euler integration: move position by velocity

        # Iterate through x, y, and z dimensions to check for wall hits
        for i in range(3):
            # If the particle edge (position + radius) exceeds the cube boundary...
            if abs(self.pos[i]) + PARTICLE_RADIUS > CUBE_SIZE / 2:
                self.vel[i] *= -1  # Flip the velocity component for a bounce
                # Snap the particle to be exactly touching the wall to prevent clipping
                self.pos[i] = np.sign(self.pos[i]) * (CUBE_SIZE / 2 - PARTICLE_RADIUS)


def handle_particle_collisions(particles):
    """
    Performs O(N^2) collision detection and response between all active particles.
    Uses an elastic collision model based on conservation of momentum.
    """
    # Double loop to compare every particle with every other particle exactly once
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            p1, p2 = particles[i], particles[j]  # Grab the two particles to compare
            dist_vec = p1.pos - p2.pos  # Vector pointing from p2 to p1
            distance = np.linalg.norm(dist_vec)  # Calculate the scalar distance (magnitude)

            # Collision occurs if distance is less than the sum of the radii
            if distance < (PARTICLE_RADIUS * 2):
                # Calculate how much the particles are overlapping
                overlap = (PARTICLE_RADIUS * 2) - distance
                res_vec = dist_vec / distance  # Normalized direction vector of collision

                # Forcibly move particles apart so they aren't stuck together
                p1.pos += res_vec * (overlap / 2)
                p2.pos -= res_vec * (overlap / 2)

                # Physics: Calculate velocity change along the collision normal
                normal = res_vec
                relative_vel = p1.vel - p2.vel
                vel_along_normal = np.dot(relative_vel, normal)  # Scalar projection

                # Only resolve impulse if particles are actually moving toward each other
                if vel_along_normal < 0:
                    # Impulse formula for a perfectly elastic collision (conservation of momentum)
                    impulse = (2 * vel_along_normal) / (p1.mass + p2.mass)
                    # Apply impulse to velocities based on their respective masses
                    p1.vel -= impulse * p2.mass * normal
                    p2.vel += impulse * p1.mass * normal


def calculate_shannon_entropy(particles):
    """
    Calculates the Shannon Entropy of the particle distribution across a 3D grid.
    Formula: $H = -\sum p_i \log_2(p_i)$
    """
    # Create an empty 3D array to count how many particles are in each grid cell
    counts = np.zeros((GRID_RES, GRID_RES, GRID_RES))
    step = CUBE_SIZE / GRID_RES  # Width of a single grid cell

    # Map each particle's continuous 3D coordinate to a discrete grid index
    for p in particles:
        ix = int((p.pos[0] + CUBE_SIZE / 2) / step)
        iy = int((p.pos[1] + CUBE_SIZE / 2) / step)
        iz = int((p.pos[2] + CUBE_SIZE / 2) / step)
        # Constrain indices within [0, GRID_RES-1] to prevent array index errors
        ix, iy, iz = map(lambda x: min(max(x, 0), GRID_RES - 1), (ix, iy, iz))
        counts[ix, iy, iz] += 1  # Increment count for this cell

    total = len(particles)  # Total number of active particles
    if total == 0:
        return counts, 0  # Return zero if there are no particles

    # Convert counts to probabilities (what percentage of total particles are here?)
    probs = counts / total
    entropy_map = np.zeros_like(probs)  # Map to store individual cell entropy values
    mask = probs > 0  # We can only calculate log for non-empty cells
    # The Shannon Entropy formula: p * log2(p)
    entropy_map[mask] = -probs[mask] * np.log2(probs[mask])

    return entropy_map, np.sum(entropy_map)  # Return the full map and the total sum


def draw_sphere(pos, radius):
    """Renders a sphere at a given position using OpenGL Quadrics."""
    glPushMatrix()  # Save current transformation matrix
    glTranslatef(*pos)  # Move to the particle's position
    quad = gluNewQuadric()  # Create a new quadric object (needed for spheres)
    gluSphere(quad, radius, 10, 10)  # Draw sphere with specified resolution
    glPopMatrix()  # Restore previous transformation matrix


def draw_grid_cell(x, y, z, entropy_val, max_h):
    """Renders a semi-transparent 'voxel' representing a cell in the entropy grid."""
    step = CUBE_SIZE / GRID_RES
    # Scale the color based on entropy; higher entropy = more red, lower = more blue
    norm_h = min(entropy_val * 10, 1.0)

    # RGBA: Red, Green, Blue, Alpha (Transparency)
    glColor4f(norm_h, 0.2, 1.0 - norm_h, 0.15)

    glPushMatrix()
    # Position the voxel in 3D space based on its grid indices
    glTranslatef(
        -CUBE_SIZE / 2 + x * step + step / 2,
        -CUBE_SIZE / 2 + y * step + step / 2,
        -CUBE_SIZE / 2 + z * step + step / 2,
    )
    s = step / 2 * 0.95  # The half-width of the voxel, slightly shrunk for visual gaps

    # Draw the 6 faces of the cube (GL_QUADS expects 4 vertices per face)
    glBegin(GL_QUADS)
    for v in [(s, s, -s), (-s, s, -s), (-s, s, s), (s, s, s),  # Top face
              (s, -s, s), (-s, -s, s), (-s, -s, -s), (s, -s, -s)]:  # Bottom face (partial list)
        glVertex3fv(v)
    glEnd()
    glPopMatrix()


# --- Slider UI (rendered via pygame 2D overlay on OpenGL) ---
class Slider:
    """A standard UI slider built with Pygame Rects for interaction."""

    def __init__(self, x, y, w, h, min_val, max_val, initial):
        self.rect = pygame.Rect(x, y, w, h)  # Boundary of the slider track
        self.min_val = min_val  # Lowest possible value
        self.max_val = max_val  # Highest possible value
        self.value = initial  # Current value
        self.dragging = False  # State to track if mouse is held down
        self.handle_radius = h + 4  # Visual size of the circular handle

    @property
    def handle_x(self):
        """Calculates the horizontal pixel position of the slider handle."""
        # Calculate percentage of completion
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        # Convert percentage to screen pixel coordinate
        return int(self.rect.x + ratio * self.rect.w)

    @property
    def handle_rect(self):
        """Returns the collision area for the slider handle."""
        cx = self.handle_x
        cy = self.rect.centery
        r = self.handle_radius
        # Returns a pygame Rect centered on the handle for collision detection
        return pygame.Rect(cx - r, cy - r, r * 2, r * 2)

    def handle_event(self, event):
        """Processes mouse clicks and drags to update slider value."""
        # Check if left mouse button is pressed over the handle
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
        # Release the drag state
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        # If moving mouse while dragging, update the value
        if event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = event.pos[0] - self.rect.x  # Get mouse position relative to slider start
            ratio = max(0.0, min(1.0, rel_x / self.rect.w))  # Clamp ratio between 0 and 1
            # Interpolate between min and max value
            self.value = int(self.min_val + ratio * (self.max_val - self.min_val))

    def draw(self, surface, font):
        """Renders the slider components onto the Pygame surface."""
        # Draw the dark gray track
        pygame.draw.rect(surface, (60, 60, 60), self.rect, border_radius=4)
        # Draw the blue "filled" portion of the track
        filled_w = self.handle_x - self.rect.x
        if filled_w > 0:
            filled_rect = pygame.Rect(self.rect.x, self.rect.y, filled_w, self.rect.h)
            pygame.draw.rect(surface, (80, 140, 220), filled_rect, border_radius=4)
        # Draw the circular handle (knob)
        pygame.draw.circle(surface, (200, 220, 255), (self.handle_x, self.rect.centery), self.handle_radius)
        # Draw a white outline on the handle
        pygame.draw.circle(surface, (255, 255, 255), (self.handle_x, self.rect.centery), self.handle_radius, 2)
        # Render and draw the text label showing current particle count
        label = font.render(f"Particles: {self.value}", True, (255, 255, 255))
        surface.blit(label, (self.rect.x, self.rect.y - 22))


def draw_overlay(screen_size, overlay_surface, slider, font, total_h):
    """Bridge between 2D Pygame UI and 3D OpenGL using texture mapping."""
    overlay_surface.fill((0, 0, 0, 0))  # Clear surface with full transparency

    # Render 2D components to the Pygame surface first
    slider.draw(overlay_surface, font)
    entropy_text = font.render(f"Shannon Entropy: {total_h:.3f}", True, (200, 255, 200))
    overlay_surface.blit(entropy_text, (10, screen_size[1] - 30))

    # Convert the Pygame surface data into a format OpenGL can understand (binary string)
    tex_data = pygame.image.tostring(overlay_surface, "RGBA", True)
    tex_id = glGenTextures(1)  # Generate a unique texture ID
    glBindTexture(GL_TEXTURE_2D, tex_id)  # Set this as the active texture
    # Upload the surface data to the GPU as a 2D texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 screen_size[0], screen_size[1], 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
    # Set filtering so the UI doesn't look blurry when scaled
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Temporary switch to Orthographic projection (2D mode) to draw the UI
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()  # Save 3D perspective matrix
    glLoadIdentity()
    glOrtho(0, screen_size[0], 0, screen_size[1], -1, 1)  # Set 2D coordinate system
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_TEXTURE_2D)  # Enable texture mapping
    glDisable(GL_DEPTH_TEST)  # Disable depth so UI draws on top of everything
    glColor4f(1, 1, 1, 1)  # Reset color to white so texture isn't tinted

    # Draw a single rectangle (quad) that covers the whole screen with the UI texture
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0);
    glVertex2f(0, 0)  # Bottom-left
    glTexCoord2f(1, 0);
    glVertex2f(screen_size[0], 0)  # Bottom-right
    glTexCoord2f(1, 1);
    glVertex2f(screen_size[0], screen_size[1])  # Top-right
    glTexCoord2f(0, 1);
    glVertex2f(0, screen_size[1])  # Top-left
    glEnd()

    glDisable(GL_TEXTURE_2D)  # Cleanup: Disable textures
    glEnable(GL_DEPTH_TEST)  # Re-enable depth for 3D objects in the next frame

    # Restore the original 3D perspective matrices
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    glDeleteTextures([tex_id])  # Delete texture from GPU memory to prevent leaks


def main():
    """Main simulation loop."""
    pygame.init()  # Initialize all imported pygame modules
    display = (800, 600)  # Set window resolution
    # Create window with Double Buffering and OpenGL support
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Particle Entropy Sim")

    # Initialize 3D camera: 45 degree FOV, Aspect Ratio, Near plane, Far plane
    gluPerspective(45, display[0] / display[1], 0.1, 70.0)
    glTranslatef(0.0, 0.0, -20)  # Move the camera back 20 units to see the cube
    glEnable(GL_DEPTH_TEST)  # Enable Z-buffer (so closer things block farther things)
    glEnable(GL_BLEND)  # Enable alpha transparency blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Set standard transparency math

    # Simulation Setup
    particles = [Particle() for _ in range(MAX_PARTICLES)]  # Pre-create all particles
    clock = pygame.time.Clock()  # Object to control frame rate
    font = pygame.font.SysFont("Arial", 16)  # Font for 2D text
    # Initialize slider UI at top-left
    slider = Slider(x=20, y=40, w=220, h=8, min_val=1, max_val=MAX_PARTICLES, initial=50)
    # Transparent surface used to draw the UI before sending to OpenGL
    overlay = pygame.Surface(display, pygame.SRCALPHA)

    angle_x, angle_y = 0, 0  # Variables to track cube rotation
    total_h = 0.0  # Current entropy value

    while True:  # Game Loop
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()  # Clean up Pygame
                return  # Exit the script
            slider.handle_event(event)  # Pass mouse/keyboard events to the slider

        # 2. Camera Rotation Control
        # If left mouse is pressed and we aren't clicking the slider, rotate the view
        if pygame.mouse.get_pressed()[0] and not slider.dragging:
            rel_x, rel_y = pygame.mouse.get_rel()  # Get mouse movement since last frame
            angle_x += rel_y  # Rotate around X axis based on vertical mouse move
            angle_y += rel_x  # Rotate around Y axis based on horizontal mouse move
        else:
            pygame.mouse.get_rel()  # "Consume" the relative movement so it doesn't jump later

        # 3. Simulation Logic
        num_active = slider.value  # Get number of particles from slider
        active_list = particles[:num_active]  # Slice the list to only process active ones
        handle_particle_collisions(active_list)  # Resolve sphere-to-sphere physics
        # Update entropy grid based on current particle positions
        entropy_map, total_h = calculate_shannon_entropy(active_list)

        # 4. 3D Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the screen and depth buffer
        glPushMatrix()  # Save matrix before rotating the whole scene
        glRotatef(angle_x, 1, 0, 0)  # Apply user rotation (X-axis)
        glRotatef(angle_y, 0, 1, 0)  # Apply user rotation (Y-axis)

        # Render the entropy grid voxels
        for x in range(GRID_RES):
            for y in range(GRID_RES):
                for z in range(GRID_RES):
                    draw_grid_cell(x, y, z, entropy_map[x, y, z], 1.0)

        # Render the actual particles
        glColor3f(1, 1, 1)  # Set color to white for particles
        for p in active_list:
            p.update()  # Move particle and check wall collisions
            draw_sphere(p.pos, PARTICLE_RADIUS)  # Draw the sphere at its new position

        glPopMatrix()  # Restore matrix (stops rotations from accumulating)

        # 5. UI Rendering
        draw_overlay(display, overlay, slider, font, total_h)

        pygame.display.flip()  # Swap the back buffer to the front (show the frame)
        clock.tick(60)  # Limit to 60 Frames Per Second


if __name__ == "__main__":  # Python idiom to ensure code runs only when executed directly
    main()