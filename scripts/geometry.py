import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class GameGeometry:
    "geometry definition of the game arena, obstacle, and target"
    
    # Arena boundaries
    L: float = 5.0  # half-length (x from -L to L)
    H: float = 5.0  # half-height (y from -H to H)
    
    # Obstacle
    obstacle_radius: float = 1.0
    obstacle_center: tuple = (0.0, 0.0)
    
    # Target
    target_y: float = None  # Will be set to -H
    
    # Capture radius
    epsilon: float = 0.5
    
    def __post_init__(self):
        self.target_y = -self.H
        self.x_range = (-self.L, self.L)
        self.y_range = (-self.H, self.H)
    
    def in_arena(self, x, y):
        "Check if point (x,y) is within arena boundaries"
        return (-self.L <= x <= self.L) and (-self.H <= y <= self.H)
    
    def in_obstacle(self, x, y):
        "Check if point (x,y) is inside obstacle"
        dx = x - self.obstacle_center[0]
        dy = y - self.obstacle_center[1]
        return (dx**2 + dy**2) < self.obstacle_radius**2
    
    def is_valid_position(self, x, y):
        "Position is valid if in arena and not in obstacle"
        return self.in_arena(x, y) and not self.in_obstacle(x, y)
    
    def distance_to_target(self, x, y):
        "Vertical distance to target (attacker wants to minimize this)"
        return abs(y - self.target_y)
    
    def plot_arena(self, ax=None):
        "Plot the game arena with obstacle and target"
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw arena boundary
        ax.plot([-self.L, self.L, self.L, -self.L, -self.L],
                [-self.H, -self.H, self.H, self.H, -self.H], 'k-', linewidth=2)
        
        # Draw obstacle
        circle = plt.Circle(self.obstacle_center, self.obstacle_radius, 
                          color='gray', alpha=0.5, label='Obstacle')
        ax.add_patch(circle)
        
        # Highlight target (bottom boundary)
        ax.axhline(y=self.target_y, color='green', linewidth=3, 
                  label='Target', alpha=0.7)
        
        # Mark start positions (typical)
        ax.plot(0, self.H-1, 'ro', markersize=10, label='Attacker start')
        ax.plot(0, -self.H+1, 'bo', markersize=10, label='Defender start')
        
        ax.set_xlim(-self.L-0.5, self.L+0.5)
        ax.set_ylim(-self.H-0.5, self.H+0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Game Arena')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        return ax


# Test
if __name__ == "__main__":
    geo = GameGeometry()
    fig, ax = plt.subplots()
    geo.plot_arena(ax)
    plt.show()
    
    # Some positions
    print(f"(0.9,0.6) in obstacle? {geo.in_obstacle(0.9, 0.6)}")
    print(f"(2,2) valid? {geo.is_valid_position(2, 2)}")
    print(f"(0,1) in obstacle? {geo.in_obstacle(0, 1)}")
