import matplotlib.pyplot as plt
import numpy as np
from geometry import GameGeometry


class Phase3Solver:
    "Solution for Phase 3 (after obstacle) - on the same side heading towards target"
    
    def __init__(self, geometry: GameGeometry, vA: float, vD: float):
        self.geo = geometry
        self.vA = vA
        self.vD = vD
    
    def time_to_target(self, xA, yA):
        "Time for attacker to reach target going straight down"
        distance = yA - self.geo.target_y  # yA > target_y
        target_time = distance / self.vA
        return target_time
    
    def check_capture_at_point(self, xA, yA, xD, yD, t):
        """
        Check if defender can capture attacker at time t
        Returns: (can_capture, capture_time, distance_at_capture)
        """
        # Attacker's position at time t
        xA_t = xA
        yA_t = yA - self.vA * t
        
        # Can defender reach within ε of this point by time t?
        # Defender moves in straight line toward this point
        dx = xA_t - xD
        dy = yA_t - yD
        dist_defender_needs_to_travel = np.sqrt(dx**2 + dy**2)
        dist_defender_can_travel = self.vD * t
        
        if dist_defender_can_travel >= dist_defender_needs_to_travel:
            # Defender can reach this point
            # But capture requires being within ε, not necessarily at exactly the point
            # If defender arrives early - wait at the point
            return True, t, 0.0  # At exactly the point, distance = 0 < ε
        else:
            # Defender can't reach the point, but might get close enough?
            # Defender moves as far as possible toward the point
            fraction = dist_defender_can_travel / dist_defender_needs_to_travel
            xD_reached = xD + dx * fraction
            yD_reached = yD + dy * fraction
            
            # Distance at time t
            distance = np.sqrt((xA_t - xD_reached)**2 + (yA_t - yD_reached)**2)
            
            if distance < self.geo.epsilon:
                return True, t, distance
            else:
                return False, None, distance
    
    def find_capture_time(self, xA, yA, xD, yD):
        """
        Find the earliest time when defender captures attacker
        Returns: (capture_time, capture_distance) or (None, None) if no capture
        """
        t_attacker = self.time_to_target(xA, yA)
        
        # Check multiple points along attacker's path
        # More points - more accurate but slower
        t_values = np.linspace(0, t_attacker, 100)
        
        for t in t_values:
            can_capture, capture_time, distance = self.check_capture_at_point(xA, yA, xD, yD, t)
            if can_capture:
                return t, distance
        
        return None, None
    
    def payoff(self, xA, yA, xD, yD):
        """
        Compute payoff J3(xA, xD) for Phase 3
        Positive = attacker wins, Negative = defender wins
        """
        # Ensure Phase 3 (both on same side, attacker below obstacle)
        if yA > -self.geo.obstacle_radius:
            raise ValueError("Attacker not yet in Phase 3")
        
        # Find if and when capture occurs
        capture_time, capture_distance = self.find_capture_time(xA, yA, xD, yD)
        t_attacker = self.time_to_target(xA, yA)
        
        if capture_time is not None and capture_time < t_attacker:
            # Defender captures attacker before target
            # Payoff = negative of how close attacker was to target when caught
            y_at_catch = yA - self.vA * capture_time
            distance_from_target = y_at_catch - self.geo.target_y
            return -distance_from_target
        else:
            # Attacker reaches target
            # Need to find distance between players at t = t_attacker
            
            # Attacker's position at target
            xA_target = xA
            yA_target = self.geo.target_y
            
            # Defender's position at t_attacker
            # Defender moves optimally toward attacker during this time
            dx = xA_target - xD
            dy = yA_target - yD
            distance_to_target = np.sqrt(dx**2 + dy**2)
            distance_defender_can_travel = self.vD * t_attacker
            
            if distance_defender_can_travel >= distance_to_target:
                # Defender can reach the target point exactly when attacker does
                # But they might not be exactly at the same point
                # Defender arrives at target point
                final_distance = 0.0  # They meet at target
            else:
                # Defender moves as far as possible toward target
                fraction = distance_defender_can_travel / distance_to_target
                xD_final = xD + dx * fraction
                yD_final = yD + dy * fraction
                
                # Distance at target time
                final_distance = np.sqrt((xA_target - xD_final)**2 + 
                                       (yA_target - yD_final)**2)
            
            return final_distance  # Positive = attacker wins
    
    def plot_payoff_slice(self, xA, yA):
        """
        Plot payoff function for a given attacker position
        Parameters:
            xA: attacker's x-coordinate
            yA: attacker's y-coordinate
        """
        # Create grid of defender positions
        xD_grid = np.linspace(-self.geo.L, self.geo.L, 30)
        yD_grid = np.linspace(-self.geo.H, self.geo.H, 30)
        
        XD, YD = np.meshgrid(xD_grid, yD_grid)
        J = np.zeros_like(XD)
        
        for i in range(len(xD_grid)):
            for j in range(len(yD_grid)):
                xD = xD_grid[i]
                yD = yD_grid[j]
                
                # Check if valid position
                if not self.geo.is_valid_position(xD, yD):
                    J[j, i] = np.nan
                    continue
                
                J[j, i] = self.payoff(xA, yA, xD, yD)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(XD, YD, J, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax, label='Payoff (+=Attacker, -=Defender)')
        
        # Mark obstacle
        circle = plt.Circle((0, 0), self.geo.obstacle_radius, color='gray', alpha=0.5)
        ax.add_patch(circle)
        
        # Mark attacker position
        ax.plot(xA, yA, 'r*', markersize=15, label=f'Attacker at ({xA}, {yA})')
        
        ax.set_xlabel('Defender x')
        ax.set_ylabel('Defender y')
        ax.set_title(f'Phase 3 Payoff Map\nAttacker fixed at ({xA}, {yA})')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_capture_region(self, xA, yA):
        """
        Plot the region where defender can capture attacker
        Parameters:
            xA: attacker's x-coordinate
            yA: attacker's y-coordinate
        """
        # Create grid of defender positions
        xD_grid = np.linspace(-self.geo.L, self.geo.L, 40)
        yD_grid = np.linspace(-self.geo.H, self.geo.H, 40)
        
        XD, YD = np.meshgrid(xD_grid, yD_grid)
        capture = np.zeros_like(XD, dtype=bool)
        
        for i in range(len(xD_grid)):
            for j in range(len(yD_grid)):
                xD = xD_grid[i]
                yD = yD_grid[j]
                
                if not self.geo.is_valid_position(xD, yD):
                    capture[j, i] = False
                    continue
                
                capture_time, _ = self.find_capture_time(xA, yA, xD, yD)
                capture[j, i] = capture_time is not None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(XD, YD, capture, levels=1, colors=['lightcoral', 'lightgreen'])
        ax.contour(XD, YD, capture, levels=[0.5], colors='black', linewidths=2)
        
        # Mark obstacle
        circle = plt.Circle((0, 0), self.geo.obstacle_radius, color='gray', alpha=0.5)
        ax.add_patch(circle)
        
        # Mark attacker
        ax.plot(xA, yA, 'r*', markersize=15, label=f'Attacker at ({xA}, {yA})')
        
        ax.set_xlabel('Defender x')
        ax.set_ylabel('Defender y')
        ax.set_title(f'Capture Region\nGreen=Can Capture, Red=Cannot\nAttacker at ({xA}, {yA})')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    

# Test
if __name__ == "__main__":
    
    geo = GameGeometry(epsilon=0.5)  # Make sure epsilon is set
    solver = Phase3Solver(geo, vA=1.0, vD=1.2)

    print("PHASE 3 SOLVER TEST")

    # Test case 1: Defender close and slightly above attacker
    print(f"Attacker at (2.0, -2.0), Defender at (0.5, -1.0)")
    val1 = solver.payoff(2.0, -2.0, 0.5, -1.0)
    t_cap1, dist1 = solver.find_capture_time(2.0, -2.0, 0.5, -1.0)
    if val1 < 0:
        print(f"DEFENDER WINS: Catches attacker {-val1:.2f} units from target")
        print(f"Capture at t={t_cap1:.2f}s, distance={dist1:.3f} (ε={geo.epsilon})")
    else:
        print(f"ATTACKER WINS: Reaches target with defender {val1:.2f} units away")
    
    print("Plot 1: Attacker at (2.0, -2.0) - Payoff Map")
    fig1, ax1 = solver.plot_payoff_slice(2.0, -2.0)
    plt.show()

    print("Plot 2: Attacker at (2.0, -2.0) - Capture Region")
    fig2, ax2 = solver.plot_capture_region(2.0, -2.0)
    plt.show()

    # Test case 2: Defender far to the right and above
    print(f"Attacker at (2.0, -2.0), Defender at (3.0, -1.5)")
    val2 = solver.payoff(2.0, -2.0, 3.0, -1.5)
    t_cap2, dist2 = solver.find_capture_time(2.0, -2.0, 3.0, -1.5)
    if val2 < 0:
        print(f"DEFENDER WINS: Catches attacker {-val2:.2f} units from target")
        if t_cap2:
            print(f"Capture at t={t_cap2:.2f}s, distance={dist2:.3f}")
    else:
        print(f"ATTACKER WINS: Reaches target with defender {val2:.2f} units away")

    # Test case 3: Defender far to the right and below attacker (closer to target)
    print(f"Attacker at (2.0, -2.0), Defender at (3.0, -3.0)")
    val3 = solver.payoff(2.0, -2.0, 3.0, -3.0)
    t_cap3, dist3 = solver.find_capture_time(2.0, -2.0, 3.0, -3.0)
    if val3 < 0:
        print(f"DEFENDER WINS: Catches attacker {-val3:.2f} units from target")
        if t_cap3:
            print(f"Capture at t={t_cap3:.2f}s, distance={dist3:.3f}")
    else:
        print(f"ATTACKER WINS: Reaches target with defender {val3:.2f} units away")

    # Test case 4: Defender on opposite side
    print(f"Attacker at (2.0, -2.0), Defender at (-2.0, -1.5)")
    val4 = solver.payoff(2.0, -2.0, -2.0, -1.5)
    t_cap4, dist4 = solver.find_capture_time(2.0, -2.0, -2.0, -1.5)
    if val4 < 0:
        print(f"DEFENDER WINS: Catches attacker {-val4:.2f} units from target")
        if t_cap4:
            print(f"Capture at t={t_cap4:.2f}s, distance={dist4:.3f}")
    else:
        print(f"ATTACKER WINS: Reaches target with defender {val4:.2f} units away")

    # Test case 5: Defender with same x, far above
    print(f"Attacker at (2.0, -3.0), Defender at (2.0, -1.0)")
    val5 = solver.payoff(2.0, -3.0, 2.0, -1.0)
    t_cap5, dist5 = solver.find_capture_time(2.0, -3.0, 2.0, -1.0)
    if val5 < 0:
        print(f"DEFENDER WINS: Catches attacker {-val5:.2f} units from target")
        if t_cap5:
            print(f"Capture at t={t_cap5:.2f}s, distance={dist5:.3f}")
    else:
        print(f"ATTACKER WINS: Reaches target with defender {val5:.2f} units away")
        