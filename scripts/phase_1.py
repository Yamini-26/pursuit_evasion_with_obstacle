import numpy as np
import matplotlib.pyplot as plt
from geometry import GameGeometry
from phase_2 import Phase2Solver


class Phase1Solver:
    "Solution for Phase 1 (above obstacle) - Approach to obstacle"
    
    def __init__(self, geometry: GameGeometry, vA: float, vD: float, T_max: float):
        self.geo = geometry
        self.vA = vA
        self.vD = vD
        self.T_max = T_max
        self.phase2 = Phase2Solver(geometry, vA, vD, T_max)
        
        # Obstacle parameters
        self.obs_radius = geometry.obstacle_radius
        self.obs_top = self.obs_radius + 0.1  # y = +1.1
        self.obs_bottom = -self.obs_radius - 0.1  # y = -1.1
    
    def time_to_obstacle(self, y_pos, is_attacker=True):
        "Time to reach obstacle interface from current y"
        if is_attacker:
            # Attacker moves down to top of obstacle
            distance = y_pos - self.obs_top
            return distance / self.vA if distance > 0 else 0
        else:
            # Defender stays down below and move horizontally
            # For defender - "time to obstacle" means time to get into position and denpending on attacker's arrival
            return 0
    
    def attacker_reachable_x_at_obstacle(self, xA_start, yA_start):
        """
        What x-positions can attacker reach at obstacle top?
        They can move horizontally while descending
        """
        # Time to reach obstacle vertically
        t_descend = self.time_to_obstacle(yA_start, is_attacker=True)
        
        # During this time, they can move horizontally at speed vA
        max_horizontal_move = self.vA * t_descend
        
        # They can reach any x between:
        x_min = xA_start - max_horizontal_move
        x_max = xA_start + max_horizontal_move
        
        # But must stay in arena
        x_min = max(x_min, -self.geo.L)
        x_max = min(x_max, self.geo.L)
        
        return x_min, x_max, t_descend
    
    def value(self, xA, yA, xD, yD):
        """
        Value function for Phase 1
        Returns: (V, optimal_xA_interface, optimal_xD_interface)
        V = expected payoff with optimal play
        """
        # Step 1: Get attacker's reachable x at obstacle
        xA_min, xA_max, t_A_vertical = self.attacker_reachable_x_at_obstacle(xA, yA)
        
        # Step 2: Defender can go anywhere horizontally
        # But they have less time if attacker arrives quickly
        
        # Step 3: For each possible attacker x at interface, 
        # what's the best defender can do?
        
        best_value = -np.inf
        best_xA_interface = None
        best_xD_interface = None
        
        # Sample possible attacker interface positions
        xA_candidates = np.linspace(xA_min, xA_max, 10)
        
        for xA_interface in xA_candidates:
            # Attacker chooses this x at interface
            # How much horizontal movement needed?
            dx_A = abs(xA_interface - xA)
            t_A_horizontal = dx_A / self.vA
            
            # Total time for attacker to reach interface
            t_A_total = t_A_vertical + t_A_horizontal  # They move diagonally

            # If attacker can't even reach obstacle within T_max, it's a draw
            if t_A_total > self.T_max:
                # Game times out before Phase 2 even starts
                continue  # Skip this candidate - can't reach in time
            
            # Defender's problem: choose xD_interface to maximize (minimize?) payoff
            # Defender wants to minimize attacker's payoff
            
            # Defender can move horizontally while attacker descends
            # Defender has t_A_total seconds to move
            max_defender_move = self.vD * t_A_total
            
            # Defender can reach any x between:
            xD_min = xD - max_defender_move
            xD_max = xD + max_defender_move
            
            # Clip to arena
            xD_min = max(xD_min, -self.geo.L)
            xD_max = min(xD_max, self.geo.L)
            
            # Defender will choose xD_interface that minimizes Phase 2 value
            # Check a few candidates
            xD_candidates = np.linspace(xD_min, xD_max, 10)
            defender_best_value = np.inf
            best_xD_for_this_xA = None
            
            for xD_interface in xD_candidates:
                # Get Phase 2 value at these interface positions
                # We need to build a payoff matrix for these specific positions
                # For simplicity, use the equilibrium value from Phase 2
                
                # Build matrix for these specific interface positions
                matrix = self.phase2.build_payoff_matrix(
                    xA_start=(xA_interface, self.obs_top),
                    xD_start=(xD_interface, self.obs_bottom)
                )
                
                # Get the value of the game from this state
                # This is the Nash equilibrium value
                _, _, V_phase2 = self.phase2.solve_nash_equilibrium(matrix)
                
                # Defender wants to minimize V
                if V_phase2 < defender_best_value:
                    defender_best_value = V_phase2
                    best_xD_for_this_xA = xD_interface
            
            # Attacker wants to maximize the minimum value
            if defender_best_value > best_value:
                best_value = defender_best_value
                best_xA_interface = xA_interface
                best_xD_interface = best_xD_for_this_xA

        # If no candidate works (all exceed T_max), it's a draw
        if best_xA_interface is None:
            return 0.0, xA, xD  # Draw - can't reach obstacle in time
            
        return best_value, best_xA_interface, best_xD_interface
    
    def simulate_optimal_paths(self, xA, yA, xD, yD, dt=0.1):
        "Simulate the optimal paths from start to obstacle"
        # Get the target interface positions
        V, xA_target, xD_target = self.value(xA, yA, xD, yD)
        
        print(f"Optimal interface: Attacker at x={xA_target:.2f}, Defender at x={xD_target:.2f}")
        print(f"Expected game value: {V:.3f}")
        
        # Simulate attacker path
        t = 0
        xA_t, yA_t = xA, yA
        xD_t, yD_t = xD, yD
        
        path_A = [(xA_t, yA_t)]
        path_D = [(xD_t, yD_t)]
        times = [0]
        
        # Move until attacker reaches obstacle
        while yA_t > self.obs_top + 0.01:
            # Direction to target
            dx_A = xA_target - xA_t
            dy_A = self.obs_top - yA_t
            
            # Unit vector toward target
            dist_A = np.sqrt(dx_A**2 + dy_A**2)
            if dist_A > 0:
                uA_x = dx_A / dist_A
                uA_y = dy_A / dist_A
            else:
                uA_x, uA_y = 0, -1
            
            # Attacker move
            xA_t += self.vA * uA_x * dt
            yA_t += self.vA * uA_y * dt
            
            # Defender move toward its target
            dx_D = xD_target - xD_t
            dy_D = self.obs_bottom - yD_t
            
            dist_D = np.sqrt(dx_D**2 + dy_D**2)
            if dist_D > 0:
                uD_x = dx_D / dist_D
                uD_y = dy_D / dist_D
            else:
                uD_x, uD_y = 0, 0
            
            xD_t += self.vD * uD_x * dt
            yD_t += self.vD * uD_y * dt
            
            path_A.append((xA_t, yA_t))
            path_D.append((xD_t, yD_t))
            times.append(t)
            
            t += dt
        
        return np.array(path_A), np.array(path_D), times
    
    def plot_phase1_paths(self, xA, yA, xD, yD):
        "Plot the optimal paths in Phase 1"
        path_A, path_D, times = self.simulate_optimal_paths(xA, yA, xD, yD)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw arena
        self.geo.plot_arena(ax)

        V, xA_target, xD_target = self.value(xA, yA, xD, yD)
    
        # Clear the axes and redraw everything manually
        ax.clear()
        
        # Draw arena boundaries
        ax.plot([-self.geo.L, self.geo.L, self.geo.L, -self.geo.L, -self.geo.L],
                [-self.geo.H, -self.geo.H, self.geo.H, self.geo.H, -self.geo.H], 'k-', linewidth=2)
            
        # Draw obstacle
        circle = plt.Circle((0, 0), self.obs_radius, color='gray', alpha=0.5, label='Obstacle')
        ax.add_patch(circle)

        # Draw target (bottom boundary)
        ax.axhline(y=self.geo.target_y, color='green', linewidth=3, label='Target', alpha=0.7)
        
        # Draw paths
        ax.plot(path_A[:, 0], path_A[:, 1], 'r-', linewidth=2, label='Attacker path')
        ax.plot(path_D[:, 0], path_D[:, 1], 'b-', linewidth=2, label='Defender path')
        
        # Mark start points
        ax.plot(xA, yA, 'ro', markersize=8, label=f'Attacker start')
        ax.plot(xD, yD, 'bo', markersize=8, label=f'Defender start')
        
        # Mark interface points
        ax.plot(xA_target, self.obs_top, 'r*', markersize=12, label='Attacker at obstacle')
        ax.plot(xD_target, self.obs_bottom, 'b*', markersize=12, label='Defender at obstacle')
            
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Phase 1 Optimal Paths (V={self.value(xA,yA,xD,yD)[0]:.3f})')
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-self.geo.L-0.5, self.geo.L+0.5)
        ax.set_ylim(-self.geo.H-0.5, self.geo.H+0.5)
        
        return fig, ax


# Test Phase 1
if __name__ == "__main__":
    
    print("PHASE 1 SOLVER TEST - APPROACH TO OBSTACLE")
    
    # Create geometry
    geo = GameGeometry(epsilon=0.5)
    T_max = 2.15  # Maximum time before draw
    
    # Test with different speed ratios
    speed_ratios = [0.3, 0.5, 0.6, 1.2]
    
    for ratio in speed_ratios:
        print(f"\nSpeed Ratio vD/vA = {ratio}")
        
        # Create solvers
        vA = 1.0
        vD = vA * ratio
        phase1 = Phase1Solver(geo, vA, vD, T_max)
        
        # Starting positions
        xA_start, yA_start = 0.0, 3.0  # Attacker above
        xD_start, yD_start = 0.0, -3.0  # Defender below
        
        # Compute value
        V, xA_interface, xD_interface = phase1.value(xA_start, yA_start, xD_start, yD_start)
        
        print(f"Start: A=({xA_start},{yA_start}), D=({xD_start},{yD_start})")
        print(f"Optimal interface: A→({xA_interface:.2f},{phase1.obs_top:.1f}), D→({xD_interface:.2f},{phase1.obs_bottom:.1f})")
        print(f"Game value: {V:.3f}")
        
        # Interpret the game value
        if V > 0:
            if V > 1.5:
                print("ATTACKER WINS DECISIVELY")
            elif V > 0.5:
                print("ATTACKER HAS CLEAR ADVANTAGE")
            else:
                print("ATTACKER HAS SLIGHT EDGE")
            print(f"Attacker reaches target with defender {V:.2f} units away")
        elif V < 0:
            if V < -1.5:
                print("DEFENDER WINS DECISIVELY")
            elif V < -0.5:
                print("DEFENDER HAS CLEAR ADVANTAGE")
            else:
                print("DEFENDER HAS SLIGHT EDGE")
            print(f"Defender catches attacker {abs(V):.2f} units above target")
        else:
            print("PERFECTLY BALANCED - DRAW")
            print(f"Game ends in tie at target")
        
        # Strategic insight
        print(f"\nSTRATEGIC INSIGHT:")
        if xA_interface < -0.5:
            print(f"Attacker commits strongly to LEFT side to force mismatch")
        elif xA_interface > 0.5:
            print(f"Attacker commits strongly to RIGHT side to force mismatch")
        elif abs(xA_interface) < 0.2:
            print(f"Attacker stays centered, keeping options open")
        else:
            print(f"Attacker drifts slightly to create uncertainty")
            
        if xD_interface > 0.3:
            print(f"Defender biases toward RIGHT side to counter attacker's drift")
        elif xD_interface < -0.3:
            print(f"Defender biases toward LEFT side to counter attacker's drift")
        else:
            print(f"Defender stays centered, ready to react")

        # Plot paths
        fig, ax = phase1.plot_phase1_paths(xA_start, yA_start, xD_start, yD_start)
        plt.show()
