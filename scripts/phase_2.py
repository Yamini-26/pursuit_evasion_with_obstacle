import numpy as np
import matplotlib.pyplot as plt
from geometry import GameGeometry
from phase_3 import Phase3Solver


class Phase2Solver:
    "Solution for Phase 2 (at the obstacle) - the decision at the obstacle"
    
    def __init__(self, geometry: GameGeometry, vA: float, vD: float, T_max: float):
        self.geo = geometry
        self.vA = vA
        self.vD = vD
        self.T_max = T_max
        self.phase3 = Phase3Solver(geometry, vA, vD, T_max)
        
        # Obstacle parameters
        self.obs_radius = geometry.obstacle_radius
        self.obs_center = geometry.obstacle_center
    
    def get_clearance_point(self, side):
        "Get the point where player clears obstacle on given side"

        if side == 'L':
            return (-self.obs_radius - 0.1, -self.obs_radius - 0.1)
        else:
            return (self.obs_radius + 0.1, -self.obs_radius - 0.1)

    def time_to_reach(self, start, target, speed):
        "Straight-line time from start to target"

        dx = target[0] - start[0]
        dy = target[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2) 
        time = distance / speed
        return time

    def build_payoff_matrix(self, xA_start=(0, 1.1), xD_start=(0, -1.1)):
        matrix = np.zeros((2, 2))
        
        for i, a_side in enumerate(['L', 'R']):
            for j, d_side in enumerate(['L', 'R']):
                
                # Get clearance points
                A_clear = self.get_clearance_point(a_side)
                D_clear = self.get_clearance_point(d_side)
                
                # Times to reach clearance points
                t_A_clear = self.time_to_reach(xA_start, A_clear, self.vA)
                t_D_clear = self.time_to_reach(xD_start, D_clear, self.vD)
                
                if a_side == d_side:
                    # SAME SIDE
                    if t_D_clear < t_A_clear:
                        # Defender arrives first at clearance point
                        wait_time = t_A_clear - t_D_clear
                        
                        # During wait time, defender can move UP toward attacker
                        # Defender's position when attacker clears obstacle
                        y_defender_when_A_clears = D_clear[1] + self.vD * wait_time
                        
                        # But defender can't go above obstacle bottom? Actually they can go up
                        # Let's find where defender ends up after wait time
                        if y_defender_when_A_clears >= A_clear[1]:
                            # Defender can reach attacker's height - they meet!
                            # Use Phase 3 with defender at same height as attacker
                            matrix[i, j] = self.phase3.payoff(
                                A_clear[0], A_clear[1],
                                A_clear[0], A_clear[1]  # Defender at same point
                            )
                        else:
                            # Defender is below attacker when Phase 3 starts
                            # Use Phase 3 with defender at (D_clear[0], y_defender_when_A_clears)
                            matrix[i, j] = self.phase3.payoff(
                                A_clear[0], A_clear[1],
                                D_clear[0], y_defender_when_A_clears
                            )
                    else:
                        # Attacker arrives first at clearance point
                        head_start = t_D_clear - t_A_clear
                        
                        # Attacker has been moving down during head start
                        y_attacker_when_D_arrives = A_clear[1] - self.vA * head_start
                        
                        # When defender arrives, attacker is already below
                        # Use Phase 3 with attacker ahead
                        matrix[i, j] = self.phase3.payoff(
                            A_clear[0], y_attacker_when_D_arrives,
                            D_clear[0], D_clear[1]
                        )
                else:
                    # DIFFERENT SIDES
                    # Defender goes to wrong side first
                    t_D_wrong = self.time_to_reach(xD_start, D_clear, self.vD)
                    
                    # Then switch to correct side
                    correct_clear = self.get_clearance_point(a_side)
                    t_D_switch = self.time_to_reach(D_clear, correct_clear, self.vD)
                    
                    t_D_total = t_D_wrong + t_D_switch
                    
                    # Attacker's time to clear on their chosen side
                    t_A_clear = self.time_to_reach(xA_start, correct_clear, self.vA)
                    
                    if t_D_total < t_A_clear:
                        # Defender arrives at correct side BEFORE attacker clears
                        wait_time = t_A_clear - t_D_total
                        
                        # Defender can move up while waiting
                        y_defender_when_A_clears = correct_clear[1] + self.vD * wait_time
                        
                        matrix[i, j] = self.phase3.payoff(
                            correct_clear[0], correct_clear[1],
                            correct_clear[0], y_defender_when_A_clears
                        )
                    else:
                        # Attacker clears before defender arrives
                        head_start = t_D_total - t_A_clear
                        
                        # Attacker moves down during head start
                        y_attacker_when_D_arrives = correct_clear[1] - self.vA * head_start
                        
                        matrix[i, j] = self.phase3.payoff(
                            correct_clear[0], y_attacker_when_D_arrives,
                            correct_clear[0], correct_clear[1]
                        )
        
        return matrix
    
    def solve_nash_equilibrium(self, matrix):
        """
        Solve for mixed strategy Nash equilibrium in 2x2 game
        Returns: (p, q, V) where
        p = probability attacker goes Left
        q = probability defender guards Left
        V = value of the game
        """
        a, b = matrix[0, 0], matrix[0, 1]  # Attacker L
        c, d = matrix[1, 0], matrix[1, 1]  # Attacker R
        
        # Check for pure strategy equilibria first
        # Defender's best responses
        defender_best_vs_L = 'L' if a <= c else 'R'  # Defender minimizes
        defender_best_vs_R = 'L' if b <= d else 'R'
        
        attacker_best_vs_L = 'L' if a >= b else 'R'  # Attacker maximizes
        attacker_best_vs_R = 'L' if c >= d else 'R'
        
        # Check if pure equilibrium exists
        if defender_best_vs_L == 'L' and attacker_best_vs_L == 'L':
            return 1.0, 1.0, a  # (L, L) equilibrium
        if defender_best_vs_R == 'R' and attacker_best_vs_R == 'R':
            return 0.0, 0.0, d  # (R, R) equilibrium
        if defender_best_vs_L == 'R' and attacker_best_vs_R == 'L':
            return 0.0, 1.0, b  # (L, R) equilibrium - check signs
        
        # Mixed strategy equilibrium
        # Attacker makes defender indifferent between L and R
        # q*a + (1-q)*c = q*b + (1-q)*d
        # Solve for q
        denominator = a - b - c + d
        if abs(denominator) < 1e-10:
            q = 0.5  # Avoid division by zero
        else:
            q = (d - c) / denominator
        
        # Defender makes attacker indifferent between L and R
        # p*a + (1-p)*b = p*c + (1-p)*d
        if abs(denominator) < 1e-10:
            p = 0.5
        else:
            p = (d - b) / denominator
        
        # Clip to [0, 1]
        p = np.clip(p, 0, 1)
        q = np.clip(q, 0, 1)
        
        # Value of the game
        V = p * (q * a + (1-q) * b) + (1-p) * (q * c + (1-q) * d)
        
        return p, q, V
    
    def analyze_speed_ratio(self, vA_fixed=1.0):
        "Analyze how equilibrium changes with speed ratio vD/vA"
        speed_ratios = np.linspace(0.5, 2.0, 20)
        p_vals = []
        q_vals = []
        V_vals = []
        
        for ratio in speed_ratios:
            # Create solver with this speed ratio
            solver = Phase2Solver(self.geo, vA_fixed, vA_fixed * ratio, self.T_max)
            matrix = solver.build_payoff_matrix()
            p, q, V = solver.solve_nash_equilibrium(matrix)
            p_vals.append(p)
            q_vals.append(q)
            V_vals.append(V)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(speed_ratios, p_vals, 'b-', linewidth=2, label='p (Attacker L)')
        ax1.plot(speed_ratios, q_vals, 'r--', linewidth=2, label='q (Defender L)')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Speed Ratio vD/vA')
        ax1.set_ylabel('Probability')
        ax1.set_title('Mixed Strategy Probabilities vs Speed Ratio')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(speed_ratios, V_vals, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Speed Ratio vD/vA')
        ax2.set_ylabel('Game Value')
        ax2.set_title('Value of Game (+=Attacker, -=Defender)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def print_matrix_with_equilibrium(self, matrix, p, q, V):
        print("PAYOFF MATRIX (Attacker payoff)")
        print(f"{'':<15} {'Defender L':<12} {'Defender R':<12}")
        print("-" * 40)
        print(f"{'Attacker L':<15} {matrix[0,0]:>+8.3f}     {matrix[0,1]:>+8.3f}")
        print(f"{'Attacker R':<15} {matrix[1,0]:>+8.3f}     {matrix[1,1]:>+8.3f}")
        print("-" * 40)
        # Interpret each value
        print("\nInterpretation:")
        for i, a_side in enumerate(['L', 'R']):
            for j, d_side in enumerate(['L', 'R']):
                val = matrix[i, j]
                if val < 0:
                    print(f"  A={a_side}, D={d_side}: Defender catches attacker {abs(val):.2f} units above target")
                elif val > 0:
                    print(f"  A={a_side}, D={d_side}: Attacker reaches target with defender {val:.2f} units away")
                else:
                    print(f"  A={a_side}, D={d_side}: DRAW (time runs out)")
        print(f"\nNASH EQUILIBRIUM:")
        print(f"  Attacker goes LEFT with probability p = {p:.3f}")
        print(f"  Defender guards LEFT with probability q = {q:.3f}")
        print(f"  Value of game V = {V:.3f} ", end="")
        if V > 0:
            print(f"(Attacker advantage - reaches target with defender {V:.2f} units away)")
        elif V < 0:
            print(f"(Defender advantage - catches attacker {abs(V):.2f} units above target)")
        else:
            print(f"(Fair game - DRAW on average)")


# Test Phase 2
if __name__ == "__main__":
    
    print("PHASE 2 SOLVER TEST - OBSTACLE DECISION POINT")
    
    # Create geometry
    geo = GameGeometry(epsilon=0.5)
    T_max = 2.0  # Maximum time before draw
    
    # Test different speed ratios
    speed_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    
    print("Testing different speed ratios")
    
    for ratio in speed_ratios:
        print(f"\nSpeed Ratio vD/vA = {ratio}")
        solver = Phase2Solver(geo, vA=1.0, vD=1.0*ratio, T_max=T_max)
        
        # Build payoff matrix (using default interface positions)
        matrix = solver.build_payoff_matrix(
            xA_start=(0, 1.1),   # Attacker just above obstacle
            xD_start=(0, -1.1)   # Defender just below obstacle
        )
        
        # Solve for Nash equilibrium
        p, q, V = solver.solve_nash_equilibrium(matrix)
        
        # Print results
        solver.print_matrix_with_equilibrium(matrix, p, q, V)
    
    # Analyze across speed ratios
    print("Analyzing equilibrium vs speed ratio")
    
    solver = Phase2Solver(geo, vA=1.0, vD=1.2, T_max=T_max)
    fig, axes = solver.analyze_speed_ratio()
    plt.suptitle("Phase 2: Nash Equilibrium vs Speed Ratio")
    plt.show()
