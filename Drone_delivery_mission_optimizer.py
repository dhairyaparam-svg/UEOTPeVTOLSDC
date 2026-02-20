import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------
# CONSTANTS & BOUNDS
# ------------------------------------------------------------------------------------------------------------

# Mission Constraints
rate_of_climb_bounds = [-3, 5]      # m/s (allows descent/landing)
cruise_speed_bounds = [5, 25]       # m/s
optimal_altitude_bounds = [50, 4000] # m
optimal_throttle_bounds = [0.5, 1.5]

# Loiter Parameters (NEW for delivery mission)
loiter_altitude_bounds = [10, 100]  # m (low altitude for delivery)
loiter_duration_bounds = [80]  # seconds (time for package handoff)
loiter_speed = 1.0                  # m/s (slow hover/circular pattern)

# Physical Parameters
weight = 2                          # kg
number_of_rotors = 4
prop_diameter_inches = 10 
aerodynamic_thrust_loss = 0.05 
additional_thrust_factor = 0.0003

# Aerodynamics
full_face_drag_coefficient = 0.2 
zero_angle_downforce_coefficient = 0.02
downforce_coefficient_slope = 2 * ca.pi * 0.07 
zero_angle_drag_coefficient_cd0 = 0.098 
drag_slope_k1 = 0.532 

# Propeller/Motor Parameters
efficiency_density_slope_k3 = 100 
efficiency_density_intercept_m1_0 = 4.73 
motor_efficiency_ground_g_per_w = 9.77 
density_thrust_loss_factor = 1 

# Battery & ESC
battery_cells = 4
battery_voltage_full = 4.2  
batteries_in_parallel = 1
battery_capacity_mah = 5200
reserve_power_factor = 0.2
esc_efficiency = 0.9

# Atmospheric Constants (ISA Model)
rho_0 = 1.225
temp_lapse_rate = -0.0065
temp_0 = 288.15
gravity = 9.80665
m_air = 0.0289644
r_gas = 8.314462618

# Wind Model Parameters
v_wind_ref = 25
k_wind = 3.74 * 2 * 0.1  # Adjusted for more realistic wind profile (original was too aggressive)
h_max_wind = 300  
omega_wind = [0.5, 0.9, 2.42, -5.1, 0.98, 1.1] 
c_wind = [0.82230367, -0.97959362, 0.227143, 0.49432766, -0.76617972, -0.1597865] 

# Solver Parameters
N = 150  # Good balance between resolution and speed
solver_type = 'ipopt' 
max_iter = 2000  # Increased for more complex optimization

# ------------------------------------------------------------------------------------------------------------
# PRE-CALCULATIONS
# ------------------------------------------------------------------------------------------------------------

rotor_area = number_of_rotors * ca.pi * ((0.0127 * prop_diameter_inches)**2)
eff_density_intercept = (efficiency_density_intercept_m1_0 + (rho_0 * efficiency_density_slope_k3)) / 1000  
eff_density_slope = efficiency_density_slope_k3 / 1000
eff_intercept_slope_m3 = (motor_efficiency_ground_g_per_w / (1000 * rho_0))
min_throttle_adj = optimal_throttle_bounds[0] + additional_thrust_factor + aerodynamic_thrust_loss
max_throttle = optimal_throttle_bounds[1]

energy_total_available = (
    (3.6 * battery_capacity_mah)
    * batteries_in_parallel
    * battery_cells 
    * battery_voltage_full
    * esc_efficiency
    * (1 - reserve_power_factor)
    * 0.7
)

# ------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------------------------------------

def calculate_air_density(h):
    return rho_0 * (1 + (temp_lapse_rate * h) / temp_0) ** ((gravity * m_air) / ((r_gas * temp_lapse_rate) - 1))

def get_wind_soar_multiplier(h):
    val_at_h_max = 0.999
    log_term1 = ca.log(val_at_h_max / h_max_wind)
    log_term2 = ca.log(val_at_h_max) / h_max_wind
    return (1 - (ca.exp(h * (log_term1 / h_max_wind)))) * (ca.exp(log_term2 * h))

def get_wind_speed(h):
    norm_h = h / 5000.0 
    exp_term = ca.exp(-k_wind * norm_h)
    series_sum = 0
    half_size = len(omega_wind) // 2
    for i in range(half_size):
        series_sum += c_wind[i] * ca.sin(omega_wind[i] * norm_h)
    for i in range(half_size):
        series_sum += c_wind[i + half_size] * ca.cos(omega_wind[i + half_size] * norm_h)
    return get_wind_soar_multiplier(h) * exp_term * v_wind_ref * series_sum / len(omega_wind)

def find_optimal_wind_altitude():
    alts = np.linspace(optimal_altitude_bounds[0], optimal_altitude_bounds[1], 50)
    wind_speeds = [float(get_wind_speed(a)) for a in alts]
    idx = np.argmax(wind_speeds)
    return alts[idx], wind_speeds[idx]

def get_instantaneous_power(vx_ground, vz_climb, h, thrust_input, v_wind_override=None):
    rho = calculate_air_density(h)
    if v_wind_override is not None:
        v_wind = v_wind_override
    else:
        v_wind = get_wind_speed(h)
    vx_air = vx_ground + v_wind 
    v_total_air = ca.sqrt(vx_air**2 + vz_climb**2 + 1e-4) # Small epsilon for stability
    efficiency_m3 = eff_intercept_slope_m3 * rho
    efficiency_m1 = (eff_density_intercept - (eff_density_slope * rho))
    actual_thrust = (1 - aerodynamic_thrust_loss) * thrust_input * (rho / rho_0)
    thrust_coeff = actual_thrust / (max_throttle * weight)
    efficiency_term = (efficiency_m3 - (efficiency_m1 * thrust_coeff))
    efficiency_term = ca.fmax(efficiency_term, 0.001) 
    tilt_factor = ca.fmin(ca.fmax(weight / actual_thrust, -0.99), 0.99) 
    sqrt_tilt = ca.sqrt(1 - tilt_factor**2)
    
    # Numerically stable drag calculation
    # Weights for climb vs horizontal drag, summing to 1 except for epsilon
    w_climb = vz_climb**2 / (vx_air**2 + vz_climb**2 + 1e-6)
    w_horiz = vx_air**2 / (vx_air**2 + vz_climb**2 + 1e-6)
    
    drag_coeff = (w_climb * full_face_drag_coefficient) + \
                 (w_horiz * (zero_angle_drag_coefficient_cd0 + \
                 (drag_slope_k1 * (ca.arcsin(sqrt_tilt) ** 2))))
    
    drag_power_numerator = (vx_air**2 + vz_climb**2) * sqrt_tilt * rho * drag_coeff * rotor_area
    downforce_coeff = zero_angle_downforce_coefficient + (downforce_coefficient_slope * (ca.arcsin(sqrt_tilt)))
    downforce = rotor_area * rho * downforce_coeff * (vx_air**2)
    return (drag_power_numerator / efficiency_term) + ((weight + (downforce / gravity)) / efficiency_term) + (weight * gravity * vz_climb)

# ------------------------------------------------------------------------------------------------------------
# RIGID BODY DYNAMICS & L1 CONTROLLER (Simulating Real Drone)
# ------------------------------------------------------------------------------------------------------------

class RigidBodyDrone:
    def __init__(self, m=2.0, Iyy=0.02):
        self.m = m
        self.Iyy = Iyy  # Simplification for 2D pitch-plane dynamics
        self.g = 9.80665
        
        # State: [x, z, vx, vz, theta, q]
        # x, z: position
        # vx, vz: velocity
        # theta: pitch angle (positive is pitch up/backward, negative is pitch down/forward)
        # q: pitch rate
        self.state = np.zeros(6)
        
    def dynamics(self, state, thrust, tau, v_wind=0):
        x, z, vx, vz, theta, q = state
        
        # Relative velocity (air velocity)
        vx_air = vx + v_wind
        v_total_air = np.sqrt(vx_air**2 + vz**2 + 1e-6)
        
        # Simplified drag model (matching optimizer's intent roughly)
        cd = 0.2 if abs(vz) > abs(vx_air) else 0.05
        rho = 1.225 # Simplified for simulation step
        drag_force = 0.5 * rho * v_total_air**2 * cd * 0.1 # 0.1 is effective area approx
        
        drag_x = -drag_force * (vx_air / v_total_air)
        drag_z = -drag_force * (vz / v_total_air)
        
        # Equations of motion
        # Note: In our coordinate system, theta=0 is hover. 
        # Negative theta tilts drone forward -> positive vx acceleration.
        ax = (thrust / self.m) * np.sin(-theta) + (drag_x / self.m)
        az = (thrust / self.m) * np.cos(theta) - self.g + (drag_z / self.m)
        alpha = tau / self.Iyy
        
        return np.array([vx, vz, ax, az, q, alpha])

    def step(self, thrust, tau, dt, v_wind=0):
        # RK4 integration
        k1 = self.dynamics(self.state, thrust, tau, v_wind)
        k2 = self.dynamics(self.state + 0.5 * dt * k1, thrust, tau, v_wind)
        k3 = self.dynamics(self.state + 0.5 * dt * k2, thrust, tau, v_wind)
        k4 = self.dynamics(self.state + dt * k3, thrust, tau, v_wind)
        self.state += (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

class CascadedPIDController:
    def __init__(self):
        # Position Loop Gains
        self.kp_pos_x = 1.2
        self.kp_pos_z = 2.0
        
        # Velocity Loop Gains
        self.kp_vel_x = 1.0
        self.ki_vel_x = 0
        self.kd_vel_x = 3
        self.kp_vel_z = 2.0
        self.ki_vel_z = 0
        self.kd_vel_z = 5
        
        # Attitude Loop Gains (Theta)
        self.kp_theta = 30.0
        self.kd_theta = 8.0
        
        # Integrator states for I-terms
        self.vel_x_int = 0.0
        self.vel_z_int = 0.0
        
    def update(self, drone_state, target_pos, target_vel, dt):
        """
        Cascaded PID Control Loop:
        1. Position Error -> Velocity Command
        2. Velocity Error -> Acceleration Command
        3. Acceleration -> Target Attitude & Thrust
        4. Attitude Error -> Torque (tau)
        """
        x, z, vx, vz, theta, q = drone_state
        tx, tz = target_pos
        tvx, tvz = target_vel
        
        # 1. Outer Loop: Position to Velocity
        # We add the feed-forward target velocity from the optimizer
        vx_cmd = self.kp_pos_x * (tx - x) + tvx
        vz_cmd = self.kp_pos_z * (tz - z) + tvz
        
        # Physical constraints on commanded velocity
        vx_cmd = np.clip(vx_cmd, -30, 30)
        vz_cmd = np.clip(vz_cmd, -8, 8)
        
        # 2. Middle Loop: Velocity to Acceleration
        err_vx = vx_cmd - vx
        err_vz = vz_cmd - vz
        
        self.vel_x_int += err_vx * dt
        self.vel_z_int += err_vz * dt
        
        # Anti-windup for integrators
        self.vel_x_int = np.clip(self.vel_x_int, -10, 10)
        self.vel_z_int = np.clip(self.vel_z_int, -5, 5)
        
        ax_cmd = self.kp_vel_x * err_vx + self.ki_vel_x * self.vel_x_int + (self.kd_vel_x * (err_vx - self.vel_x_int))
        az_cmd = self.kp_vel_z * err_vz + self.ki_vel_z * self.vel_z_int + (self.kd_vel_z * (err_vz - self.vel_z_int))
        
        # 3. Conversion: Acceleration to Thrust and Pitch
        # g compensation is critical here
        acc_total_z = az_cmd + 9.80665
        acc_total_x = ax_cmd
        
        # Thrust required to achieve total acceleration vector (m=2.0)
        thrust_cmd = 2.0 * np.sqrt(acc_total_x**2 + acc_total_z**2)
        
        # Target theta to align thrust vector with ax_cmd
        # Negative theta = pitch forward = positive x acceleration
        theta_cmd = -np.arctan2(acc_total_x, acc_total_z)
        
        # 4. Inner Loop: Attitude to Torque
        # Simple PD on angle and rate
        tau = self.kp_theta * (theta_cmd - theta) - self.kd_theta * q
        
        # Physical saturation
        thrust_cmd = np.clip(thrust_cmd, 5, 50) # 0.25g to 2.5g
        tau = np.clip(tau, -5.0, 5.0)
        
        return thrust_cmd, tau, theta_cmd

def run_real_drone_simulation(opt_result):
    if not opt_result['success']: return None
    
    dt_sim = 0.01 # Faster timestep for PID stability
    t_end = opt_result['Total_Time']
    steps = int(t_end / dt_sim)
    
    drone = RigidBodyDrone(m=weight)
    controller = CascadedPIDController()
    
    # Interpolate optimized trajectory
    t_opt = np.linspace(0, t_end, N+1)
    x_ref = np.interp(np.linspace(0, t_end, steps), t_opt, opt_result['X'])
    z_ref = np.interp(np.linspace(0, t_end, steps), t_opt, opt_result['H'])
    vx_ref = np.interp(np.linspace(0, t_end, steps), t_opt[1:], opt_result['Vx'])
    vz_ref = np.interp(np.linspace(0, t_end, steps), t_opt[1:], opt_result['Vz'])
    
    sim_data = {'x': [], 'z': [], 'vx': [], 'vz': [], 'theta': [], 't': [], 'theta_cmd': []}
    
    # Init state
    drone.state[0] = opt_result['X'][0]
    drone.state[1] = opt_result['H'][0]
    
    for i in range(steps):
        t = i * dt_sim
        tar_pos = (x_ref[i], z_ref[i])
        tar_vel = (vx_ref[i], vz_ref[i])
        
        v_wind = float(get_wind_speed(drone.state[1]))
        
        thrust, tau, th_cmd = controller.update(drone.state, tar_pos, tar_vel, dt_sim)
        drone.step(thrust, tau, dt_sim, v_wind)
        
        # Sub-sample data to keep plot manageable
        if i % 5 == 0:
            sim_data['x'].append(drone.state[0])
            sim_data['z'].append(drone.state[1])
            sim_data['vx'].append(drone.state[2])
            sim_data['vz'].append(drone.state[3])
            sim_data['theta'].append(drone.state[4])
            sim_data['theta_cmd'].append(th_cmd)
            sim_data['t'].append(t)
        
    return sim_data

# ------------------------------------------------------------------------------------------------------------
# DELIVERY MISSION OPTIMIZER (7-Phase Mission)
# ------------------------------------------------------------------------------------------------------------

def solve_delivery_mission(loiter_duration=60, verbose=True):
    """
    WIND-AWARE delivery mission optimization that actively exploits wind conditions:
    
    STRATEGY:
    - Outbound: Fly at HIGHER altitude to benefit from tailwind (wind is negative)
    - Return: Fly at LOWER altitude to avoid headwind
    - Energy cost of climbing is balanced against wind benefits
    
    Phase 1: Climb HIGH (outbound)
    Phase 2: Cruise HIGH with tailwind (outbound)
    Phase 3: Descent to delivery point
    Phase 4: Loiter at delivery point
    Phase 5: Climb to MODERATE altitude (return)
    Phase 6: Cruise LOW to avoid headwind (return)
    Phase 7: Descent to home
    
    Args:
        loiter_duration: Time spent loitering at delivery point (seconds)
        verbose: Print optimization progress
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Wind-Aware Delivery Mission (Loiter: {loiter_duration}s)")
        print(f"Strategy: HIGH altitude outbound (tailwind), LOW altitude return (avoid headwind)")
        print(f"{'='*80}")
    
    opti = ca.Opti()
    
    # State variables
    X = opti.variable(N+1)      # Horizontal position
    H = opti.variable(N+1)      # Altitude
    E = opti.variable(N+1)      # Energy consumed
    
    # Control variables
    Vx = opti.variable(N)       # Horizontal velocity
    Vz = opti.variable(N)       # Vertical velocity
    Thrust = opti.variable(N)   # Thrust
    
    # Mission parameters
    Total_Time = opti.variable()
    Delivery_Distance = opti.variable()  # How far we can deliver
    
    # Wind-aware altitude variables
    H_outbound_cruise = opti.variable()  # Altitude during outbound cruise (HIGH for tailwind)
    H_return_cruise = opti.variable()    # Altitude during return cruise (LOW to avoid headwind)
    
    dt = Total_Time / N
    
    # Objective: Maximize delivery distance
    opti.minimize(-Delivery_Distance)
    
    # ------------------------------------------------------------------------------------------------------------
    # INITIAL & FINAL CONDITIONS
    # ------------------------------------------------------------------------------------------------------------
    
    # Start at home, ground level
    opti.subject_to(X[0] == 0)
    opti.subject_to(H[0] == 0)
    opti.subject_to(E[0] == 0)
    opti.subject_to(Vx[0] == 0)
    
    # End at home, ground level
    opti.subject_to(X[N] == 0)
    opti.subject_to(H[N] == 0)
    opti.subject_to(Vx[N-1] == 0)
    
    # ------------------------------------------------------------------------------------------------------------
    # DYNAMICS
    # ------------------------------------------------------------------------------------------------------------
    
    for k in range(N):
        opti.subject_to(X[k+1] == X[k] + Vx[k] * dt)
        opti.subject_to(H[k+1] == H[k] + Vz[k] * dt)
        P_inst = get_instantaneous_power(Vx[k], Vz[k], H[k], Thrust[k])
        opti.subject_to(E[k+1] == E[k] + P_inst * dt)
    
    # ------------------------------------------------------------------------------------------------------------
    # GENERAL BOUNDS
    # ------------------------------------------------------------------------------------------------------------
    
    opti.subject_to(E[N] <= energy_total_available)
    opti.subject_to(opti.bounded(0, H, 4500))
    opti.subject_to(opti.bounded(-25, Vx, 25))  # Allow negative for return
    opti.subject_to(opti.bounded(rate_of_climb_bounds[0], Vz, rate_of_climb_bounds[1]))
    opti.subject_to(opti.bounded(weight * min_throttle_adj, Thrust, weight * max_throttle))
    opti.subject_to(opti.bounded(120, Total_Time, 7200))
    opti.subject_to(opti.bounded(1000, Delivery_Distance, 15000))  # Reasonable delivery range
    
    # Wind-aware altitude bounds
    # Wind-aware altitude bounds
    opti.subject_to(opti.bounded(300, H_outbound_cruise, 2000))   # Higher for tailwind
    opti.subject_to(opti.bounded(50, H_return_cruise, 800))      # Stay airborne, avoid headwind
    
    # KEY: Outbound cruise MUST be higher than return cruise for this wind profile
    opti.subject_to(H_outbound_cruise >= H_return_cruise + 100)  # Relaxed slightly for feasibility
    
    # ------------------------------------------------------------------------------------------------------------
    # MISSION STRUCTURE CONSTRAINTS
    # ------------------------------------------------------------------------------------------------------------
    
    # Position constraints: must reach delivery distance and return
    mid_point = N // 2
    outbound_cruise_start = int(N * 0.15)
    outbound_cruise_end = int(N * 0.35)
    return_cruise_start = int(N * 0.65)
    return_cruise_end = int(N * 0.85)
    final_descent_start = int(N * 0.97)  # Only descend in final 10% of mission
    
    # At midpoint, should be near delivery location
    opti.subject_to(X[mid_point] >= Delivery_Distance * 0.95)
    opti.subject_to(X[mid_point] <= Delivery_Distance * 1.05)
    
    # Position should not exceed delivery distance significantly
    for k in range(N+1):
        opti.subject_to(X[k] >= -100)  # Small tolerance for numerical stability
        opti.subject_to(X[k] <= Delivery_Distance * 1.1)
    
    # Outbound: generally positive velocity in first third
    for k in range(5, mid_point - 15):
        opti.subject_to(Vx[k] >= -3)  # Mostly forward (with tolerance)
    
    # Return: generally negative velocity in last third
    for k in range(mid_point + 15, N - 5):
        opti.subject_to(Vx[k] <= 3)  # Mostly backward (with tolerance)
    
    # Wind-aware altitude constraints during cruise phases
    for k in range(outbound_cruise_start, outbound_cruise_end):
        # Outbound cruise: maintain HIGH altitude for tailwind
        opti.subject_to(H[k] >= H_outbound_cruise - 80)
        opti.subject_to(H[k] <= H_outbound_cruise + 80)
    
    for k in range(return_cruise_start, return_cruise_end):
        # Return cruise: maintain LOW altitude to avoid headwind
        opti.subject_to(H[k] >= H_return_cruise - 80)
        opti.subject_to(H[k] <= H_return_cruise + 80)
    
    # Prevent early descent: maintain cruise altitude until very close to home
    # The return cruise phase ends at 85%, and final descent starts later
    for k in range(return_cruise_start, final_descent_start):
        # Must stay within 100m of the target return cruise altitude, and always above 100m
        opti.subject_to(H[k] >= H_return_cruise)
        opti.subject_to(H[k] >= optimal_altitude_bounds[0])  
    
    # Explicitly prevent traveling at ground level during horizontal return
    for k in range(mid_point + 10, final_descent_start):
        # Ensure altitude is significant if we are far from home
        # If more than 500m from home, stay above 100m
        distance_to_home = X[k]
        opti.subject_to(H[k] >= ca.if_else(distance_to_home > 100, optimal_altitude_bounds[0],0))  # Stay above 100m if more than 100m from home, otherwise can descend
    
    # ------------------------------------------------------------------------------------------------------------
    # LOITER PHASE CONSTRAINTS
    # ------------------------------------------------------------------------------------------------------------
    
    # Loiter happens around the midpoint
    loiter_window = max(5, int(N * 0.08))  # 8% of mission is loiter
    
    for k in range(mid_point - loiter_window, mid_point + loiter_window):
        if 0 <= k < N:
            # Low altitude during loiter (safe for package delivery)
            opti.subject_to(H[k] <= loiter_altitude_bounds[1])
            # Very low horizontal speed during loiter (hovering)
            opti.subject_to(opti.bounded(-loiter_speed, Vx[k], loiter_speed))
    
    # At the exact midpoint, should be nearly stationary and at low altitude
    opti.subject_to(opti.bounded(-1.5, Vx[mid_point], 1.5))
    opti.subject_to(H[mid_point] <= loiter_altitude_bounds[1] + 10)
    
    # ------------------------------------------------------------------------------------------------------------
    # INITIAL GUESSES - CAREFULLY CHOSEN TO AVOID NaN
    # ------------------------------------------------------------------------------------------------------------
    
    opti.set_initial(Total_Time, 900)  # 15 minutes - conservative
    opti.set_initial(Delivery_Distance, 3500)  # 3.5 km - conservative
    opti.set_initial(H_outbound_cruise, 900)   # Start with high outbound
    opti.set_initial(H_return_cruise, 100)     # Start with low return
    
    # Position guess: smooth outbound and return
    x_guess = np.concatenate([
        np.linspace(0, 3500, mid_point + 1),  # Outbound
        np.linspace(3500, 0, N - mid_point)   # Return
    ])
    opti.set_initial(X, x_guess)
    
    # Altitude guess: MUST be positive and reasonable to avoid NaN in power model
    # Use conservative altitudes that won't cause numerical issues
    n_sixth = N // 6
    n_final_descent = N + 1 - 5*n_sixth
    n_cruise_return = int(n_final_descent * 0.7)  # 70% cruise
    n_final_desc = n_final_descent - n_cruise_return  # 30% final descent
    
    h_guess = np.concatenate([
        np.linspace(10, 900, n_sixth),           # Climb HIGH outbound
        np.ones(n_sixth) * 900,                  # Cruise HIGH outbound (tailwind)
        np.linspace(900, 50, n_sixth),           # Descend to loiter
        np.ones(n_sixth) * 50,                   # Loiter
        np.linspace(50, 300, n_sixth),           # Climb to LOW altitude return
        np.ones(n_cruise_return) * 300,          # Cruise LOW return (maintain altitude)
        np.linspace(300, 10, n_final_desc)       # Final descent to home
    ])
    # Ensure no zero altitudes which can cause NaN
    h_guess = np.maximum(h_guess, 5.0)
    opti.set_initial(H, h_guess)
    
    # Velocity guess: smooth and conservative
    vx_guess = np.concatenate([
        np.linspace(0, 10, n_sixth),            # Accelerate
        np.ones(2*n_sixth) * 10,                # Cruise out
        np.linspace(10, 0, n_sixth),            # Decelerate to loiter
        np.linspace(0, -10, n_sixth),           # Accelerate return
        np.ones(N - 5*n_sixth) * -10            # Cruise back
    ])
    opti.set_initial(Vx, vx_guess)
    
    # Vertical velocity: start at zero (no climb/descent initially)
    opti.set_initial(Vz, 0)
    
    # Thrust: start at weight (hovering)
    opti.set_initial(Thrust, weight * 1.1)  # Slightly above weight for safety
    
    # Solver options
    solver_opts = {
        "expand": True,
        "ipopt.print_level": 3 if verbose else 0,
        "ipopt.max_iter": max_iter,
        "ipopt.tol": 1e-5,  # Slightly relaxed tolerance
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 15,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.start_with_resto": "yes",  # Use restoration phase if needed
        "ipopt.required_infeasibility_reduction": 0.9
    }
    opti.solver(solver_type, solver_opts)
    
    try:
        sol = opti.solve()
        
        result = {
            'X': sol.value(X),
            'H': sol.value(H),
            'E': sol.value(E),
            'Vx': sol.value(Vx),
            'Vz': sol.value(Vz),
            'Thrust': sol.value(Thrust),
            'Total_Time': sol.value(Total_Time),
            'Delivery_Distance': sol.value(Delivery_Distance),
            'loiter_duration': loiter_duration,
            'success': True
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"[SUCCESS] OPTIMIZATION SUCCESSFUL")
            print(f"{'='*80}")
            print(f"Delivery Distance: {result['Delivery_Distance']/1000:.2f} km")
            print(f"Total Mission Time: {result['Total_Time']/60:.1f} minutes")
            print(f"Energy Used: {result['E'][-1]:.1f} J / {energy_total_available:.1f} J ({100*result['E'][-1]/energy_total_available:.1f}%)")
            print(f"Max Altitude: {np.max(result['H']):.1f} m")
            print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"\n[FAILED] Optimization failed: {str(e)}\n")
        return {'success': False, 'loiter_duration': loiter_duration}

# ------------------------------------------------------------------------------------------------------------
# BASELINE ONE-WAY MISSION (For Comparison)
# ------------------------------------------------------------------------------------------------------------

def solve_baseline_mission():
    """
    Original one-way mission: Climb → Cruise → Descent
    """
    print("\nRunning Baseline One-Way Mission (Climb -> Cruise -> Descent)...")
    opti = ca.Opti()
    X = opti.variable(N+1); H = opti.variable(N+1); E = opti.variable(N+1)
    Vx = opti.variable(N); Vz = opti.variable(N); Thrust = opti.variable(N)
    Total_Time = opti.variable()
    dt = Total_Time / N

    opti.minimize(-X[N]) 

    opti.subject_to(X[0] == 0); opti.subject_to(H[0] == 0); opti.subject_to(E[0] == 0)
    opti.subject_to(H[N] == 0)
    opti.subject_to(Vx[0] == 0); opti.subject_to(Vx[N-1] == 0)

    for k in range(N):
        opti.subject_to(X[k+1] == X[k] + Vx[k] * dt)
        opti.subject_to(H[k+1] == H[k] + Vz[k] * dt)
        P_inst = get_instantaneous_power(Vx[k], Vz[k], H[k], Thrust[k])
        opti.subject_to(E[k+1] == E[k] + P_inst * dt)

    opti.subject_to(E[N] <= energy_total_available)
    opti.subject_to(opti.bounded(0, H, 4500))
    opti.subject_to(opti.bounded(0, Vx, 35))
    opti.subject_to(opti.bounded(rate_of_climb_bounds[0], Vz, rate_of_climb_bounds[1]))
    opti.subject_to(opti.bounded(weight * min_throttle_adj, Thrust, weight * max_throttle))
    opti.subject_to(opti.bounded(60, Total_Time, 7200))

    opti.set_initial(Total_Time, 1200)
    opti.set_initial(X, np.linspace(0, 10000, N+1))
    opti.set_initial(H, 200)
    opti.set_initial(Vx, 15)
    opti.set_initial(Thrust, weight)

    opti.solver(solver_type, {"ipopt.print_level": 0, "ipopt.max_iter": max_iter, "expand": True})
    
    try:
        sol = opti.solve()
        return {
            'X': sol.value(X), 'H': sol.value(H), 'E': sol.value(E),
            'Vx': sol.value(Vx), 'Vz': sol.value(Vz), 'Thrust': sol.value(Thrust),
            'Total_Time': sol.value(Total_Time), 
            'range': sol.value(X[N]),
            'success': True
        }
    except:
        return {'success': False}

# ------------------------------------------------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------------------------------------------------

def plot_mission_comparison(baseline, delivery_results):
    """
    Compare baseline one-way mission with delivery missions using multiple clean windows
    """
    plt.style.use('seaborn-v0_8')
    
    # Filter successful delivery results
    successful_deliveries = [r for r in delivery_results if r['success']]
    
    if not successful_deliveries:
        print("No successful delivery missions to plot!")
        return
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(successful_deliveries)))
    
    # ============================================================================
    # WINDOW 1: Trajectory Comparison (Altitude vs Distance)
    # ============================================================================
    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(111)
    
    if baseline and baseline['success']:
        ax1.plot(baseline['X']/1000, baseline['H'], 'b-', linewidth=3, 
                label=f"One-Way Mission (Range: {baseline['range']/1000:.1f} km)", alpha=0.7)
    
    for i, res in enumerate(successful_deliveries):
        ax1.plot(res['X']/1000, res['H'], linewidth=2.5, color=colors[i],
                label=f"Delivery (Loiter: {res['loiter_duration']}s, Dist: {res['Delivery_Distance']/1000:.1f} km)")
    
    ax1.set_xlabel('Distance (km)', fontsize=13)
    ax1.set_ylabel('Altitude (m)', fontsize=13)
    ax1.set_title('Mission Trajectory: Altitude vs Distance', fontsize=15, fontweight='bold')
    
    # NEW: Plot real drone traces
    for i, res in enumerate(successful_deliveries):
        if 'sim' in res and res['sim'] is not None:
            ax1.plot(np.array(res['sim']['x'])/1000, res['sim']['z'], '--', 
                    color=colors[i], alpha=0.8, linewidth=1.5, label=f"Real Drone Trace ({res['loiter_duration']}s)")
            
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    fig1.tight_layout()
    
    # ============================================================================
    # WINDOW 2: Wind-Aware Altitude Profile (showing HIGH out, LOW return)
    # ============================================================================
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = fig2.add_subplot(111)
    
    # Plot altitude over time for delivery missions
    for i, res in enumerate(successful_deliveries):
        t = np.linspace(0, res['Total_Time'], N+1)
        ax2.plot(t/60, res['H'], linewidth=2.5, color=colors[i],
                label=f"Optimized: {res['loiter_duration']}s")
        if 'sim' in res and res['sim'] is not None:
            ax2.plot(np.array(res['sim']['t'])/60, res['sim']['z'], '--', 
                    color=colors[i], alpha=0.6, label=f"Real: {res['loiter_duration']}s")
    
    # Add annotations for wind-aware strategy
    mid_time = successful_deliveries[0]['Total_Time'] / 120  # midpoint in minutes
    ax2.axvline(x=mid_time, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Delivery Point')
    ax2.text(mid_time * 0.3, ax2.get_ylim()[1] * 0.9, 'HIGH Altitude\n(Tailwind)', 
            fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax2.text(mid_time * 1.7, ax2.get_ylim()[1] * 0.9, 'LOW Altitude\n(Avoid Headwind)', 
            fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax2.set_xlabel('Time (minutes)', fontsize=13)
    ax2.set_ylabel('Altitude (m)', fontsize=13)
    ax2.set_title('Wind-Aware Altitude Profile: HIGH Outbound, LOW Return', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    fig2.tight_layout()
    
    # ============================================================================
    # WINDOW 3: Velocity Profiles
    # ============================================================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Horizontal velocity
    for i, res in enumerate(successful_deliveries):
        t = np.linspace(0, res['Total_Time'], N)
        ax3a.plot(t/60, res['Vx'], linewidth=2.5, color=colors[i], 
                label=f"Opt: {res['loiter_duration']}s")
        if 'sim' in res and res['sim'] is not None:
            ax3a.plot(np.array(res['sim']['t'])/60, res['sim']['vx'], '--', 
                    color=colors[i], alpha=0.7, label=f"Real: {res['loiter_duration']}s")
            
    ax3a.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3a.set_xlabel('Time (minutes)', fontsize=12)
    ax3a.set_ylabel('Horizontal Velocity (m/s)', fontsize=12)
    ax3a.set_title('Horizontal Velocity Comparison', fontsize=13, fontweight='bold')
    ax3a.grid(True, alpha=0.3)
    ax3a.legend(fontsize=8, loc='best')
    
    # Vertical velocity
    for i, res in enumerate(successful_deliveries):
        t = np.linspace(0, res['Total_Time'], N)
        ax3b.plot(t/60, res['Vz'], linewidth=2.5, color=colors[i], label=f"Opt")
        if 'sim' in res and res['sim'] is not None:
            ax3b.plot(np.array(res['sim']['t'])/60, res['sim']['vz'], '--', 
                    color=colors[i], alpha=0.7, label=f"Real")
            
    ax3b.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3b.set_xlabel('Time (minutes)', fontsize=12)
    ax3b.set_ylabel('Vertical Velocity (m/s)', fontsize=12)
    ax3b.set_title('Vertical Velocity Comparison', fontsize=13, fontweight='bold')
    ax3b.grid(True, alpha=0.3)
    
    fig3.suptitle('Velocity Profiles', fontsize=15, fontweight='bold')
    fig3.tight_layout()
    
    # ============================================================================
    # WINDOW 4: Energy Consumption
    # ============================================================================
    fig4 = plt.figure(figsize=(12, 6))
    ax4 = fig4.add_subplot(111)
    
    for i, res in enumerate(successful_deliveries):
        t = np.linspace(0, res['Total_Time'], N+1)
        ax4.plot(t/60, res['E'], linewidth=2.5, color=colors[i],
                label=f"Loiter: {res['loiter_duration']}s")
    
    ax4.axhline(y=energy_total_available, color='r', linestyle='--', 
               label='Battery Limit', linewidth=3, alpha=0.7)
    ax4.fill_between([0, ax4.get_xlim()[1]], 0, energy_total_available, 
                     color='green', alpha=0.1, label='Available Energy')
    
    ax4.set_xlabel('Time (minutes)', fontsize=13)
    ax4.set_ylabel('Energy Consumed (J)', fontsize=13)
    ax4.set_title('Energy Consumption Over Mission', fontsize=15, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10, loc='best')
    fig4.tight_layout()
    
    # ============================================================================
    # WINDOW 5: Performance Metrics
    # ============================================================================
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))
    
    loiter_times = [r['loiter_duration'] for r in successful_deliveries]
    delivery_dists = [r['Delivery_Distance']/1000 for r in successful_deliveries]
    mission_times = [r['Total_Time']/60 for r in successful_deliveries]
    
    # Delivery distance vs loiter time
    ax5a.plot(loiter_times, delivery_dists, 'o-', linewidth=3, markersize=10, 
             color='darkgreen', markerfacecolor='lightgreen', markeredgewidth=2)
    ax5a.set_xlabel('Loiter Duration (seconds)', fontsize=12)
    ax5a.set_ylabel('Delivery Distance (km)', fontsize=12)
    ax5a.set_title('Delivery Range vs Loiter Time', fontsize=13, fontweight='bold')
    ax5a.grid(True, alpha=0.3)
    
    # Mission time vs loiter time
    ax5b.plot(loiter_times, mission_times, 's-', linewidth=3, markersize=10, 
             color='darkorange', markerfacecolor='lightyellow', markeredgewidth=2)
    ax5b.set_xlabel('Loiter Duration (seconds)', fontsize=12)
    ax5b.set_ylabel('Total Mission Time (minutes)', fontsize=12)
    ax5b.set_title('Mission Duration vs Loiter Time', fontsize=13, fontweight='bold')
    ax5b.grid(True, alpha=0.3)
    
    fig5.suptitle('Performance Metrics', fontsize=15, fontweight='bold')
    fig5.tight_layout()
    
    return [fig1, fig2, fig3, fig4, fig5]

# ------------------------------------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DRONE DELIVERY MISSION TRAJECTORY OPTIMIZATION")
    print("Mission Profile: Climb -> Cruise -> Loiter -> Descent -> Climb -> Cruise -> Descent")
    print("="*80)
    
    # Find optimal wind conditions
    print("\nAnalyzing wind conditions...")
    best_wind_h, max_v_wind = find_optimal_wind_altitude()
    print(f"Optimal Wind Altitude: {best_wind_h:.1f} m (Wind Speed: {max_v_wind:.2f} m/s)")
    
    # Solve baseline one-way mission for comparison
    baseline_result = solve_baseline_mission()
    
    # Solve delivery missions with different loiter durations
    loiter_durations = loiter_duration_bounds  # seconds
    delivery_results = []
    
    print("\n" + "="*80)
    print("OPTIMIZING DELIVERY MISSIONS WITH VARIOUS LOITER DURATIONS")
    print("="*80)
    
    for loiter_time in loiter_durations:
        result = solve_delivery_mission(loiter_duration=loiter_time, verbose=True)
        if result['success']:
            print(f"Running real drone simulation for loiter {loiter_time}s...")
            result['sim'] = run_real_drone_simulation(result)
        delivery_results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("MISSION SUMMARY")
    print("="*80)
    
    if baseline_result['success']:
        print(f"\nOne-Way Mission (Baseline):")
        print(f"  Maximum Range: {baseline_result['range']/1000:.2f} km")
        print(f"  Mission Time: {baseline_result['Total_Time']/60:.1f} minutes")
    
    print(f"\nDelivery Missions (Round-Trip with Loiter):")
    for res in delivery_results:
        if res['success']:
            print(f"  Loiter {res['loiter_duration']}s: Delivery Distance = {res['Delivery_Distance']/1000:.2f} km, "
                  f"Time = {res['Total_Time']/60:.1f} min")
    
    # Visualization
    print("\nGenerating comparison plots...")
    plot_mission_comparison(baseline_result, delivery_results)
    plt.show()
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80 + "\n")
