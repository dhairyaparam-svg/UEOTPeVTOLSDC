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
#c_wind = [-0.77230367, 0.59959362, -0.20242143, -0.49432766, 0.76617972, 0.13477865] 


# Solver Parameters
N = 100 
solver_type = 'ipopt' 
max_iter = 1000

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
    alts = np.linspace(optimal_altitude_bounds[0], optimal_altitude_bounds[1], 100)
    wind_speeds = [float(get_wind_speed(a)) for a in alts]
    # Find the altitude with maximum raw wind speed (Highest positive value)
    idx = np.argmax(wind_speeds)
    return alts[idx], wind_speeds[idx]

def get_instantaneous_power(vx_ground, vz_climb, h, thrust_input, v_wind_override=None):
    rho = calculate_air_density(h)
    # Allow overriding wind for "favourable direction" (tailwind) calculations
    if v_wind_override is not None:
        v_wind = v_wind_override
    else:
        v_wind = get_wind_speed(h)
    vx_air = vx_ground + v_wind 
    v_total_air = ca.sqrt(vx_air**2 + vz_climb**2)
    efficiency_m3 = eff_intercept_slope_m3 * rho
    efficiency_m1 = (eff_density_intercept - (eff_density_slope * rho))
    actual_thrust = (1 - aerodynamic_thrust_loss) * thrust_input * (rho / rho_0)
    thrust_coeff = actual_thrust / (max_throttle * weight)
    efficiency_term = (efficiency_m3 - (efficiency_m1 * thrust_coeff))
    efficiency_term = ca.fmax(efficiency_term, 0.001) 
    tilt_factor = ca.fmin(ca.fmax(weight / actual_thrust, -0.99), 0.99) 
    sqrt_tilt = ca.sqrt(1 - tilt_factor**2)
    drag_coeff = ((vz_climb**2/v_total_air**2) * full_face_drag_coefficient) + \
                 ((vx_air**2/v_total_air**2) * (zero_angle_drag_coefficient_cd0 + \
                 (drag_slope_k1 * (ca.arcsin(sqrt_tilt) ** 2))))
    drag_power_numerator = (v_total_air**2) * sqrt_tilt * rho * drag_coeff * rotor_area
    downforce_coeff = zero_angle_downforce_coefficient + (downforce_coefficient_slope * (ca.arcsin(sqrt_tilt)))
    downforce = rotor_area * rho * downforce_coeff * (vx_air**2)
    return (drag_power_numerator / efficiency_term) + ((weight + (downforce / gravity)) / efficiency_term) + (weight * gravity * vz_climb)

# ------------------------------------------------------------------------------------------------------------
def calculate_unoptimized_range(target_h, target_vx):
    v_z_climb = 5.0
    v_wind_h = float(get_wind_speed(target_h))
    p_climb = float(get_instantaneous_power(0, v_z_climb, target_h/2, weight * 1.05, v_wind_override=0)) 
    time_climb = target_h / v_z_climb
    energy_climb = p_climb * time_climb
    
    v_z_land = -3.0
    p_land = float(get_instantaneous_power(0, v_z_land, target_h/2, weight * 0.9, v_wind_override=0)) 
    time_land = target_h / abs(v_z_land)
    energy_land = p_land * time_land
    
    energy_left = energy_total_available - energy_climb - energy_land
    if energy_left <= 0: return None
    
    v_tailwind = -abs(v_wind_h)
    p_cruise = float(get_instantaneous_power(target_vx, 0, target_h, weight, v_wind_override=v_tailwind))
    if p_cruise <= 0: return None

    time_cruise = max(0, energy_left / p_cruise)
    total_range = target_vx * time_cruise
    
    # Optional Trajectory Generation
    total_time = time_climb + time_cruise + time_land
    t_fine = np.linspace(0, total_time, N+1)
    x_tr = np.zeros(N+1); h_tr = np.zeros(N+1); e_tr = np.zeros(N+1)
    vx_tr = np.zeros(N); vz_tr = np.zeros(N)
    
    # Analytical Integration (Simplistic)
    for i in range(N):
        t = t_fine[i]
        if t < time_climb:
            vx_tr[i] = 0; vz_tr[i] = v_z_climb; p_now = p_climb
        elif t < (time_climb + time_cruise):
            vx_tr[i] = target_vx; vz_tr[i] = 0; p_now = p_cruise
        else:
            vx_tr[i] = 0; vz_tr[i] = v_z_land; p_now = p_land
            
        x_tr[i+1] = x_tr[i] + vx_tr[i] * (total_time/N)
        h_tr[i+1] = max(0, h_tr[i] + vz_tr[i] * (total_time/N))
        e_tr[i+1] = e_tr[i] + p_now * (total_time/N)
    
    return {
        'range': total_range, 'Total_Time': total_time,
        'X': x_tr, 'H': h_tr, 'E': e_tr, 'Vx': vx_tr, 'Vz': vz_tr
    }

# ------------------------------------------------------------------------------------------------------------
# BASELINE OPTIMIZER (The one "Optimal" Mission)
# ------------------------------------------------------------------------------------------------------------

def solve_baseline_mission():
    """
    Optimizes a mission with NO cruise constraints (Original Problem).
    Finds the absolute maximum range possible within all physical bounds.
    """
    print("Running Global Optimization (The Gold Star)...")
    opti = ca.Opti()
    X = opti.variable(N+1); H = opti.variable(N+1); E = opti.variable(N+1)
    Vx = opti.variable(N); Vz = opti.variable(N); Thrust = opti.variable(N)
    Total_Time = opti.variable()
    dt = Total_Time / N

    opti.minimize(-X[N]) 

    # Initial/Final Conditions
    opti.subject_to(X[0] == 0); opti.subject_to(H[0] == 0); opti.subject_to(E[0] == 0)
    opti.subject_to(H[N] == 0)
    opti.subject_to(Vx[0] == 0); opti.subject_to(Vx[N-1] == 0)

    # Dynamics
    for k in range(N):
        opti.subject_to(X[k+1] == X[k] + Vx[k] * dt)
        opti.subject_to(H[k+1] == H[k] + Vz[k] * dt)
        P_inst = get_instantaneous_power(Vx[k], Vz[k], H[k], Thrust[k])
        opti.subject_to(E[k+1] == E[k] + P_inst * dt)

    # General Bounds
    opti.subject_to(E[N] <= energy_total_available)
    opti.subject_to(opti.bounded(0, H, 4500))
    opti.subject_to(opti.bounded(0, Vx, 35))
    opti.subject_to(opti.bounded(rate_of_climb_bounds[0], Vz, rate_of_climb_bounds[1]))
    opti.subject_to(opti.bounded(weight * min_throttle_adj, Thrust, weight * max_throttle))
    opti.subject_to(opti.bounded(60, Total_Time, 7200))

    # Initial Guesses
    opti.set_initial(Total_Time, 1200)
    opti.set_initial(X, np.linspace(0, 10000, N+1))
    opti.set_initial(H, 200)
    opti.set_initial(Vx, 15)
    opti.set_initial(Thrust, weight)

    opti.solver(solver_type, {"ipopt.print_level": 0, "ipopt.max_iter": max_iter, "expand": True})
    
    try:
        sol = opti.solve()
        # Return full trajectory data for plotting
        return {
            'X': sol.value(X), 'H': sol.value(H), 'E': sol.value(E),
            'Vx': sol.value(Vx), 'Vz': sol.value(Vz), 'Thrust': sol.value(Thrust),
            'Total_Time': sol.value(Total_Time), 
            'range': sol.value(X[N]),
            'vx_mean': np.mean(sol.value(Vx)[int(0.2*N):int(0.8*N)]),
            'h_mean': np.mean(sol.value(H)[int(0.2*N):int(0.8*N)])
        }
    except:
        return None

# ------------------------------------------------------------------------------------------------------------
# SIMULATION & COMPARISON
# ------------------------------------------------------------------------------------------------------------

print("Finding optimal wind altitude...")
best_wind_h, max_v_wind = find_optimal_wind_altitude()
print(f"Optimal Wind Altitude: {best_wind_h:.1f} m (Wind: {max_v_wind:.2f} m/s)")

# Run Baseline Optimization (Global Optimum)
baseline_res = solve_baseline_mission()
if baseline_res:
    baseline_range = baseline_res['range']
    baseline_vx = baseline_res['vx_mean']
    baseline_h = baseline_res['h_mean']
    print(f"Global Optimum - Range: {baseline_range/1000:.2f} km, Alt: {baseline_h:.0f}m, Speed: {baseline_vx:.1f}m/s")
else:
    baseline_range = 0
    print("Global Optimization Failed.")

# Setup unoptimized sweep (Denser resolution as it's now fast)
altitudes_to_test = sorted([100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, int(best_wind_h)])
speeds_to_test = np.linspace(5, 25, 15) 

results = {}

unoptimized_best_trajectories = [] # Store best trajectories for plotting
z_surface = np.zeros((len(altitudes_to_test), len(speeds_to_test)))

print("\nStarting Unoptimized Analytical Sweep (Favourable Direction)...")
for i, alt in enumerate(altitudes_to_test):
    results[alt] = []
    current_speeds = []
    best_range_at_alt = 0
    best_traj_at_alt = None
    print(f"Calculating for Altitude: {alt:.0f} m")
    for j, speed in enumerate(speeds_to_test):
        res = calculate_unoptimized_range(alt, speed)
        
        if res and res['range'] > 0:
            rng = res['range']
            results[alt].append(rng)
            current_speeds.append(speed)
            z_surface[i, j] = rng / 1000.0 # Store in km
            
            if rng > best_range_at_alt:
                best_range_at_alt = rng
                best_traj_at_alt = res
            
    if results[alt]:
        print(f"  Max range at this alt: {max(results[alt])/1000:.2f} km")
        if best_traj_at_alt:
            best_traj_at_alt['alt_label'] = f"Unopt: {alt:.0f} m"
            unoptimized_best_trajectories.append(best_traj_at_alt)
    else:
        print(f"  Unreachable with vertical climb/land.")
    
    results[f"speeds_{alt}"] = current_speeds

# Add Global Optimum to comparison trajectories
if baseline_res:
    baseline_res['alt_label'] = 'GLOBAL OPTIMUM (SOLVER)'
    unoptimized_best_trajectories.append(baseline_res)

# ------------------------------------------------------------------------------------------------------------
# PLOTTING COMPARISON
# ------------------------------------------------------------------------------------------------------------
plt.style.use('seaborn-v0_8')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Wind Profile
h_range = np.linspace(0, 4000, 200)
v_range = [float(get_wind_speed(h)) for h in h_range]
ax1.plot(v_range, h_range, color='blue', linewidth=2)
ax1.axhline(y=best_wind_h, color='red', linestyle='--', label=f'Best Wind Alt ({best_wind_h:.0f}m)')
ax1.set_title("Wind Velocity Profile")
ax1.set_xlabel("Wind Speed (m/s)")
ax1.set_ylabel("Altitude (m)")
ax1.grid(True); ax1.legend()

# Plot 2: Range Comparison
for i, alt in enumerate(sorted(altitudes_to_test)):
    if alt not in results or not results[alt]: continue
    alt_speeds = results[f"speeds_{alt}"]
    label = f"{alt:.0f} m" + (" (Optimal Wind)" if alt == int(best_wind_h) else "")
    ax2.plot(alt_speeds, np.array(results[alt])/1000, 
             marker='.', markersize=3, label=label, linewidth=1.2, alpha=0.7)

# Add Global Optimum result as a Gold Star
if baseline_range > 0:
    ax2.plot(baseline_vx, baseline_range/1000, marker='*', markersize=15, color='gold', 
             markeredgecolor='black', label='Global Optimum', linestyle='None')
    ax2.annotate(f'Global Opt\n({baseline_range/1000:.1f}km)', 
                 xy=(baseline_vx, baseline_range/1000), 
                 xytext=(15, 15), textcoords='offset points', fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

ax2.set_title("Range Comparison: Unoptimized Missions vs Optimized")
ax2.set_xlabel("Cruise Ground Speed (m/s)")
ax2.set_ylabel("Max Range (km)")
ax2.legend(fontsize='small', ncol=2)
ax2.grid(True)

plt.tight_layout()
# ------------------------------------------------------------------------------------------------------------
# PLOT 3: OPTIMAL PATH STATES (In a separate window)
# ------------------------------------------------------------------------------------------------------------
if baseline_res:
    fig_opt, labs = plt.subplots(2, 2, figsize=(14, 10))
    fig_opt.canvas.manager.set_window_title('Optimal Path States')
    
    t_grid = np.linspace(0, baseline_res['Total_Time'], N+1)
    t_grid_n = np.linspace(0, baseline_res['Total_Time'], N)
    
    # 1. Altitude vs Distance (The Path)
    labs[0, 0].plot(baseline_res['X'], baseline_res['H'], 'b-', linewidth=2)
    labs[0, 0].set_title("Optimal Path (Altitude vs Distance)")
    labs[0, 0].set_xlabel("Distance (m)")
    labs[0, 0].set_ylabel("Altitude (m)")
    labs[0, 0].grid(True)
    
    # 2. Horizontal Velocity vs Time (Vx)
    labs[0, 1].plot(t_grid_n, baseline_res['Vx'], 'b-', label='Vx (Ground)')
    labs[0, 1].set_title("Horizontal Velocity vs Time")
    labs[0, 1].set_xlabel("Time (s)")
    labs[0, 1].set_ylabel("Velocity (m/s)")
    labs[0, 1].legend()
    labs[0, 1].grid(True)
    
    # 3. Energy Consumption
    labs[1, 0].plot(t_grid, baseline_res['E'], 'g-', linewidth=2)
    labs[1, 0].set_title("Energy Consumption vs Time")
    labs[1, 0].set_xlabel("Time (s)")
    labs[1, 0].set_ylabel("Energy (J)")
    labs[1, 0].grid(True)
    
    # 4. Vertical Velocity vs Time (Vz)
    labs[1, 1].plot(t_grid_n, baseline_res['Vz'], 'r-', linewidth=2, label='Vz (Climb Rate)')
    labs[1, 1].set_title("Vertical Velocity vs Time")
    labs[1, 1].set_xlabel("Time (s)")
    labs[1, 1].set_ylabel("Velocity (m/s)")
    labs[1, 1].legend()
    labs[1, 1].grid(True)
    
    plt.tight_layout()

# ------------------------------------------------------------------------------------------------------------
# PLOT 3: UNOPTIMIZED PATH STATES (In a separate window)
# ------------------------------------------------------------------------------------------------------------
if unoptimized_best_trajectories:
    fig_unopt, labs = plt.subplots(2, 2, figsize=(14, 10))
    fig_unopt.canvas.manager.set_window_title('Unoptimized Path States (Old View)')
    
    for traj in unoptimized_best_trajectories:
        if "GLOBAL OPTIMUM" in traj['alt_label']: continue
        label = traj['alt_label']
        total_time = traj['Total_Time']
        t_grid = np.linspace(0, total_time, N+1)
        t_grid_n = np.linspace(0, total_time, N)
        
        # Plotting
        labs[0, 0].plot(traj['X'], traj['H'], label=label)
        labs[0, 1].plot(t_grid_n, traj['Vx'], label=label)
        labs[1, 0].plot(t_grid, traj['E'], label=label)
        labs[1, 1].plot(t_grid_n, traj['Vz'], label=label)

    # Styling
    for i in range(2):
        for j in range(2): labs[i, j].grid(True)
    labs[0, 0].set_title("Paths (Alt vs Dist)"); labs[0, 1].set_title("Vx vs Time")
    labs[1, 0].set_title("Energy vs Time"); labs[1, 1].set_title("Vz vs Time")
    labs[0, 0].legend(fontsize='xx-small', ncol=2)
    plt.tight_layout()

# ------------------------------------------------------------------------------------------------------------
# PLOT 4: COMPARATIVE PATH STATES (Unoptimized vs Optimal)
# ------------------------------------------------------------------------------------------------------------
if unoptimized_best_trajectories:
    fig_comp, labs = plt.subplots(2, 2, figsize=(14, 10))
    fig_comp.canvas.manager.set_window_title('Comparative Path States (Unopt vs Opt)')
    
    for traj in unoptimized_best_trajectories:
        label = traj['alt_label']
        total_time = traj['Total_Time']
        t_grid = np.linspace(0, total_time, N+1)
        t_grid_n = np.linspace(0, total_time, N)
        
        # Trajectory Plotting
        is_opt = "GLOBAL OPTIMUM" in traj['alt_label']
        lw = 3 if is_opt else 1.2
        ls = '-' if is_opt else '--'
        alpha = 1.0 if is_opt else 0.7
        color = 'gold' if is_opt else None
        
        labs[0, 0].plot(traj['X'], traj['H'], label=label, linewidth=lw, linestyle=ls, alpha=alpha, color=color)
        labs[0, 1].plot(t_grid_n, traj['Vx'], label=label, linewidth=lw, linestyle=ls, alpha=alpha, color=color)
        labs[1, 0].plot(t_grid, traj['E'], label=label, linewidth=lw, linestyle=ls, alpha=alpha, color=color)
        labs[1, 1].plot(t_grid_n, traj['Vz'], label=label, linewidth=lw, linestyle=ls, alpha=alpha, color=color)

    # Styling
    for i in range(2):
        for j in range(2): labs[i, j].grid(True)
    labs[0, 0].set_title("Comparison (Alt vs Dist)"); labs[0, 1].set_title("Vx Comparison")
    labs[1, 0].set_title("Energy Comparison"); labs[1, 1].set_title("Vz Comparison")
    labs[0, 0].legend(fontsize='xx-small', ncol=2)
    plt.tight_layout()

# ------------------------------------------------------------------------------------------------------------
# PLOT 5: 3D RANGE ENVELOPE (Range vs Altitude vs Speed)
# ------------------------------------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D
fig_3d = plt.figure(figsize=(12, 8))
ax3d = fig_3d.add_subplot(111, projection='3d')
fig_3d.canvas.manager.set_window_title('3D Mission Envelope')

# Create meshgrid for plotting (Ensure X and Y are consistent with Z dimensions)
S, A = np.meshgrid(speeds_to_test, altitudes_to_test)

# Plot surface
surf = ax3d.plot_surface(S, A, z_surface, cmap='viridis', alpha=0.8, edgecolor='none')
fig_3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5, label='Max Range (km)')

# Add Global Optimum point if available
if baseline_range > 0:
    ax3d.scatter(baseline_vx, baseline_h, baseline_range/1000, 
                 color='gold', s=200, marker='*', edgecolor='black', label='Global Optimum', zorder=100)

ax3d.set_title("Range Envelope: Speed vs Altitude vs Range")
ax3d.set_xlabel("Cruise Speed (m/s)")
ax3d.set_ylabel("Altitude (m)")
ax3d.set_zlabel("Max Range (km)")
ax3d.legend()

plt.show()
