import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# This version implements WIND-AWARE optimization where:
# - Outbound journey flies HIGH to get tailwind assistance
# - Return journey flies LOW to avoid headwind

# [All the constants and helper functions remain the same as before]
# ... (keeping the file concise, I'll add a comment block indicating where to insert the previous code)

# NOTE: Insert all constants, pre-calculations, and helper functions from the previous file here
# For brevity, I'm showing only the NEW/MODIFIED solve_delivery_mission function

def solve_delivery_mission_wind_aware(loiter_duration=60, verbose=True):
    """
    Wind-aware delivery mission optimization that actively exploits wind conditions:
    
    STRATEGY:
    - Outbound: Fly at HIGHER altitude to benefit from tailwind (wind is negative/tailwind)
    - Return: Fly at LOWER altitude to avoid headwind
    - Energy cost of climbing is balanced against wind benefits
    
    Phase 1: Climb HIGH (outbound)
    Phase 2: Cruise HIGH with tailwind (outbound)
    Phase 3: Descent to delivery point
    Phase 4: Loiter at delivery point
    Phase 5: Climb to MODERATE altitude (return)
    Phase 6: Cruise LOW to avoid headwind (return)
    Phase 7: Descent to home
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Wind-Aware Delivery Mission Optimization (Loiter: {loiter_duration}s)")
        print(f"Strategy: HIGH altitude outbound (tailwind), LOW altitude return (avoid headwind)")
        print(f"{'='*80}")
    
    opti = ca.Opti()
    
    # State variables
    X = opti.variable(N+1)
    H = opti.variable(N+1)
    E = opti.variable(N+1)
    
    # Control variables
    Vx = opti.variable(N)
    Vz = opti.variable(N)
    Thrust = opti.variable(N)
    
    # Mission parameters
    Total_Time = opti.variable()
    Delivery_Distance = opti.variable()
    
    # NEW: Separate altitude variables for outbound and return cruise
    H_outbound_cruise = opti.variable()  # Altitude during outbound cruise
    H_return_cruise = opti.variable()    # Altitude during return cruise
    
    dt = Total_Time / N
    
    # Objective: Maximize delivery distance
    opti.minimize(-Delivery_Distance)
    
    # Initial & Final Conditions
    opti.subject_to(X[0] == 0)
    opti.subject_to(H[0] == 0)
    opti.subject_to(E[0] == 0)
    opti.subject_to(Vx[0] == 0)
    opti.subject_to(X[N] == 0)
    opti.subject_to(H[N] == 0)
    opti.subject_to(Vx[N-1] == 0)
    
    # Dynamics
    for k in range(N):
        opti.subject_to(X[k+1] == X[k] + Vx[k] * dt)
        opti.subject_to(H[k+1] == H[k] + Vz[k] * dt)
        P_inst = get_instantaneous_power(Vx[k], Vz[k], H[k], Thrust[k])
        opti.subject_to(E[k+1] == E[k] + P_inst * dt)
    
    # General Bounds
    opti.subject_to(E[N] <= energy_total_available)
    opti.subject_to(opti.bounded(0, H, 4500))
    opti.subject_to(opti.bounded(-25, Vx, 25))
    opti.subject_to(opti.bounded(rate_of_climb_bounds[0], Vz, rate_of_climb_bounds[1]))
    opti.subject_to(opti.bounded(weight * min_throttle_adj, Thrust, weight * max_throttle))
    opti.subject_to(opti.bounded(120, Total_Time, 7200))
    opti.subject_to(opti.bounded(1000, Delivery_Distance, 15000))
    
    # Altitude bounds for cruise phases
    opti.subject_to(opti.bounded(200, H_outbound_cruise, 2000))   # Higher for tailwind
    opti.subject_to(opti.bounded(100, H_return_cruise, 800))      # Lower to avoid headwind
    
    # KEY CONSTRAINT: Outbound cruise should be HIGHER than return cruise
    #opti.subject_to(H_outbound_cruise >= H_return_cruise + 100)  # At least 100m difference
    
    # Mission Structure
    mid_point = N // 2
    outbound_cruise_start = int(N * 0.15)
    outbound_cruise_end = int(N * 0.35)
    return_cruise_start = int(N * 0.65)
    return_cruise_end = int(N * 0.85)
    
    # Midpoint constraints
    opti.subject_to(X[mid_point] >= Delivery_Distance * 0.85)
    opti.subject_to(X[mid_point] <= Delivery_Distance * 1.05)
    
    # Position bounds
    for k in range(N+1):
        opti.subject_to(X[k] >= -100)
        opti.subject_to(X[k] <= Delivery_Distance * 1.1)
    
    # Velocity direction constraints
    for k in range(5, mid_point - 15):
        opti.subject_to(Vx[k] >= -3)  # Outbound: mostly forward
    for k in range(mid_point + 15, N - 5):
        opti.subject_to(Vx[k] <= 3)   # Return: mostly backward
    
    # NEW: Enforce cruise altitudes during cruise phases
    for k in range(outbound_cruise_start, outbound_cruise_end):
        # Outbound cruise: maintain high altitude
        opti.subject_to(H[k] >= H_outbound_cruise - 50)
        opti.subject_to(H[k] <= H_outbound_cruise + 50)
    
    for k in range(return_cruise_start, return_cruise_end):
        # Return cruise: maintain low altitude
        opti.subject_to(H[k] >= H_return_cruise - 50)
        opti.subject_to(H[k] <= H_return_cruise + 50)
    
    # Loiter Phase
    loiter_window = max(5, int(N * 0.08))
    for k in range(mid_point - loiter_window, mid_point + loiter_window):
        if 0 <= k < N:
            opti.subject_to(H[k] <= loiter_altitude_bounds[1])
            opti.subject_to(opti.bounded(-loiter_speed, Vx[k], loiter_speed))
    
    opti.subject_to(opti.bounded(-1.5, Vx[mid_point], 1.5))
    opti.subject_to(H[mid_point] <= loiter_altitude_bounds[1] + 10)
    
    # Initial Guesses
    opti.set_initial(Total_Time, 900)
    opti.set_initial(Delivery_Distance, 3500)
    opti.set_initial(H_outbound_cruise, 800)   # Start with high outbound
    opti.set_initial(H_return_cruise, 300)     # Start with low return
    
    # Position guess
    x_guess = np.concatenate([
        np.linspace(0, 3500, mid_point + 1),
        np.linspace(3500, 0, N - mid_point)
    ])
    opti.set_initial(X, x_guess)
    
    # Altitude guess with asymmetric cruise altitudes
    n_sixth = N // 6
    h_guess = np.concatenate([
        np.linspace(10, 800, n_sixth),           # Climb HIGH outbound
        np.ones(n_sixth) * 800,                  # Cruise HIGH outbound
        np.linspace(800, 50, n_sixth),           # Descend to loiter
        np.ones(n_sixth) * 50,                   # Loiter
        np.linspace(50, 300, n_sixth),           # Climb to LOW altitude return
        np.linspace(300, 10, N + 1 - 5*n_sixth) # Cruise LOW return and descend
    ])
    h_guess = np.maximum(h_guess, 5.0)
    opti.set_initial(H, h_guess)
    
    # Velocity guess
    vx_guess = np.concatenate([
        np.linspace(0, 10, n_sixth),
        np.ones(2*n_sixth) * 10,
        np.linspace(10, 0, n_sixth),
        np.linspace(0, -10, n_sixth),
        np.ones(N - 5*n_sixth) * -10
    ])
    opti.set_initial(Vx, vx_guess)
    opti.set_initial(Vz, 0)
    opti.set_initial(Thrust, weight * 1.1)
    
    # Solver options
    solver_opts = {
        "expand": True,
        "ipopt.print_level": 3 if verbose else 0,
        "ipopt.max_iter": max_iter,
        "ipopt.tol": 1e-5,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 15,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.start_with_resto": "yes",
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
            'H_outbound_cruise': sol.value(H_outbound_cruise),
            'H_return_cruise': sol.value(H_return_cruise),
            'loiter_duration': loiter_duration,
            'success': True
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"[SUCCESS] WIND-AWARE OPTIMIZATION SUCCESSFUL")
            print(f"{'='*80}")
            print(f"Delivery Distance: {result['Delivery_Distance']/1000:.2f} km")
            print(f"Total Mission Time: {result['Total_Time']/60:.1f} minutes")
            print(f"Energy Used: {result['E'][-1]:.1f} J / {energy_total_available:.1f} J ({100*result['E'][-1]/energy_total_available:.1f}%)")
            print(f"Outbound Cruise Altitude: {result['H_outbound_cruise']:.1f} m (HIGH for tailwind)")
            print(f"Return Cruise Altitude: {result['H_return_cruise']:.1f} m (LOW to avoid headwind)")
            print(f"Altitude Difference: {result['H_outbound_cruise'] - result['H_return_cruise']:.1f} m")
            
            # Calculate wind speeds at cruise altitudes
            wind_out = float(get_wind_speed(result['H_outbound_cruise']))
            wind_ret = float(get_wind_speed(result['H_return_cruise']))
            print(f"Wind at outbound altitude: {wind_out:.2f} m/s (negative = tailwind)")
            print(f"Wind at return altitude: {wind_ret:.2f} m/s (negative = tailwind)")
            print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"\n[FAILED] Optimization failed: {str(e)}\n")
        return {'success': False, 'loiter_duration': loiter_duration}

# This is a TEMPLATE showing the key changes needed
# The full implementation would include all the original code plus these modifications
