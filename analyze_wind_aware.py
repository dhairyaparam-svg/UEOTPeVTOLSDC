import numpy as np
import sys
sys.path.insert(0, 'c:/Users/IITGN/Downloads')

# Import the optimizer
from Drone_delivery_mission_optimizer import solve_delivery_mission, get_wind_speed

# Run one optimization
print("Running wind-aware optimization...")
result = solve_delivery_mission(loiter_duration=30, verbose=False)

if result['success']:
    # Analyze the trajectory
    mid_point = 75  # N // 2
    outbound_cruise_start = int(150 * 0.15)
    outbound_cruise_end = int(150 * 0.35)
    return_cruise_start = int(150 * 0.65)
    return_cruise_end = int(150 * 0.85)
    
    # Calculate average altitudes during cruise phases
    outbound_alt = np.mean(result['H'][outbound_cruise_start:outbound_cruise_end])
    return_alt = np.mean(result['H'][return_cruise_start:return_cruise_end])
    
    # Calculate wind speeds at those altitudes
    wind_out = float(get_wind_speed(outbound_alt))
    wind_ret = float(get_wind_speed(return_alt))
    
    print("\n" + "="*80)
    print("WIND-AWARE TRAJECTORY ANALYSIS")
    print("="*80)
    print(f"Delivery Distance: {result['Delivery_Distance']/1000:.2f} km")
    print(f"Total Mission Time: {result['Total_Time']/60:.1f} minutes")
    print(f"\nOUTBOUND CRUISE:")
    print(f"  Average Altitude: {outbound_alt:.1f} m")
    print(f"  Wind Speed: {wind_out:.2f} m/s (negative = tailwind)")
    print(f"\nRETURN CRUISE:")
    print(f"  Average Altitude: {return_alt:.1f} m")
    print(f"  Wind Speed: {wind_ret:.2f} m/s (negative = tailwind)")
    print(f"\nWIND EXPLOITATION:")
    print(f"  Altitude Difference: {outbound_alt - return_alt:.1f} m")
    print(f"  Wind Benefit: {abs(wind_out) - abs(wind_ret):.2f} m/s")
    
    if outbound_alt > return_alt:
        print(f"  SUCCESS: Outbound flies HIGHER than return (as intended)")
    else:
        print(f"  WARNING: Outbound NOT higher than return")
    
    print("="*80)
else:
    print("Optimization failed!")
