import numpy as np
import matplotlib.pyplot as plt
import argparse

# Wind model parameters (copied from the main project)
v_wind_ref = 25
k_wind = 3.74 * 2  # Adjusted for more realistic wind profile (original was too aggressive)
h_max_wind = 300
omega_wind = [0.5, 0.9, 2.42, -5.1, 0.98, 1.1]
c_wind = [-0.82230367, 0.97959362, -0.227143, -0.49432766, 0.76617972, 0.1597865]


def get_wind_soar_multiplier(h):
    val_at_h_max = 0.999
    log_term1 = np.log(val_at_h_max / h_max_wind)
    log_term2 = np.log(val_at_h_max) / h_max_wind
    return (1 - (np.exp(h * (log_term1 / h_max_wind)))) * (np.exp(log_term2 * h))


def get_wind_speed(h):
    """Vectorized wind speed model. h may be scalar or numpy array (meters)."""
    h = np.asarray(h, dtype=float)
    norm_h = h / 5000.0
    exp_term = np.exp(-k_wind * norm_h)
    series_sum = np.zeros_like(norm_h)
    half_size = len(omega_wind) // 2
    for i in range(half_size):
        series_sum += c_wind[i] * np.sin(omega_wind[i] * norm_h)
    for i in range(half_size):
        series_sum += c_wind[i + half_size] * np.cos(omega_wind[i + half_size] * norm_h)
    return get_wind_soar_multiplier(h) * exp_term * v_wind_ref * series_sum / len(omega_wind)


def plot_wind_profile(min_h=0, max_h=4000, n_points=400, save_path=None):
    alts = np.linspace(min_h, max_h, n_points)
    speeds = get_wind_speed(alts)

    # Best wind altitude (max raw wind speed)
    idx = np.nanargmax(speeds)
    best_h = float(alts[idx])
    best_v = float(speeds[idx])

    # Landscape / horizontal layout: altitude on x-axis to reduce vertical paper usage
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(alts, speeds, color='tab:blue', linewidth=2)
    # Mark best altitude as a vertical line
    ax.axvline(x=best_h, color='red', linestyle='--', label=f'Best Wind Alt ({best_h:.0f} m)')
    # Mark the point of maximum wind on the curve
    ax.scatter([best_h], [best_v], color='red', s=30)
    ax.set_title('Wind Speed vs Altitude (Horizontal Layout)')
    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.grid(True)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")

    plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Plot wind speed vs altitude using the project wind model')
    p.add_argument('--min', dest='min_h', type=float, default=0, help='Minimum altitude (m)')
    p.add_argument('--max', dest='max_h', type=float, default=4000, help='Maximum altitude (m)')
    p.add_argument('--points', dest='n_points', type=int, default=400, help='Number of altitude samples')
    p.add_argument('--save', dest='save', type=str, default=None, help='Optional path to save PNG')
    args = p.parse_args()

    plot_wind_profile(min_h=args.min_h, max_h=args.max_h, n_points=args.n_points, save_path=args.save)
