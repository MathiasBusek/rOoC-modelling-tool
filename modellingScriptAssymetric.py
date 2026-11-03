import argparse
import copy
import csv
from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
import struct
import sys
from pathlib import Path

import matplotlib
if "--headless" in sys.argv:
    matplotlib.use("Agg")
else:
    matplotlib.use("QtAgg")
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Fixed parameters
default_viscosity_mpa_s = 0.8
viscosity = default_viscosity_mpa_s * 1e-3  # in Pa*s
density = 1000  # in kg/m3
gravity = 9.81  # in m/s2
default_cap_force = 1  # capillary pressure in Pa
WSS_ref = 7.4  # dyne/cm2
default_reference_wss_pa = 0.74  # Pa at the configured reference flow
default_reference_flow_uL_s = 1.0

start_angle = 0.0
phaseDiff = np.pi / 2

default_tilt = 30  # in degrees
# time step for simulation
default_time_step = 0.001
# default total simulation time
default_stop_time = 300
default_stl_wetting_volume_bins = 300
default_stl_wetting_tilt_step_deg = 0.15
# Limit explicit STL updates to a small transported-volume fraction.  The GUI
# time step remains the output interval; only the internal integration substeps.
max_stl_transport_fraction_per_substep = 0.01
max_stl_integration_substeps = 128
# The STL equilibrium split is cached on a finite angular grid.  Smooth the
# resulting target before differentiating it into a redistribution flow.
stl_equilibrium_filter_time = 0.05

plot_colors = ["r", "g", "c", "m", "y", "k"]
default_mode_settings = {
    "symmetric": {
        "title": "symmetric",
        "speed1": 0.3,
        "speed2": 0.3,
        "slow_start": None,
        "slow_duration": None,
    },
    "assymmetric": {
        "title": "assymmetric",
        "speed1": 0.3,
        "speed2": 4.0,
        "slow_start": 0.0,
        "slow_duration": 0.3,
    },
    "single_reservoir": {
        "title": "single reservoir",
        "speed1": 0.3,
        "speed2": None,
        "slow_start": None,
        "slow_duration": None,
    },
    "single_reservoir_assymmetric": {
        "title": "single reservoir (assymmetric tilt)",
        "speed1": 0.3,
        "speed2": 4.0,
        "slow_start": 0.0,
        "slow_duration": 0.3,
    },
    "simple_tilting": {
        "title": "simple tilting: two cylindrical reservoirs, one channel",
        "speed1": 0.3,
        "speed2": None,
        "slow_start": None,
        "slow_duration": None,
    },
    "four_reservoir_tilting": {
        "title": "four-reservoir tilting: cylindrical ring with four channels",
        "speed1": 0.3,
        "speed2": None,
        "slow_start": None,
        "slow_duration": None,
    },
    "nested_four_reservoir_loops": {
        "title": "four reservoirs: nested inner and outer recirculating loops",
        "speed1": 0.3,
        "speed2": None,
        "slow_start": None,
        "slow_duration": None,
    },
}
default_reservoir_settings = {
    "width": 4.0,
    "height": 6.0,
    "length": 18.0,
    "v_fill_ml": 0.3,
}
default_geometry_settings = {
    "primary": {
        "shape": "rectangular",
        "width": 1.2,
        "height": 0.75,
        "length": 18.0,
        "radius": 0.25,
    },
    "secondary": {
        "shape": "rectangular",
        "width": 1.2,
        "height": 0.75,
        "length": 18.0,
        "radius": 0.25,
    },
}
progress_update_every = 2000
stl_axis_names = ("x", "y", "z")


def assymetricRotation(t, speed_low, speed_fast, slow_start, slow_duration, phase_diff):
    if slow_start is None or slow_duration is None:
        raise ValueError("Assymmetric mode requires slow_start and slow_duration.")
    if slow_start < 0 or slow_start > 0.95:
        raise ValueError("Slow traversal start must be between 0 and 0.95.")
    if slow_duration <= 0 or slow_duration > 1:
        raise ValueError("Slow traversal duration must be between 0 and 1.")
    if slow_start + slow_duration > 1:
        raise ValueError("Slow traversal start plus duration must not exceed 1.")

    fast_before = slow_start * 60 / speed_fast
    slow_segment = slow_duration * 60 / speed_low
    fast_after = (1 - slow_start - slow_duration) * 60 / speed_fast
    full_period = fast_before + slow_segment + fast_after
    current_period = math.floor(t / full_period)
    cycle_position = t - current_period * full_period

    if cycle_position < fast_before:
        phase = cycle_position / 60 * speed_fast
    elif cycle_position < fast_before + slow_segment:
        phase = slow_start + (cycle_position - fast_before) / 60 * speed_low
    else:
        phase = (
            slow_start
            + slow_duration
            + (cycle_position - fast_before - slow_segment) / 60 * speed_fast
        )

    return np.sin(2 * np.pi * phase + phase_diff)


def rotation_signal(t, mode, speed1, phase_diff, speed2=None, slow_start=None, slow_duration=None):
    if mode in ("symmetric", "single_reservoir", "simple_tilting", "four_reservoir_tilting", "nested_four_reservoir_loops"):
        return np.sin(2 * np.pi * speed1 * t / 60 + phase_diff)

    if mode in ("assymmetric", "single_reservoir_assymmetric"):
        if speed2 is None or slow_start is None or slow_duration is None:
            raise ValueError("Assymmetric mode requires speed2, slow_start, and slow_duration.")
        return assymetricRotation(t, speed1, speed2, slow_start, slow_duration, phase_diff)

    raise ValueError(f"Unsupported mode: {mode}")


def get_actuation_period_seconds(selected_case):
    mode = selected_case["mode"]
    speed1 = selected_case["speed1"]

    if mode in ("symmetric", "single_reservoir", "simple_tilting", "four_reservoir_tilting", "nested_four_reservoir_loops"):
        return 60.0 / speed1

    if mode in ("assymmetric", "single_reservoir_assymmetric"):
        speed2 = selected_case["speed2"]
        slow_start = selected_case["slow_start"]
        slow_duration = selected_case["slow_duration"]
        if speed2 is None or slow_start is None or slow_duration is None:
            raise ValueError(
                "Assymmetric mode requires speed2, slow_start, and slow_duration."
            )
        fast_before = slow_start * 60.0 / speed2
        slow_segment = slow_duration * 60.0 / speed1
        fast_after = (1.0 - slow_start - slow_duration) * 60.0 / speed2
        return fast_before + slow_segment + fast_after

    raise ValueError(f"Unsupported mode: {mode}")


def get_last_full_period_mask(time_points, selected_case):
    if len(time_points) == 0:
        raise ValueError("Time vector is empty.")

    period_seconds = get_actuation_period_seconds(selected_case)
    end_time = float(time_points[-1])
    start_time = end_time - period_seconds
    if start_time <= float(time_points[0]):
        return np.ones_like(time_points, dtype=bool), period_seconds, float(time_points[0]), end_time

    mask = time_points >= start_time
    if np.count_nonzero(mask) < 2:
        return np.ones_like(time_points, dtype=bool), period_seconds, float(time_points[0]), end_time
    return mask, period_seconds, start_time, end_time


def compute_real_fourier_coeffs_from_discrete_set(y_dataset, term_count):
    """Compute real Fourier coefficients (a_n, b_n) for evenly spaced data."""
    values = np.asarray(y_dataset, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Fourier fitting requires at least two samples.")
    if not 0 <= int(term_count) < len(values):
        raise ValueError("Fourier term count must be smaller than the number of samples.")

    sample_count = len(values)
    sample_index = np.arange(sample_count, dtype=float)
    coefficients = []
    for harmonic in range(int(term_count) + 1):
        angle = 2.0 * np.pi * harmonic * sample_index / sample_count
        coefficients.append(
            (
                (2.0 / sample_count) * np.sum(values * np.cos(angle)),
                (2.0 / sample_count) * np.sum(values * np.sin(angle)),
            )
        )
    return np.asarray(coefficients, dtype=float)


def fit_func_by_fourier_series_with_real_coeffs(sample_positions, coefficients, period_length):
    """Evaluate a real Fourier series at positions within one sampled period."""
    if period_length <= 0.0:
        raise ValueError("Fourier period length must be positive.")

    positions = np.asarray(sample_positions, dtype=float)
    result = np.full_like(positions, coefficients[0, 0] / 2.0, dtype=float)
    for harmonic in range(1, len(coefficients)):
        angle = 2.0 * np.pi * harmonic * positions / period_length
        result += (
            coefficients[harmonic, 0] * np.cos(angle)
            + coefficients[harmonic, 1] * np.sin(angle)
        )
    return result


def get_last_complete_period_samples(time_points, values, selected_case):
    """Resample the final whole actuation cycle for a Fourier-series fit."""
    time_values = np.asarray(time_points, dtype=float)
    data_values = np.asarray(values, dtype=float)
    if len(time_values) != len(data_values) or len(time_values) < 2:
        raise ValueError("Fourier fitting requires matching time and data arrays.")

    period_seconds = get_actuation_period_seconds(selected_case)
    simulation_start = float(time_values[0])
    elapsed_time = float(time_values[-1] - simulation_start)
    complete_period_count = int(math.floor((elapsed_time + 1e-9) / period_seconds))
    if complete_period_count < 1:
        raise ValueError(
            f"The simulation must cover one full actuation period (T={period_seconds:.2f} s)."
        )

    end_time = simulation_start + complete_period_count * period_seconds
    start_time = end_time - period_seconds
    median_time_step = float(np.median(np.diff(time_values)))
    sample_count = max(4, int(round(period_seconds / median_time_step)))
    fit_times = np.linspace(start_time, end_time, sample_count, endpoint=False)
    fit_values = np.interp(fit_times, time_values, data_values)
    return fit_times, fit_values, period_seconds, start_time, end_time


def get_realtime_animation_time_values(plot_data, fps, frame_count=None):
    """Sample the final actuation cycle at real-time fps or a chosen frame count."""
    time_points = np.asarray(plot_data["time_points"], dtype=float)
    if len(time_points) < 2:
        raise ValueError("Animation requires at least two simulation time points.")

    period_seconds = get_actuation_period_seconds(plot_data["selected_case"])
    simulation_start = float(time_points[0])
    elapsed_time = float(time_points[-1] - simulation_start)
    if elapsed_time + 1e-9 < period_seconds:
        raise ValueError(
            f"Animation requires one complete actuation period (T={period_seconds:.2f} s). "
            "Increase the computation time."
        )

    # Match the flow-average window: use the last period-length interval of
    # the completed simulation, regardless of where the period began.
    end_time = float(time_points[-1])
    start_time = end_time - period_seconds
    if frame_count is None:
        # endpoint=False gives one frame per playback interval, making the
        # saved GIF duration equal the simulated actuation-period duration.
        frame_count = max(2, int(round(period_seconds * float(fps))))
    return np.linspace(start_time, end_time, int(frame_count), endpoint=False)


def build_fourier_period_fit(time_points, values, selected_case, term_count):
    """Fit one complete actuation cycle and return display-ready Fourier data."""
    fit_times, fit_values, period_seconds, start_time, end_time = (
        get_last_complete_period_samples(time_points, values, selected_case)
    )
    maximum_terms = (len(fit_values) - 1) // 2
    if not 1 <= int(term_count) <= maximum_terms:
        raise ValueError(
            f"Fourier terms must be between 1 and {maximum_terms} for this period."
        )

    coefficients = compute_real_fourier_coeffs_from_discrete_set(fit_values, term_count)
    fitted_values = fit_func_by_fourier_series_with_real_coeffs(
        fit_times - start_time,
        coefficients,
        period_seconds,
    )
    return {
        "time_points": fit_times,
        "fitted_values": fitted_values,
        "coefficients": coefficients,
        "period_seconds": period_seconds,
        "start_time": start_time,
        "end_time": end_time,
    }


def format_fourier_series_text(coefficients, period_seconds):
    """Format Fourier coefficients in the real-series form used by the fit code."""
    coefficient_array = np.asarray(coefficients, dtype=float)
    if coefficient_array.ndim != 2 or coefficient_array.shape[1] != 2:
        raise ValueError("Fourier coefficients must be an array of (a_n, b_n) pairs.")

    def coefficient_text(value):
        return f"{0.0 if abs(value) < 5e-13 else value:.4f}"

    period_text = f"{float(period_seconds):.2f}"
    lines = [coefficient_text(coefficient_array[0, 0] / 2.0)]
    for harmonic in range(1, len(coefficient_array)):
        cosine_coefficient, sine_coefficient = coefficient_array[harmonic]
        lines.append(
            f"+ cos(2*pi*{harmonic}*t/{period_text})*{coefficient_text(cosine_coefficient)} "
            f"+sin(2*pi*{harmonic}*t/{period_text})*{coefficient_text(sine_coefficient)}"
        )
    return "\n".join(lines)


def calculate_last_period_mean(time_points, values, selected_case):
    mask, period_seconds, start_time, end_time = get_last_full_period_mask(
        time_points,
        selected_case,
    )
    return (
        float(np.mean(values[mask])),
        period_seconds,
        start_time,
        end_time,
    )


def calculate_last_period_mean_absolute(time_points, values, selected_case):
    mask, period_seconds, start_time, end_time = get_last_full_period_mask(
        time_points,
        selected_case,
    )
    return (
        float(np.mean(np.abs(values[mask]))),
        period_seconds,
        start_time,
        end_time,
    )


def calculate_last_period_max_absolute(time_points, values, selected_case):
    mask, period_seconds, start_time, end_time = get_last_full_period_mask(
        time_points,
        selected_case,
    )
    return (
        float(np.max(np.abs(values[mask]))),
        period_seconds,
        start_time,
        end_time,
    )


def build_reservoir_geometry(width, height, length, v_fill_ml):
    return {
        "type": "rectangular",
        "width": width,
        "height": height,
        "length": length,
        "v_fill_ml": v_fill_ml,
        "v_fill_ul": v_fill_ml * 1000,
        "volume_ul_max": width * height * length,
        "left_port": {"x": 0.0, "y": 0.0},
        "right_port": {"x": length, "y": 0.0},
    }


def build_cylindrical_reservoir_geometry(radius, height, v_fill_ml):
    """Build one of two identical upright cylinders with a centered bottom port."""
    capacity_ul = math.pi * float(radius) ** 2 * float(height)
    return {
        "type": "cylindrical",
        "radius": float(radius),
        "width": 2.0 * float(radius),
        "height": float(height),
        "length": 2.0 * float(radius),
        "v_fill_ml": float(v_fill_ml),
        "v_fill_ul": float(v_fill_ml) * 1000.0,
        "volume_ul_max": capacity_ul,
        "left_port": {"x": 0.0, "y": 0.0},
        "right_port": {"x": 0.0, "y": 0.0},
        "_free_surface_cache": {},
    }


def get_cylindrical_free_surface_intercept(
    volume_ul,
    tilt_length,
    tilt_width,
    reservoir_geometry,
):
    """Return the liquid-plane height at the centre of an upright cylinder."""
    if volume_ul <= 0.0:
        return -float(reservoir_geometry["height"])

    radius = float(reservoir_geometry["radius"])
    height = float(reservoir_geometry["height"])
    capacity = float(reservoir_geometry["volume_ul_max"])
    target_volume = float(np.clip(volume_ul, 0.0, capacity))
    volume_step = max(capacity / 400.0, 0.05)
    tilt_step = math.radians(0.15)
    cache_key = (
        int(round(target_volume / volume_step)),
        int(round(float(tilt_length) / tilt_step)),
        int(round(float(tilt_width) / tilt_step)),
    )
    cached_intercept = reservoir_geometry["_free_surface_cache"].get(cache_key)
    if cached_intercept is not None:
        return cached_intercept

    quantized_volume = min(capacity, cache_key[0] * volume_step)
    slope = math.hypot(
        math.tan(cache_key[1] * tilt_step),
        math.tan(cache_key[2] * tilt_step),
    )
    radial_position = np.linspace(-radius, radius, 161)
    chord_width = 2.0 * np.sqrt(np.maximum(radius**2 - radial_position**2, 0.0))

    def enclosed_volume(intercept):
        liquid_height = np.clip(intercept - slope * radial_position, 0.0, height)
        return float(np.trapezoid(liquid_height * chord_width, radial_position))

    lower_bound = -slope * radius
    upper_bound = height + slope * radius
    for _ in range(42):
        midpoint = 0.5 * (lower_bound + upper_bound)
        if enclosed_volume(midpoint) < quantized_volume:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    intercept = 0.5 * (lower_bound + upper_bound)
    reservoir_geometry["_free_surface_cache"][cache_key] = intercept
    return intercept


def get_cylindrical_port_head(volume_ul, tilt_length, tilt_width, reservoir_geometry):
    """Hydrostatic head at a cylinder's centered bottom port under tilt."""
    return max(
        0.0,
        get_cylindrical_free_surface_intercept(
            volume_ul,
            tilt_length,
            tilt_width,
            reservoir_geometry,
        ),
    )


def get_cylindrical_gate_head(
    volume_ul,
    tilt_length,
    tilt_width,
    reservoir_geometry,
    local_x,
    local_y,
):
    """Return the hydrostatic head at an offset bottom gate of a cylinder.

    The cylinder's free surface is evaluated in its local coordinate system.
    Ring-channel gates are placed at the cardinal points of the cylinder wall,
    so each reservoir can independently feed or receive through two channels.
    """
    intercept = get_cylindrical_free_surface_intercept(
        volume_ul,
        tilt_length,
        tilt_width,
        reservoir_geometry,
    )
    local_surface_height = (
        intercept
        - math.tan(tilt_length) * float(local_x)
        - math.tan(tilt_width) * float(local_y)
    )
    return max(0.0, local_surface_height)


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    return 0.5 * (
        np.dot(x_coords, np.roll(y_coords, -1))
        - np.dot(y_coords, np.roll(x_coords, -1))
    )


def ensure_counterclockwise(points):
    polygon = np.asarray(points, dtype=float)
    if polygon_area(polygon) < 0:
        polygon = polygon[::-1]
    return polygon


def simplify_polygon(points, tolerance=1e-6):
    polygon = np.asarray(points, dtype=float)
    if len(polygon) == 0:
        return polygon

    simplified = []
    for point in polygon:
        if (not simplified) or np.linalg.norm(point - simplified[-1]) > tolerance:
            simplified.append(point)

    if len(simplified) > 1 and np.linalg.norm(simplified[0] - simplified[-1]) <= tolerance:
        simplified.pop()

    if len(simplified) < 3:
        return np.asarray(simplified, dtype=float)

    cleaned = []
    point_count = len(simplified)
    for index in range(point_count):
        previous_point = np.asarray(simplified[index - 1], dtype=float)
        point = np.asarray(simplified[index], dtype=float)
        next_point = np.asarray(simplified[(index + 1) % point_count], dtype=float)
        vector_1 = point - previous_point
        vector_2 = next_point - point
        norm_1 = np.linalg.norm(vector_1)
        norm_2 = np.linalg.norm(vector_2)
        if (norm_1 <= tolerance) or (norm_2 <= tolerance):
            continue
        cross_value = abs(vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0])
        if cross_value / (norm_1 * norm_2) <= tolerance:
            continue
        cleaned.append(point)

    return ensure_counterclockwise(np.asarray(cleaned, dtype=float))


def convex_hull(points, tolerance=1e-9):
    unique_points = []
    seen = set()
    for point in np.asarray(points, dtype=float):
        key = tuple(np.round(point / tolerance).astype(int))
        if key not in seen:
            seen.add(key)
            unique_points.append((float(point[0]), float(point[1])))

    if len(unique_points) <= 1:
        return np.asarray(unique_points, dtype=float)

    sorted_points = sorted(unique_points)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull_points = lower[:-1] + upper[:-1]
    return ensure_counterclockwise(np.asarray(hull_points, dtype=float))


def points_in_triangle_grid(x_grid, y_grid, triangle, tolerance=1e-9):
    point_a = triangle[0]
    point_b = triangle[1]
    point_c = triangle[2]
    denominator = (
        (point_b[1] - point_c[1]) * (point_a[0] - point_c[0])
        + (point_c[0] - point_b[0]) * (point_a[1] - point_c[1])
    )
    if abs(denominator) <= tolerance:
        return np.zeros_like(x_grid, dtype=bool)

    alpha = (
        (point_b[1] - point_c[1]) * (x_grid - point_c[0])
        + (point_c[0] - point_b[0]) * (y_grid - point_c[1])
    ) / denominator
    beta = (
        (point_c[1] - point_a[1]) * (x_grid - point_c[0])
        + (point_a[0] - point_c[0]) * (y_grid - point_c[1])
    ) / denominator
    gamma = 1.0 - alpha - beta
    return (
        (alpha >= -tolerance)
        & (beta >= -tolerance)
        & (gamma >= -tolerance)
    )


def rasterize_projected_triangles(projected_triangles, max_resolution=500):
    all_points = projected_triangles.reshape(-1, 2)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)
    raw_span_x = max(float(max_x - min_x), 1e-6)
    raw_span_y = max(float(max_y - min_y), 1e-6)
    padding_x = max(raw_span_x * 0.02, 1e-3)
    padding_y = max(raw_span_y * 0.02, 1e-3)
    min_x -= padding_x
    max_x += padding_x
    min_y -= padding_y
    max_y += padding_y
    span_x = max(float(max_x - min_x), 1e-6)
    span_y = max(float(max_y - min_y), 1e-6)

    if span_x >= span_y:
        grid_width = max_resolution
        grid_height = max(120, int(round(max_resolution * span_y / span_x)))
    else:
        grid_height = max_resolution
        grid_width = max(120, int(round(max_resolution * span_x / span_y)))

    x_coords = np.linspace(min_x, max_x, grid_width)
    y_coords = np.linspace(min_y, max_y, grid_height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    occupancy = np.zeros((grid_height, grid_width), dtype=bool)

    x_step = span_x / max(grid_width - 1, 1)
    y_step = span_y / max(grid_height - 1, 1)

    for triangle in projected_triangles:
        triangle = np.asarray(triangle, dtype=float)
        tri_min_x = float(np.min(triangle[:, 0]))
        tri_max_x = float(np.max(triangle[:, 0]))
        tri_min_y = float(np.min(triangle[:, 1]))
        tri_max_y = float(np.max(triangle[:, 1]))

        x_start = max(0, int(math.floor((tri_min_x - min_x) / x_step)) - 1)
        x_end = min(grid_width - 1, int(math.ceil((tri_max_x - min_x) / x_step)) + 1)
        y_start = max(0, int(math.floor((tri_min_y - min_y) / y_step)) - 1)
        y_end = min(grid_height - 1, int(math.ceil((tri_max_y - min_y) / y_step)) + 1)

        occupancy[y_start : y_end + 1, x_start : x_end + 1] |= points_in_triangle_grid(
            x_grid[y_start : y_end + 1, x_start : x_end + 1],
            y_grid[y_start : y_end + 1, x_start : x_end + 1],
            triangle,
        )

    return x_coords, y_coords, occupancy


def extract_projected_boundary(projected_triangles):
    x_coords, y_coords, occupancy = rasterize_projected_triangles(projected_triangles)
    if not np.any(occupancy):
        raise ValueError("Could not rasterize the projected STL footprint.")

    contour_figure, contour_axis = plt.subplots()
    contour_set = contour_axis.contour(
        x_coords,
        y_coords,
        occupancy.astype(float),
        levels=[0.5],
    )
    plt.close(contour_figure)

    contour_segments = contour_set.allsegs[0] if contour_set.allsegs else []
    if not contour_segments:
        raise ValueError("Could not extract the top-view boundary from the STL projection.")

    boundary = max(contour_segments, key=lambda segment: abs(polygon_area(np.asarray(segment))))
    return simplify_polygon(np.asarray(boundary, dtype=float), tolerance=1e-4)


def extract_projected_boundaries(projected_triangles):
    x_coords, y_coords, occupancy = rasterize_projected_triangles(projected_triangles)
    if not np.any(occupancy):
        raise ValueError("Could not rasterize the projected STL footprint.")

    contour_figure, contour_axis = plt.subplots()
    contour_set = contour_axis.contour(
        x_coords,
        y_coords,
        occupancy.astype(float),
        levels=[0.5],
    )
    plt.close(contour_figure)

    contour_segments = contour_set.allsegs[0] if contour_set.allsegs else []
    if not contour_segments:
        raise ValueError("Could not extract the top-view boundary from the STL projection.")

    return [
        simplify_polygon(np.asarray(segment, dtype=float), tolerance=1e-4)
        for segment in contour_segments
        if len(segment) >= 3
    ]


def clip_polygon_below_line(points, slope, intercept, tolerance=1e-9):
    def signed_distance(point):
        return intercept - (point[1] + slope * point[0])

    if len(points) == 0:
        return np.empty((0, 2), dtype=float)

    clipped = []
    point_count = len(points)
    for index in range(point_count):
        start = points[index]
        end = points[(index + 1) % point_count]
        start_inside = signed_distance(start) >= -tolerance
        end_inside = signed_distance(end) >= -tolerance

        if start_inside and end_inside:
            clipped.append(end)
            continue

        denominator = (end[1] - start[1]) + slope * (end[0] - start[0])
        intersection = None
        if abs(denominator) > tolerance:
            t_value = (intercept - start[1] - slope * start[0]) / denominator
            t_value = min(1.0, max(0.0, t_value))
            intersection = start + t_value * (end - start)

        if start_inside and (not end_inside):
            if intersection is not None:
                clipped.append(intersection)
        elif (not start_inside) and end_inside:
            if intersection is not None:
                clipped.append(intersection)
            clipped.append(end)

    if len(clipped) < 3:
        return np.empty((0, 2), dtype=float)

    return simplify_polygon(np.asarray(clipped, dtype=float))


def snap_point_to_polygon(point, polygon):
    point = np.asarray(point, dtype=float)
    best_point = np.asarray(polygon[0], dtype=float)
    best_distance = float("inf")
    point_count = len(polygon)

    for index in range(point_count):
        start = np.asarray(polygon[index], dtype=float)
        end = np.asarray(polygon[(index + 1) % point_count], dtype=float)
        edge = end - start
        edge_norm_sq = float(np.dot(edge, edge))
        if np.isclose(edge_norm_sq, 0.0):
            candidate = start
        else:
            t_value = np.dot(point - start, edge) / edge_norm_sq
            t_value = min(1.0, max(0.0, t_value))
            candidate = start + t_value * edge
        distance = float(np.linalg.norm(point - candidate))
        if distance < best_distance:
            best_distance = distance
            best_point = candidate

    return best_point


def build_profile_reservoir_geometry(
    profile_points,
    width,
    v_fill_ml,
    left_port,
    right_port,
    source_path=None,
    extrusion_axis="auto",
    profile_axes=None,
    top_view_points=None,
    top_view_triangles=None,
    shell_triangles=None,
    left_port_world=None,
    right_port_world=None,
    axis_mins=None,
    axis_maxs=None,
    axis_names=None,
):
    profile_polygon = simplify_polygon(profile_points)
    if len(profile_polygon) < 3:
        raise ValueError("Reservoir profile must contain at least three distinct points.")

    min_x = float(np.min(profile_polygon[:, 0]))
    min_y = float(np.min(profile_polygon[:, 1]))
    normalized_polygon = profile_polygon - np.array([min_x, min_y])
    area_mm2 = abs(polygon_area(normalized_polygon))
    if area_mm2 <= 0:
        raise ValueError("Reservoir profile area must be positive.")

    imported_mesh = (
        shell_triangles is not None
        and source_path not in (None, "rectangular-default")
    )
    if imported_mesh:
        # STL ports are selected in the rotated top view.  Do not project them
        # back onto the legacy cross-section, which may no longer align after
        # an X/Y rotation.
        left_port_array = np.asarray(left_port, dtype=float)
        right_port_array = np.asarray(right_port, dtype=float)
    else:
        left_port_array = snap_point_to_polygon(
            np.asarray(left_port, dtype=float) - np.array([min_x, min_y]),
            normalized_polygon,
        )
        right_port_array = snap_point_to_polygon(
            np.asarray(right_port, dtype=float) - np.array([min_x, min_y]),
            normalized_polygon,
        )
    left_port_data = {"x": float(left_port_array[0]), "y": float(left_port_array[1])}
    right_port_data = {"x": float(right_port_array[0]), "y": float(right_port_array[1])}
    if not imported_mesh and left_port_data["x"] > right_port_data["x"]:
        left_port_data, right_port_data = right_port_data, left_port_data

    profile_bounds = {
        "min_x": 0.0,
        "max_x": float(np.max(normalized_polygon[:, 0])),
        "min_y": 0.0,
        "max_y": float(np.max(normalized_polygon[:, 1])),
    }
    if top_view_points is None:
        normalized_top_view = np.asarray(
            [
                [0.0, 0.0],
                [profile_bounds["max_x"], 0.0],
                [profile_bounds["max_x"], float(width)],
                [0.0, float(width)],
            ],
            dtype=float,
        )
    else:
        top_view_polygon = simplify_polygon(top_view_points)
        if len(top_view_polygon) < 3:
            raise ValueError("Reservoir top view must contain at least three distinct points.")
        top_min = np.min(top_view_polygon, axis=0)
        normalized_top_view = top_view_polygon - top_min
    extrusion_axis_index = None
    if isinstance(extrusion_axis, str) and extrusion_axis in stl_axis_names:
        extrusion_axis_index = stl_axis_names.index(extrusion_axis)
    elif isinstance(extrusion_axis, int):
        extrusion_axis_index = extrusion_axis

    left_port_width = float(width) / 2
    right_port_width = float(width) / 2
    if (
        left_port_world is not None
        and right_port_world is not None
        and axis_mins is not None
        and extrusion_axis_index is not None
    ):
        extrusion_axis_name = stl_axis_names[extrusion_axis_index]
        left_port_width = float(left_port_world[extrusion_axis_name]) - float(axis_mins[extrusion_axis_index])
        right_port_width = float(right_port_world[extrusion_axis_name]) - float(axis_mins[extrusion_axis_index])

    geometry = {
        "type": "profile",
        "width": float(width),
        "height": profile_bounds["max_y"],
        "length": profile_bounds["max_x"],
        "profile_points": normalized_polygon,
        "profile_area_mm2": area_mm2,
        "profile_bounds": profile_bounds,
        "profile_offset_x": min_x,
        "profile_offset_y": min_y,
        "v_fill_ml": v_fill_ml,
        "v_fill_ul": v_fill_ml * 1000,
        "volume_ul_max": area_mm2 * width,
        "left_port": left_port_data,
        "right_port": right_port_data,
        "source_path": None if source_path is None else str(source_path),
        "extrusion_axis": extrusion_axis,
        "extrusion_axis_index": extrusion_axis_index,
        "profile_axes": profile_axes,
        "top_view_points": normalized_top_view,
        "top_view_offset_x": float(top_min[0]) if top_view_points is not None else 0.0,
        "top_view_offset_y": float(top_min[1]) if top_view_points is not None else 0.0,
        "top_view_triangles": top_view_triangles,
        "shell_triangles": shell_triangles,
        "left_port_world": left_port_world,
        "right_port_world": right_port_world,
        "left_port_width": left_port_width,
        "right_port_width": right_port_width,
        "axis_mins": axis_mins,
        "axis_maxs": axis_maxs,
        "axis_names": axis_names,
        "_head_cache": {},
        "_physical_head_cache": {},
        "_head_cache_resolution": {
            "volume_ul": max((area_mm2 * width) / 300, 0.1),
            "tilt_length_rad": math.radians(0.15),
            "tilt_width_rad": math.radians(0.15),
        },
    }
    if imported_mesh:
        # The wetting solver integrates this same rotated shell mesh.  Use its
        # dimensions and capacity instead of the pre-rotation profile prism.
        mesh_context = get_stl_footprint_context(geometry)
        mesh_points = np.asarray(shell_triangles, dtype=float).reshape(-1, 3)
        geometry["length"] = float(np.max(mesh_points[:, 0]) - np.min(mesh_points[:, 0]))
        geometry["width"] = float(np.max(mesh_points[:, 1]) - np.min(mesh_points[:, 1]))
        geometry["height"] = float(np.max(mesh_points[:, 2]) - np.min(mesh_points[:, 2]))
        geometry["volume_ul_max"] = float(mesh_context["total_capacity_ul"])
        geometry["mesh_based_dimensions"] = True
        geometry["_head_cache_resolution"]["volume_ul"] = max(
            geometry["volume_ul_max"] / 300, 0.1
        )
    return geometry


def parse_ascii_stl(text):
    vertices = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("vertex"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(vertices) < 3 or len(vertices) % 3 != 0:
        raise ValueError("ASCII STL did not contain a valid triangle list.")

    return np.asarray(vertices, dtype=float).reshape(-1, 3, 3)


def load_stl_triangles(stl_path):
    stl_file = Path(stl_path)
    raw_data = stl_file.read_bytes()
    if len(raw_data) >= 84:
        triangle_count = struct.unpack_from("<I", raw_data, 80)[0]
        expected_size = 84 + triangle_count * 50
        if expected_size == len(raw_data):
            triangles = []
            offset = 84
            for _ in range(triangle_count):
                unpacked = struct.unpack_from("<12fH", raw_data, offset)
                triangle = np.asarray(unpacked[3:12], dtype=float).reshape(3, 3)
                triangles.append(triangle)
                offset += 50
            if triangles:
                return np.asarray(triangles, dtype=float)

    return parse_ascii_stl(raw_data.decode("utf-8"))


def get_default_stl_extrusion_axis():
    # Preserve the imported STL coordinate system by default:
    # x = longitudinal direction, y = extruded width, z = height.
    return stl_axis_names.index("y")


def intersect_triangle_with_plane(triangle, axis_index, plane_coordinate, tolerance):
    intersections = []
    for edge_index in range(3):
        start = triangle[edge_index]
        end = triangle[(edge_index + 1) % 3]
        start_distance = start[axis_index] - plane_coordinate
        end_distance = end[axis_index] - plane_coordinate
        start_on_plane = abs(start_distance) <= tolerance
        end_on_plane = abs(end_distance) <= tolerance

        if start_on_plane and end_on_plane:
            intersections.extend((start, end))
            continue
        if start_on_plane:
            intersections.append(start)
        if end_on_plane:
            intersections.append(end)
        if start_distance * end_distance < 0:
            fraction = (plane_coordinate - start[axis_index]) / (
                end[axis_index] - start[axis_index]
            )
            intersections.append(start + fraction * (end - start))

    unique_points = []
    for point in intersections:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in unique_points):
            unique_points.append(np.asarray(point, dtype=float))

    if len(unique_points) < 2:
        return None
    if len(unique_points) == 2:
        return np.asarray(unique_points, dtype=float)

    max_distance = -1.0
    best_pair = None
    for index, point_a in enumerate(unique_points):
        for point_b in unique_points[index + 1 :]:
            distance = float(np.linalg.norm(point_a - point_b))
            if distance > max_distance:
                max_distance = distance
                best_pair = (point_a, point_b)
    return np.asarray(best_pair, dtype=float)


def quantize_point(point, tolerance):
    return tuple(np.round(np.asarray(point, dtype=float) / tolerance).astype(int))


def build_profile_loops(segments, tolerance):
    node_lookup = {}
    node_points = []
    edges = []

    def get_node_id(point):
        key = quantize_point(point, tolerance)
        if key not in node_lookup:
            node_lookup[key] = len(node_points)
            node_points.append(np.asarray(point, dtype=float))
        return node_lookup[key]

    for segment in segments:
        start_id = get_node_id(segment[0])
        end_id = get_node_id(segment[1])
        if start_id != end_id:
            edges.append((start_id, end_id))

    adjacency = {node_id: [] for node_id in range(len(node_points))}
    for edge_index, (start_id, end_id) in enumerate(edges):
        adjacency[start_id].append(edge_index)
        adjacency[end_id].append(edge_index)

    unused_edges = set(range(len(edges)))
    loops = []
    while unused_edges:
        current_edge = unused_edges.pop()
        start_id, next_id = edges[current_edge]
        loop_node_ids = [start_id, next_id]
        previous_edge = current_edge
        current_node = next_id

        while current_node != start_id:
            candidate_edges = [
                edge_index
                for edge_index in adjacency[current_node]
                if edge_index in unused_edges
            ]
            if not candidate_edges:
                loop_node_ids = []
                break

            next_edge = candidate_edges[0]
            unused_edges.remove(next_edge)
            edge_start, edge_end = edges[next_edge]
            current_node = edge_end if edge_start == current_node else edge_start
            loop_node_ids.append(current_node)
            previous_edge = next_edge

        if len(loop_node_ids) < 4:
            continue

        loop_points = np.asarray([node_points[node_id] for node_id in loop_node_ids[:-1]], dtype=float)
        simplified_loop = simplify_polygon(loop_points, tolerance=tolerance * 10)
        if len(simplified_loop) >= 3 and abs(polygon_area(simplified_loop)) > tolerance:
            loops.append(simplified_loop)

    return loops


def load_profile_from_stl(stl_path, extrusion_axis=None):
    triangles = load_stl_triangles(stl_path)
    if len(triangles) == 0:
        raise ValueError("STL file did not contain any triangles.")

    vertices = triangles.reshape(-1, 3)
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    spans = maxs - mins
    axis_index = get_default_stl_extrusion_axis() if extrusion_axis is None else extrusion_axis
    extrusion_width = float(spans[axis_index])
    if extrusion_width <= 0:
        raise ValueError("Selected extrusion axis has zero thickness.")

    slice_coordinate = float((mins[axis_index] + maxs[axis_index]) / 2)
    tolerance = max(float(np.max(spans)), 1.0) * 1e-6
    remaining_axes = [axis for axis in range(3) if axis != axis_index]
    length_axis_index = remaining_axes[0]
    height_axis_index = remaining_axes[1]

    segments = []
    for triangle in triangles:
        segment = intersect_triangle_with_plane(
            triangle,
            axis_index,
            slice_coordinate,
            tolerance,
        )
        if segment is None:
            continue
        segment_2d = segment[:, [length_axis_index, height_axis_index]]
        if np.linalg.norm(segment_2d[0] - segment_2d[1]) > tolerance:
            segments.append(segment_2d)

    if not segments:
        raise ValueError("Could not extract a 2D cross-section from the STL.")

    loops = build_profile_loops(segments, tolerance=max(extrusion_width, 1.0) * 1e-5)
    projected_profile_triangles = triangles[:, :, [length_axis_index, height_axis_index]]
    projected_profile_polygon = extract_projected_boundary(projected_profile_triangles)
    projected_profile_min = np.min(projected_profile_polygon, axis=0)
    normalized_projected_profile = projected_profile_polygon - projected_profile_min

    normalized_loop = None
    if loops:
        largest_loop = max(loops, key=lambda loop: abs(polygon_area(loop)))
        largest_loop = simplify_polygon(largest_loop)
        bounds_min = np.min(largest_loop, axis=0)
        normalized_loop = largest_loop - bounds_min

    use_projected_profile = normalized_loop is None or len(loops) != 1
    if not use_projected_profile:
        projected_span_x = float(np.max(normalized_projected_profile[:, 0]))
        slice_span_x = float(np.max(normalized_loop[:, 0]))
        if projected_span_x > 0 and slice_span_x < 0.9 * projected_span_x:
            use_projected_profile = True

    normalized_profile = (
        normalized_projected_profile if use_projected_profile else normalized_loop
    )
    projected_top_triangles = triangles[:, :, [length_axis_index, axis_index]]
    top_view_polygon = extract_projected_boundary(projected_top_triangles)
    top_view_min = np.min(top_view_polygon, axis=0)
    top_view_polygon = top_view_polygon - top_view_min
    normalized_top_triangles = projected_top_triangles - top_view_min
    shell_triangles = triangles[:, :, [length_axis_index, axis_index, height_axis_index]]
    shell_min = np.min(shell_triangles.reshape(-1, 3), axis=0)
    normalized_shell_triangles = shell_triangles - shell_min

    return {
        "profile_points": normalized_profile,
        "width": extrusion_width,
        "extrusion_axis_index": axis_index,
        "extrusion_axis_name": stl_axis_names[axis_index],
        "profile_axes": (
            stl_axis_names[length_axis_index],
            stl_axis_names[height_axis_index],
        ),
        "profile_axis_indices": (length_axis_index, height_axis_index),
        "top_view_points": top_view_polygon,
        "top_view_triangles": normalized_top_triangles,
        "shell_triangles": normalized_shell_triangles,
        "top_view_axes": (
            stl_axis_names[length_axis_index],
            stl_axis_names[axis_index],
        ),
        "top_view_axis_indices": (length_axis_index, axis_index),
        "axis_mins": mins.tolist(),
        "axis_maxs": maxs.tolist(),
        "source_path": str(Path(stl_path).resolve()),
    }


def rotate_stl_profile_template_in_xy_plane(profile_template, angle_degrees):
    """Rotate an imported STL about its X/Y bounding-box center.

    The water-level solver works from the transformed top-view and shell mesh,
    so this is a geometry operation rather than a display-only rotation.
    """
    angle_degrees = float(angle_degrees)
    if not -180.0 <= angle_degrees <= 180.0:
        raise ValueError("STL X/Y rotation must be between -180 and 180 degrees.")

    rotated_template = copy.deepcopy(profile_template)
    rotated_template["xy_rotation_degrees"] = angle_degrees
    if abs(angle_degrees) <= 1e-12:
        return rotated_template

    axis_mins = np.asarray(profile_template["axis_mins"], dtype=float)
    axis_maxs = np.asarray(profile_template["axis_maxs"], dtype=float)
    x_axis_index = stl_axis_names.index("x")
    y_axis_index = stl_axis_names.index("y")
    shell_axis_indices = (
        profile_template["profile_axis_indices"][0],
        profile_template["extrusion_axis_index"],
        profile_template["profile_axis_indices"][1],
    )
    shell_x_index = shell_axis_indices.index(x_axis_index)
    shell_y_index = shell_axis_indices.index(y_axis_index)
    shell_values = np.asarray(profile_template["shell_triangles"], dtype=float)
    shell_x = shell_values[..., shell_x_index] + axis_mins[x_axis_index]
    shell_y = shell_values[..., shell_y_index] + axis_mins[y_axis_index]

    center_x = 0.5 * (axis_mins[x_axis_index] + axis_maxs[x_axis_index])
    center_y = 0.5 * (axis_mins[y_axis_index] + axis_maxs[y_axis_index])
    angle_radians = math.radians(angle_degrees)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)

    def rotate_xy(x_values, y_values):
        centered_x = x_values - center_x
        centered_y = y_values - center_y
        return (
            center_x + cosine * centered_x - sine * centered_y,
            center_y + sine * centered_x + cosine * centered_y,
        )

    rotated_shell_x, rotated_shell_y = rotate_xy(shell_x, shell_y)
    rotated_axis_mins = axis_mins.copy()
    rotated_axis_maxs = axis_maxs.copy()
    rotated_axis_mins[x_axis_index] = float(np.min(rotated_shell_x))
    rotated_axis_maxs[x_axis_index] = float(np.max(rotated_shell_x))
    rotated_axis_mins[y_axis_index] = float(np.min(rotated_shell_y))
    rotated_axis_maxs[y_axis_index] = float(np.max(rotated_shell_y))

    def rotate_projected_values(values, world_axis_indices):
        rotated_values = np.asarray(values, dtype=float).copy()
        if x_axis_index not in world_axis_indices or y_axis_index not in world_axis_indices:
            return rotated_values
        x_coordinate_index = world_axis_indices.index(x_axis_index)
        y_coordinate_index = world_axis_indices.index(y_axis_index)
        x_values = rotated_values[..., x_coordinate_index] + axis_mins[x_axis_index]
        y_values = rotated_values[..., y_coordinate_index] + axis_mins[y_axis_index]
        rotated_x, rotated_y = rotate_xy(x_values, y_values)
        rotated_values[..., x_coordinate_index] = rotated_x - rotated_axis_mins[x_axis_index]
        rotated_values[..., y_coordinate_index] = rotated_y - rotated_axis_mins[y_axis_index]
        return rotated_values

    rotated_template["top_view_points"] = rotate_projected_values(
        profile_template["top_view_points"], profile_template["top_view_axis_indices"]
    )
    rotated_template["top_view_triangles"] = rotate_projected_values(
        profile_template["top_view_triangles"], profile_template["top_view_axis_indices"]
    )
    rotated_template["shell_triangles"] = rotate_projected_values(
        profile_template["shell_triangles"], shell_axis_indices
    )
    rotated_template["axis_mins"] = rotated_axis_mins.tolist()
    rotated_template["axis_maxs"] = rotated_axis_maxs.tolist()
    return rotated_template


def solve_profile_free_surface(profile_points, target_area, slope):
    polygon = np.asarray(profile_points, dtype=float)
    full_area = abs(polygon_area(polygon))
    if target_area <= 0:
        return float(np.min(polygon[:, 1] + slope * polygon[:, 0])) - 1.0
    if target_area >= full_area:
        return float(np.max(polygon[:, 1] + slope * polygon[:, 0])) + 1.0

    signed_heights = polygon[:, 1] + slope * polygon[:, 0]
    lower_bound = float(np.min(signed_heights))
    upper_bound = float(np.max(signed_heights))
    for _ in range(60):
        midpoint = 0.5 * (lower_bound + upper_bound)
        clipped = clip_polygon_below_line(polygon, slope, midpoint)
        current_area = abs(polygon_area(clipped)) if len(clipped) >= 3 else 0.0
        if current_area < target_area:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    return 0.5 * (lower_bound + upper_bound)


def integrate_profile_volume(profile_points, width, slope_length, slope_width, intercept, sample_count=21):
    if width <= 0:
        return 0.0

    if abs(slope_width) <= 1e-12:
        clipped = clip_polygon_below_line(profile_points, slope_length, intercept)
        current_area = abs(polygon_area(clipped)) if len(clipped) >= 3 else 0.0
        return current_area * width

    sample_positions = np.linspace(0.0, width, sample_count)
    areas = []
    for width_position in sample_positions:
        shifted_intercept = intercept - slope_width * width_position
        clipped = clip_polygon_below_line(profile_points, slope_length, shifted_intercept)
        areas.append(abs(polygon_area(clipped)) if len(clipped) >= 3 else 0.0)

    return float(np.trapezoid(areas, sample_positions))


def solve_profile_free_surface_3d(reservoir_geometry, target_volume, slope_length, slope_width):
    profile_points = reservoir_geometry["profile_points"]
    full_volume = reservoir_geometry["volume_ul_max"]
    if target_volume <= 0:
        return float(
            np.min(profile_points[:, 1] + slope_length * profile_points[:, 0])
        ) - abs(slope_width) * reservoir_geometry["width"] - 1.0
    if target_volume >= full_volume:
        return float(
            np.max(profile_points[:, 1] + slope_length * profile_points[:, 0])
        ) + abs(slope_width) * reservoir_geometry["width"] + 1.0

    signed_heights = profile_points[:, 1] + slope_length * profile_points[:, 0]
    lower_bound = float(np.min(signed_heights)) - abs(slope_width) * reservoir_geometry["width"]
    upper_bound = float(np.max(signed_heights)) + abs(slope_width) * reservoir_geometry["width"]
    for _ in range(60):
        midpoint = 0.5 * (lower_bound + upper_bound)
        current_volume = integrate_profile_volume(
            profile_points,
            reservoir_geometry["width"],
            slope_length,
            slope_width,
            midpoint,
        )
        if current_volume < target_volume:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    return 0.5 * (lower_bound + upper_bound)


def get_profile_port_heads(V, tilt_length, tilt_width, reservoir_geometry):
    if V <= 0:
        return 0.0, 0.0

    target_volume = np.clip(
        float(V),
        0.0,
        reservoir_geometry["volume_ul_max"],
    )
    cache_resolution = reservoir_geometry["_head_cache_resolution"]
    volume_step = cache_resolution["volume_ul"]
    tilt_length_step = cache_resolution["tilt_length_rad"]
    tilt_width_step = cache_resolution["tilt_width_rad"]
    cache_key = (
        int(round(target_volume / volume_step)),
        int(round(float(tilt_length) / tilt_length_step)),
        int(round(float(tilt_width) / tilt_width_step)),
    )
    cached_heads = reservoir_geometry["_head_cache"].get(cache_key)
    if cached_heads is not None:
        return cached_heads

    quantized_volume = min(
        reservoir_geometry["volume_ul_max"],
        cache_key[0] * volume_step,
    )
    quantized_tilt_length = cache_key[1] * tilt_length_step
    quantized_tilt_width = cache_key[2] * tilt_width_step
    slope_length = math.tan(quantized_tilt_length)
    slope_width = math.tan(quantized_tilt_width)
    intercept = solve_profile_free_surface_3d(
        reservoir_geometry,
        quantized_volume,
        slope_length,
        slope_width,
    )
    left_port = reservoir_geometry["left_port"]
    right_port = reservoir_geometry["right_port"]
    left_port_width = reservoir_geometry["left_port_width"]
    right_port_width = reservoir_geometry["right_port_width"]
    left_head = max(
        0.0,
        float(intercept - slope_length * left_port["x"] - slope_width * left_port_width - left_port["y"]),
    )
    right_head = max(
        0.0,
        float(intercept - slope_length * right_port["x"] - slope_width * right_port_width - right_port["y"]),
    )
    heads = (left_head, right_head)
    reservoir_geometry["_head_cache"][cache_key] = heads
    return heads


def get_profile_port_heads_uncached(V, tilt_length, tilt_width, reservoir_geometry):
    """Solve STL/profile port heads at the exact current volume and tilt."""
    if V <= 0:
        return 0.0, 0.0
    target_volume = np.clip(float(V), 0.0, reservoir_geometry["volume_ul_max"])
    slope_length = math.tan(float(tilt_length))
    slope_width = math.tan(float(tilt_width))
    intercept = solve_profile_free_surface_3d(
        reservoir_geometry,
        target_volume,
        slope_length,
        slope_width,
    )
    left_port = reservoir_geometry["left_port"]
    right_port = reservoir_geometry["right_port"]
    left_head = max(
        0.0,
        float(
            intercept
            - slope_length * left_port["x"]
            - slope_width * reservoir_geometry["left_port_width"]
            - left_port["y"]
        ),
    )
    right_head = max(
        0.0,
        float(
            intercept
            - slope_length * right_port["x"]
            - slope_width * reservoir_geometry["right_port_width"]
            - right_port["y"]
        ),
    )
    return left_head, right_head


def get_port_head(V, tilt, port_name, reservoir_geometry):
    if reservoir_geometry["type"] == "profile":
        left_head, right_head = get_profile_port_heads(V, tilt, 0.0, reservoir_geometry)
        return right_head if port_name == "right" else left_head
    return getHeigth(V, tilt, port_name == "right", reservoir_geometry)


def get_port_separation_along_length(reservoir_geometry):
    if reservoir_geometry["type"] == "profile":
        return abs(
            float(reservoir_geometry["right_port"]["x"])
            - float(reservoir_geometry["left_port"]["x"])
        )
    return float(reservoir_geometry["length"])


def get_oriented_port_heads(
    V,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    use_cached_heads=True,
):
    if uses_stl_footprint_partition(reservoir_geometry):
        # Imported STL reservoirs use the rotated shell mesh rather than the
        # older extruded-profile approximation in both model modes.
        equilibrium = get_stl_footprint_equilibrium(
            V,
            signed_tilt_length,
            signed_tilt_width,
            reservoir_geometry,
        )
        return equilibrium["left_head"], equilibrium["right_head"]

    tilt_length = abs(signed_tilt_length)
    tilt_width = signed_tilt_width
    if reservoir_geometry["type"] == "profile":
        head_solver = get_profile_port_heads if use_cached_heads else get_profile_port_heads_uncached
        left_head, right_head = head_solver(V, tilt_length, tilt_width, reservoir_geometry)
    else:
        left_head = get_port_head(V, tilt_length, "left", reservoir_geometry)
        right_head = get_port_head(V, tilt_length, "right", reservoir_geometry)
    if signed_tilt_length < 0:
        return right_head, left_head
    return left_head, right_head


def get_physical_port_heads(V, signed_tilt_length, signed_tilt_width, reservoir_geometry):
    if reservoir_geometry["type"] == "profile":
        target_volume = np.clip(
            float(V),
            0.0,
            reservoir_geometry["volume_ul_max"],
        )
        cache_resolution = reservoir_geometry["_head_cache_resolution"]
        volume_step = cache_resolution["volume_ul"]
        tilt_length_step = cache_resolution["tilt_length_rad"]
        tilt_width_step = cache_resolution["tilt_width_rad"]
        cache_key = (
            int(round(target_volume / volume_step)),
            int(round(float(signed_tilt_length) / tilt_length_step)),
            int(round(float(signed_tilt_width) / tilt_width_step)),
        )
        cached_heads = reservoir_geometry["_physical_head_cache"].get(cache_key)
        if cached_heads is not None:
            return cached_heads

        quantized_volume = min(
            reservoir_geometry["volume_ul_max"],
            cache_key[0] * volume_step,
        )
        quantized_tilt_length = cache_key[1] * tilt_length_step
        quantized_tilt_width = cache_key[2] * tilt_width_step
        slope_length = math.tan(quantized_tilt_length)
        slope_width = math.tan(quantized_tilt_width)
        intercept = solve_profile_free_surface_3d(
            reservoir_geometry,
            quantized_volume,
            slope_length,
            slope_width,
        )
        left_port = reservoir_geometry["left_port"]
        right_port = reservoir_geometry["right_port"]
        left_head = max(
            0.0,
            float(
                intercept
                - slope_length * left_port["x"]
                - slope_width * reservoir_geometry["left_port_width"]
                - left_port["y"]
            ),
        )
        right_head = max(
            0.0,
            float(
                intercept
                - slope_length * right_port["x"]
                - slope_width * reservoir_geometry["right_port_width"]
                - right_port["y"]
            ),
        )
        heads = (left_head, right_head)
        reservoir_geometry["_physical_head_cache"][cache_key] = heads
        return heads

    tilt_length = abs(signed_tilt_length)
    left_head = get_port_head(V, tilt_length, "left", reservoir_geometry)
    right_head = get_port_head(V, tilt_length, "right", reservoir_geometry)
    if signed_tilt_length < 0:
        return right_head, left_head
    return left_head, right_head


def apply_capillary_cutoff(dh_value, upstream_head, downstream_head, dh_min):
    if dh_value > 0:
        if upstream_head == 0:
            return 0
        if (downstream_head == 0) and (abs(dh_value) < dh_min):
            return 0
        return dh_value

    if downstream_head == 0:
        return 0
    if (upstream_head == 0) and (abs(dh_value) < dh_min):
        return 0
    return dh_value


def limit_single_reservoir_flow_by_volume(
    flow_rate,
    left_volume,
    right_volume,
    compartments,
    time_step,
):
    """Limit one time-step transport to the donor volume and receiver capacity."""
    if compartments.get("uses_footprint_partition", False):
        left_capacity = float(compartments["left_capacity_ul"])
        right_capacity = float(compartments["right_capacity_ul"])
    else:
        left_capacity = float(compartments["left_geometry"]["volume_ul_max"])
        right_capacity = float(compartments["right_geometry"]["volume_ul_max"])

    # Positive Q moves liquid from the right local pool to the left local pool.
    max_positive_flow = min(
        max(float(right_volume), 0.0),
        max(left_capacity - float(left_volume), 0.0),
    ) / time_step
    max_negative_flow = min(
        max(float(left_volume), 0.0),
        max(right_capacity - float(right_volume), 0.0),
    ) / time_step
    return float(np.clip(flow_rate, -max_negative_flow, max_positive_flow))


def limit_single_reservoir_flow_before_pool_merge(
    flow_rate,
    total_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    volume_offset,
    time_step,
):
    """Stop a dry-pool time step at first contact instead of overshooting it."""
    if not uses_stl_footprint_partition(reservoir_geometry) or abs(flow_rate) <= 1e-12:
        return flow_rate

    current_state = get_stl_port_pool_state(
        total_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        volume_offset,
    )
    if current_state["connected"]:
        return 0.0

    proposed_offset = volume_offset - flow_rate * time_step
    proposed_state = get_stl_port_pool_state(
        total_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        proposed_offset,
    )
    if not proposed_state["connected"]:
        return flow_rate

    # Find the largest transport fraction that leaves a dry gap.  A small
    # margin avoids a later volume-state reset caused by grid-level contact.
    lower_fraction = 0.0
    upper_fraction = 1.0
    for _ in range(18):
        midpoint = 0.5 * (lower_fraction + upper_fraction)
        midpoint_offset = volume_offset - flow_rate * time_step * midpoint
        midpoint_state = get_stl_port_pool_state(
            total_volume,
            signed_tilt_length,
            signed_tilt_width,
            reservoir_geometry,
            midpoint_offset,
        )
        if midpoint_state["connected"]:
            upper_fraction = midpoint
        else:
            lower_fraction = midpoint
    return flow_rate * max(0.0, lower_fraction * (1.0 - 1e-6))


def clip_polygon_against_vertical_line(points, x_limit, keep_left=True, tolerance=1e-9):
    polygon = np.asarray(points, dtype=float)
    if len(polygon) == 0:
        return np.empty((0, 2), dtype=float)

    def is_inside(point):
        if keep_left:
            return point[0] <= x_limit + tolerance
        return point[0] >= x_limit - tolerance

    clipped = []
    point_count = len(polygon)
    for index in range(point_count):
        start = polygon[index]
        end = polygon[(index + 1) % point_count]
        start_inside = is_inside(start)
        end_inside = is_inside(end)

        intersection = None
        delta_x = end[0] - start[0]
        if abs(delta_x) > tolerance:
            t_value = (x_limit - start[0]) / delta_x
            if -tolerance <= t_value <= 1.0 + tolerance:
                y_value = start[1] + t_value * (end[1] - start[1])
                intersection = np.asarray([x_limit, y_value], dtype=float)

        if start_inside and end_inside:
            clipped.append(end)
        elif start_inside and not end_inside:
            if intersection is not None:
                clipped.append(intersection)
        elif (not start_inside) and end_inside:
            if intersection is not None:
                clipped.append(intersection)
            clipped.append(end)

    if not clipped:
        return np.empty((0, 2), dtype=float)
    return simplify_polygon(np.asarray(clipped, dtype=float), tolerance=tolerance * 10)


def getHeigth(V, tilt, isRight, reservoir_geometry):
    # Empty reservoir
    if V <= 0:
        return 0

    res_width = reservoir_geometry["width"]
    res_heigth = reservoir_geometry["height"]
    res_length = reservoir_geometry["length"]

    # Cross-sectional area of liquid in the tilted plane [mm^2]
    A = np.clip(V / res_width, 0.0, res_heigth * res_length)

    # No tilt: the liquid surface stays flat across the reservoir
    if np.isclose(tilt, 0):
        return A / res_length

    L = res_length
    H = res_heigth
    m = np.tan(tilt)

    if np.isclose(m, 0):
        return A / L

    # Positive tilt makes the left side the high side and the right side the low side.
    low_clip_limit = H * H / (2 * m)
    full_wet_limit = 0.5 * m * L * L
    full_high_limit = H * L - full_wet_limit

    if (m * L > H) and (low_clip_limit < A < (H * L - low_clip_limit)):
        h_left = H
        h_right = 0.0
    elif A <= full_wet_limit:
        h_left = min(np.sqrt(2 * A * m), H)
        h_right = 0.0
    elif A >= full_high_limit:
        h_left = H
        h_right = max(H - np.sqrt(2 * (H * L - A) * m), 0.0)
    else:
        h_mean = A / L
        h_left = h_mean + 0.5 * m * L
        h_right = h_mean - 0.5 * m * L

    return h_right if isRight else h_left


def build_time_vector(stop_time, time_step=default_time_step):
    return np.arange(0, stop_time + time_step, time_step)


def getTiltFunc(
    time_points,
    mode,
    speed1,
    alpha,
    speed2=None,
    slow_start=None,
    slow_duration=None,
    progress_callback=None,
    progress_base=0,
    progress_total=1,
):
    tilt_y = []
    tilt_z = []
    point_count = len(time_points)
    for index, time in enumerate(time_points, start=1):
        tilt_y.append(
            -alpha
            * rotation_signal(
                time,
                mode,
                speed1,
                start_angle,
                speed2=speed2,
                slow_start=slow_start,
                slow_duration=slow_duration,
            )
        )
        tilt_z.append(
            -alpha
            * rotation_signal(
                time,
                mode,
                speed1,
                start_angle + phaseDiff,
                speed2=speed2,
                slow_start=slow_start,
                slow_duration=slow_duration,
            )
        )
        if progress_callback and (
            index == 1
            or index == point_count
            or index % progress_update_every == 0
        ):
            progress_callback(
                progress_base + index,
                progress_total,
                "Computing tilt traces...",
            )
    return np.array(tilt_y), np.array(tilt_z)


def build_geometry(width=None, height=None, length=None, shape="rectangular", radius=None):
    """Build either a rectangular channel or a circular pipe geometry."""
    geometry = {
        "shape": shape,
        "width": width,
        "height": height,
        "length": length,
        "radius": radius,
    }
    if shape == "circular":
        if radius is None or length is None:
            raise ValueError("Circular pipe geometry requires radius and length.")
    elif shape != "rectangular":
        raise ValueError(f"Unsupported channel shape: {shape}")
    elif width is None or height is None or length is None:
        raise ValueError("Rectangular channel geometry requires width, height, and length.")
    return geometry


def get_channel_volume(channel_geometry):
    if channel_geometry.get("shape", "rectangular") == "circular":
        return math.pi * channel_geometry["radius"] ** 2 * channel_geometry["length"]
    return (
        channel_geometry["width"]
        * channel_geometry["height"]
        * channel_geometry["length"]
    )


def get_hydraulic_resistance(channel_geometry):
    if channel_geometry.get("shape", "rectangular") == "circular":
        pipe_radius_m = channel_geometry["radius"] / 1000
        pipe_length_m = channel_geometry["length"] / 1000
        return 8 * viscosity * pipe_length_m / (math.pi * pipe_radius_m**4)
    channel_width = channel_geometry["width"]
    channel_height = channel_geometry["height"]
    channel_length = channel_geometry["length"]
    return (12 * viscosity * channel_length / 1000) / (
        (np.power(channel_height / 1000, 3) * channel_width / 1000)
        * (1 - 0.63 * (channel_height / channel_width))
    )


def calculate_wss_from_q1(q_1, channel_geometry):
    if channel_geometry.get("shape", "rectangular") == "circular":
        pipe_radius_m = channel_geometry["radius"] * 1e-3
        q_1_m3_s = q_1 * 1e-9
        return 4 * q_1_m3_s * viscosity / (math.pi * pipe_radius_m**3)
    channel_width_m = channel_geometry["width"] * 1e-3
    channel_height_m = channel_geometry["height"] * 1e-3
    q_1_m3_s = q_1 * 1e-9
    return (
        6
        * q_1_m3_s
        * viscosity
        / (channel_width_m * channel_height_m * channel_height_m)
    )


def calculate_reference_wss_from_q1(q_1, reference_wss_pa, reference_flow_uL_s):
    """Scale a signed Q1 trace from a user-defined WSS/flow reference pair."""
    if reference_flow_uL_s <= 0.0:
        raise ValueError("Reference flow rate must be greater than zero.")
    return q_1 * reference_wss_pa / reference_flow_uL_s


@dataclass
class ChannelWssProfileResult:
    width_mm: float
    height_mm: float
    mean_velocity_m_s: float
    volumetric_flow_m3_s: float
    representative_flow_uL_s: float
    representative_flow_basis: str
    reynolds_number: float
    regime_label: str
    friction_factor: float
    pressure_gradient_pa_m: float
    average_wall_shear_stress_pa: float
    left_wall_position_mm: np.ndarray
    left_wall_wss_pa: np.ndarray
    bottom_wall_position_mm: np.ndarray
    bottom_wall_wss_pa: np.ndarray
    profile_description: str


@dataclass
class CircularChannelWssResult:
    radius_mm: float
    mean_velocity_m_s: float
    volumetric_flow_m3_s: float
    representative_flow_uL_s: float
    representative_flow_basis: str
    reynolds_number: float
    wall_shear_stress_pa: float


@dataclass
class WallSelection:
    wall_name: str
    start_mm: float
    end_mm: float


def rectangular_laminar_poisseuille_number(aspect_ratio):
    beta = max(0.0, min(aspect_ratio, 1.0))
    return 96.0 * (
        1.0
        - 1.3553 * beta
        + 1.9467 * beta**2
        - 1.7012 * beta**3
        + 0.9564 * beta**4
        - 0.2537 * beta**5
    )


def stable_cosh_ratio(argument, limit):
    denominator = 1.0 + math.exp(-2.0 * limit)
    return (
        np.exp(argument - limit) + np.exp(-argument - limit)
    ) / denominator


@lru_cache(maxsize=64)
def laminar_profile_shape(width_mm, height_mm, term_count=80, sample_points=181):
    width_m = width_mm / 1000.0
    height_m = height_mm / 1000.0
    x_positions = np.linspace(0.0, width_m, sample_points)
    y_positions = np.linspace(0.0, height_m, sample_points)
    centered_x = x_positions - width_m / 2.0

    velocity_shape = np.zeros((sample_points, sample_points))
    left_wall_shape = np.zeros(sample_points)
    bottom_wall_shape = np.zeros(sample_points)
    modes = np.arange(1, 2 * term_count, 2, dtype=float)

    for mode in modes:
        limit = math.pi * mode * width_m / (2.0 * height_m)
        x_ratio = stable_cosh_ratio(math.pi * mode * centered_x / height_m, limit)
        one_minus_ratio = 1.0 - x_ratio
        sin_y = np.sin(math.pi * mode * y_positions / height_m)

        velocity_shape += (
            4.0
            * height_m**2
            / (math.pi**3 * mode**3)
            * np.outer(one_minus_ratio, sin_y)
        )
        left_wall_shape += (
            4.0
            * height_m
            / (math.pi**2 * mode**2)
            * math.tanh(limit)
            * sin_y
        )
        bottom_wall_shape += (
            4.0
            * height_m
            / (math.pi**2 * mode**2)
            * one_minus_ratio
        )

    return (
        x_positions,
        y_positions,
        float(np.mean(velocity_shape)),
        left_wall_shape,
        bottom_wall_shape,
    )


def average_wall_segment(position_mm, wss_pa, start_mm, end_mm):
    sample_positions = np.linspace(start_mm, end_mm, 400)
    sampled_wss = np.interp(sample_positions, position_mm, wss_pa)
    return float(np.mean(sampled_wss))


def get_channel_wss_wall_length(channel_geometry, wall_name):
    if wall_name == "Left side wall":
        return float(channel_geometry["height"])
    return float(channel_geometry["width"])


def build_channel_wss_selection(channel_geometry, wall_name, start_text, end_text):
    wall_length = get_channel_wss_wall_length(channel_geometry, wall_name)
    start_mm = validate_float_range(start_text, f"{wall_name} range start", 0.0, wall_length)
    end_mm = validate_float_range(end_text, f"{wall_name} range end", 0.0, wall_length)
    if start_mm >= end_mm:
        raise ValueError("Channel WSS range end must be greater than the start.")
    return WallSelection(wall_name=wall_name, start_mm=start_mm, end_mm=end_mm)


def profile_title_suffix(result):
    if result.regime_label == "Laminar":
        return "Laminar fully developed solution"
    return "Laminar-shape approximation scaled to the average WSS"


def calculate_channel_wss_profile(channel_geometry, representative_flow_uL_s, representative_flow_basis):
    width_mm = float(channel_geometry["width"])
    height_mm = float(channel_geometry["height"])
    width_m = width_mm / 1000.0
    height_m = height_mm / 1000.0
    area_m2 = width_m * height_m
    wetted_perimeter_m = 2.0 * (width_m + height_m)
    hydraulic_diameter_m = 4.0 * area_m2 / wetted_perimeter_m
    aspect_ratio = min(width_m, height_m) / max(width_m, height_m)

    volumetric_flow_m3_s = representative_flow_uL_s * 1e-9
    mean_velocity_m_s = volumetric_flow_m3_s / area_m2
    reynolds_number = density * abs(mean_velocity_m_s) * hydraulic_diameter_m / viscosity

    x_positions, y_positions, mean_shape, left_wall_shape, bottom_wall_shape = laminar_profile_shape(
        width_mm,
        height_mm,
    )

    if reynolds_number < 2300.0:
        regime_label = "Laminar"
        pressure_gradient_pa_m = viscosity * mean_velocity_m_s / mean_shape
        friction_factor = (
            2.0
            * abs(pressure_gradient_pa_m)
            * hydraulic_diameter_m
            / max(density * mean_velocity_m_s**2, 1e-12)
        )
        profile_description = "Laminar fully developed solution"
    else:
        laminar_po = rectangular_laminar_poisseuille_number(aspect_ratio)
        laminar_friction_factor = laminar_po / max(reynolds_number, 1e-12)
        turbulent_friction_factor = (
            0.79 * math.log(max(reynolds_number, 1.0)) - 1.64
        ) ** -2

        if reynolds_number < 4000.0:
            regime_label = "Transition"
            blend = (reynolds_number - 2300.0) / (4000.0 - 2300.0)
            friction_factor = (
                (1.0 - blend) * laminar_friction_factor
                + blend * turbulent_friction_factor
            )
        else:
            regime_label = "Turbulent"
            friction_factor = turbulent_friction_factor

        pressure_gradient_pa_m = math.copysign(
            friction_factor
            * density
            * mean_velocity_m_s**2
            / (2.0 * hydraulic_diameter_m),
            mean_velocity_m_s if mean_velocity_m_s != 0 else 1.0,
        )
        profile_description = "Laminar-shape approximation scaled to the average WSS"

    average_wall_shear_stress_pa = pressure_gradient_pa_m * area_m2 / wetted_perimeter_m
    left_wall_wss_pa = pressure_gradient_pa_m * left_wall_shape
    bottom_wall_wss_pa = pressure_gradient_pa_m * bottom_wall_shape

    return ChannelWssProfileResult(
        width_mm=width_mm,
        height_mm=height_mm,
        mean_velocity_m_s=mean_velocity_m_s,
        volumetric_flow_m3_s=volumetric_flow_m3_s,
        representative_flow_uL_s=representative_flow_uL_s,
        representative_flow_basis=representative_flow_basis,
        reynolds_number=reynolds_number,
        regime_label=regime_label,
        friction_factor=friction_factor,
        pressure_gradient_pa_m=pressure_gradient_pa_m,
        average_wall_shear_stress_pa=average_wall_shear_stress_pa,
        left_wall_position_mm=y_positions * 1000.0,
        left_wall_wss_pa=left_wall_wss_pa,
        bottom_wall_position_mm=x_positions * 1000.0,
        bottom_wall_wss_pa=bottom_wall_wss_pa,
        profile_description=profile_description,
    )


def calculate_circular_channel_wss(
    channel_geometry,
    representative_flow_uL_s,
    representative_flow_basis,
):
    """Calculate the uniform pipe-wall WSS from the selected flow magnitude."""
    radius_mm = float(channel_geometry["radius"])
    radius_m = radius_mm * 1e-3
    volumetric_flow_m3_s = abs(float(representative_flow_uL_s)) * 1e-9
    cross_section_area_m2 = math.pi * radius_m**2
    mean_velocity_m_s = volumetric_flow_m3_s / cross_section_area_m2
    reynolds_number = density * mean_velocity_m_s * (2.0 * radius_m) / viscosity
    wall_shear_stress_pa = (
        4.0 * viscosity * volumetric_flow_m3_s / (math.pi * radius_m**3)
    )
    return CircularChannelWssResult(
        radius_mm=radius_mm,
        mean_velocity_m_s=mean_velocity_m_s,
        volumetric_flow_m3_s=volumetric_flow_m3_s,
        representative_flow_uL_s=abs(float(representative_flow_uL_s)),
        representative_flow_basis=representative_flow_basis,
        reynolds_number=reynolds_number,
        wall_shear_stress_pa=wall_shear_stress_pa,
    )


def build_channel_geometry_axis(axis, result, selection, selected_average_wss_pa):
    width_mm = result.width_mm
    height_mm = result.height_mm
    axis.set_facecolor("#f7f7f5")
    axis.fill(
        [0.0, width_mm, width_mm, 0.0],
        [0.0, 0.0, height_mm, height_mm],
        color="#9ecae1",
        alpha=0.65,
    )
    axis.plot(
        [0.0, width_mm, width_mm, 0.0, 0.0],
        [0.0, 0.0, height_mm, height_mm, 0.0],
        color="#1d3557",
        linewidth=2.2,
    )

    if selection.wall_name == "Bottom wall":
        axis.plot(
            [selection.start_mm, selection.end_mm],
            [0.0, 0.0],
            color="#e76f51",
            linewidth=5.0,
            solid_capstyle="butt",
            label="Selected bottom segment",
        )
    else:
        axis.plot(
            [0.0, 0.0],
            [selection.start_mm, selection.end_mm],
            color="#e76f51",
            linewidth=5.0,
            solid_capstyle="butt",
            label="Selected left-wall segment",
        )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.08 * width_mm, 1.08 * width_mm)
    axis.set_ylim(-0.12 * height_mm, 1.08 * height_mm)
    axis.set_xlabel("Width [mm]")
    axis.set_ylabel("Height [mm]")
    axis.set_title("Primary channel geometry")
    axis.grid(True, alpha=0.2)
    axis.text(
        0.5,
        -0.24,
        (
            f"{result.representative_flow_basis}: {result.representative_flow_uL_s:.4f} uL/s\n"
            f"Selected average WSS: {selected_average_wss_pa:.4f} Pa\n"
            f"Regime: {result.regime_label}"
        ),
        transform=axis.transAxes,
        va="top",
        ha="center",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#c7c7c7"},
        clip_on=False,
    )


def build_channel_wss_axis(
    axis,
    position_mm,
    wss_pa,
    average_wss_pa,
    selection,
    active_wall_name,
    line_color,
    x_label,
    title,
):
    axis.plot(position_mm, wss_pa, color=line_color, linewidth=2.0)
    axis.axhline(
        average_wss_pa,
        color="#b23a48",
        linestyle="--",
        linewidth=1.4,
        label="Perimeter average WSS",
    )

    if selection.wall_name == active_wall_name:
        selected_average = average_wall_segment(
            position_mm,
            wss_pa,
            selection.start_mm,
            selection.end_mm,
        )
        segment_positions = np.linspace(selection.start_mm, selection.end_mm, 300)
        segment_wss = np.interp(segment_positions, position_mm, wss_pa)
        axis.axvspan(
            selection.start_mm,
            selection.end_mm,
            color="#d9ed92",
            alpha=0.28,
            label="Selected range",
        )
        axis.plot(
            segment_positions,
            segment_wss,
            color="#264653",
            linewidth=3.0,
            label="Selected segment",
        )
        axis.hlines(
            selected_average,
            selection.start_mm,
            selection.end_mm,
            colors="#e76f51",
            linewidth=2.0,
            label="Selected average",
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel("Wall shear stress [Pa]")
    axis.set_title(title)
    axis.set_xlim(float(position_mm[0]), float(position_mm[-1]))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")


def create_channel_wss_figure(result, selection):
    if selection.wall_name == "Left side wall":
        selected_average_wss_pa = average_wall_segment(
            result.left_wall_position_mm,
            result.left_wall_wss_pa,
            selection.start_mm,
            selection.end_mm,
        )
    else:
        selected_average_wss_pa = average_wall_segment(
            result.bottom_wall_position_mm,
            result.bottom_wall_wss_pa,
            selection.start_mm,
            selection.end_mm,
        )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    build_channel_geometry_axis(axes[0], result, selection, selected_average_wss_pa)
    if selection.wall_name == "Bottom wall":
        build_channel_wss_axis(
            axes[1],
            result.bottom_wall_position_mm,
            result.bottom_wall_wss_pa,
            result.average_wall_shear_stress_pa,
            selection,
            "Bottom wall",
            "#2a9d8f",
            "Position along bottom wall [mm]",
            f"Bottom-wall WSS\n{profile_title_suffix(result)}",
        )
    else:
        build_channel_wss_axis(
            axes[1],
            result.left_wall_position_mm,
            result.left_wall_wss_pa,
            result.average_wall_shear_stress_pa,
            selection,
            "Left side wall",
            "#1d3557",
            "Position along side wall [mm]",
            f"Left-wall WSS\n{profile_title_suffix(result)}",
        )
    figure.suptitle(f"Primary channel {selection.wall_name.lower()} shear-stress profile", fontsize=13)
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    return figure, selected_average_wss_pa


def create_circular_channel_wss_figure(result):
    """Create a uniform WSS view for a circular pipe cross section."""
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    theta = np.linspace(0.0, 2.0 * math.pi, 160)
    circle_x = result.radius_mm * np.cos(theta)
    circle_y = result.radius_mm * np.sin(theta)
    axes[0].fill(circle_x, circle_y, color="#9ecae1", alpha=0.65)
    axes[0].plot(circle_x, circle_y, color="#1d3557", linewidth=2.2)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(-1.2 * result.radius_mm, 1.2 * result.radius_mm)
    axes[0].set_ylim(-1.2 * result.radius_mm, 1.2 * result.radius_mm)
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("Circular primary channel")
    axes[0].grid(True, alpha=0.2)
    axes[0].text(
        0.5,
        -0.18,
        (
            f"Radius: {result.radius_mm:.4f} mm\n"
            f"{result.representative_flow_basis}: "
            f"{result.representative_flow_uL_s:.4f} uL/s\n"
            f"Re={result.reynolds_number:.2f}"
        ),
        transform=axes[0].transAxes,
        va="top",
        ha="center",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#c7c7c7"},
        clip_on=False,
    )

    axes[1].bar(["Pipe wall"], [result.wall_shear_stress_pa], color="#2a9d8f", width=0.55)
    axes[1].axhline(
        result.wall_shear_stress_pa,
        color="#b23a48",
        linestyle="--",
        linewidth=1.4,
        label="Uniform wall WSS",
    )
    axes[1].set_ylabel("Wall shear stress [Pa]")
    axes[1].set_title("Circular-pipe wall shear stress")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(loc="best")
    axes[1].text(
        0.5,
        1.02,
        f"{result.wall_shear_stress_pa:.5f} Pa",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
    )
    figure.suptitle("Primary circular-pipe wall shear stress", fontsize=13)
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    return figure, result.wall_shear_stress_pa


def export_channel_wss_csv(csv_path, result, selection, selected_average_wss_pa):
    csv_file = Path(csv_path).with_suffix(".csv")
    if isinstance(result, CircularChannelWssResult):
        with csv_file.open("w", encoding="utf-8", newline="") as handle:
            handle.write("Channel shape,Circular pipe\n")
            handle.write(f"Radius mm,{result.radius_mm:.6f}\n")
            handle.write(f"Representative flow basis,{result.representative_flow_basis}\n")
            handle.write(f"Representative flow uL/s,{result.representative_flow_uL_s:.6f}\n")
            handle.write(f"Mean velocity m/s,{result.mean_velocity_m_s:.9f}\n")
            handle.write(f"Reynolds number,{result.reynolds_number:.6f}\n")
            handle.write(
                "WSS formula,4*viscosity*flow_rate/(pi*radius^3)\n"
            )
            handle.write(f"Uniform wall WSS Pa,{result.wall_shear_stress_pa:.9f}\n")
        return csv_file

    row_count = max(
        len(result.left_wall_position_mm),
        len(result.bottom_wall_position_mm),
    )
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"Representative flow basis,{result.representative_flow_basis}\n")
        handle.write(f"Representative flow uL/s,{result.representative_flow_uL_s:.6f}\n")
        handle.write(f"Mean velocity m/s,{result.mean_velocity_m_s:.9f}\n")
        handle.write(f"Reynolds number,{result.reynolds_number:.6f}\n")
        handle.write(f"Flow regime,{result.regime_label}\n")
        handle.write(f"Pressure gradient Pa/m,{result.pressure_gradient_pa_m:.6f}\n")
        handle.write(
            f"Perimeter average WSS Pa,{result.average_wall_shear_stress_pa:.6f}\n"
        )
        handle.write(f"Selected wall,{selection.wall_name}\n")
        handle.write(f"Selected start mm,{selection.start_mm:.6f}\n")
        handle.write(f"Selected end mm,{selection.end_mm:.6f}\n")
        handle.write(f"Selected average WSS Pa,{selected_average_wss_pa:.6f}\n")
        handle.write("\n")
        handle.write(
            "left_wall_position_mm,left_wall_wss_pa,bottom_wall_position_mm,bottom_wall_wss_pa\n"
        )
        for index in range(row_count):
            left_position = (
                f"{result.left_wall_position_mm[index]:.6f}"
                if index < len(result.left_wall_position_mm)
                else ""
            )
            left_wss = (
                f"{result.left_wall_wss_pa[index]:.6f}"
                if index < len(result.left_wall_wss_pa)
                else ""
            )
            bottom_position = (
                f"{result.bottom_wall_position_mm[index]:.6f}"
                if index < len(result.bottom_wall_position_mm)
                else ""
            )
            bottom_wss = (
                f"{result.bottom_wall_wss_pa[index]:.6f}"
                if index < len(result.bottom_wall_wss_pa)
                else ""
            )
            handle.write(
                f"{left_position},{left_wss},{bottom_position},{bottom_wss}\n"
            )
    return csv_file


def calculate_dh_min(cap_force):
    return 1000 * cap_force / (density * gravity)


def get_simulation_setup(mode, reservoir_geometry, primary_geometry, secondary_geometry=None):
    if mode == "symmetric":
        secondary_geometry = primary_geometry
    elif mode == "assymmetric":
        if secondary_geometry is None:
            raise ValueError("Assymmetric mode requires a second channel geometry.")
    elif mode in (
        "single_reservoir",
        "single_reservoir_assymmetric",
        "simple_tilting",
        "four_reservoir_tilting",
    ):
        secondary_geometry = None
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    channel_count = 4 if mode == "four_reservoir_tilting" else 1
    adjusted_volume = reservoir_geometry["v_fill_ul"] - channel_count * get_channel_volume(primary_geometry)
    if secondary_geometry is not None:
        adjusted_volume -= get_channel_volume(secondary_geometry)
    if adjusted_volume <= 0:
        raise ValueError("Channel volume is too large for the filled reservoir volume.")
    if mode in ("single_reservoir", "single_reservoir_assymmetric"):
        if adjusted_volume >= reservoir_geometry["volume_ul_max"]:
            raise ValueError("Volume too high for reservoir geometry.")
        resistance_1 = get_hydraulic_resistance(primary_geometry)
        return adjusted_volume, resistance_1, None, primary_geometry, None

    if mode == "simple_tilting":
        if reservoir_geometry["type"] != "cylindrical":
            raise ValueError("Simple tilting mode requires cylindrical reservoir geometry.")
        if adjusted_volume > 2 * reservoir_geometry["volume_ul_max"]:
            raise ValueError(
                "Filling volume does not fit into the two cylindrical reservoirs."
            )
        resistance_1 = get_hydraulic_resistance(primary_geometry)
        return adjusted_volume, resistance_1, None, primary_geometry, None

    if mode == "four_reservoir_tilting":
        if reservoir_geometry["type"] != "cylindrical":
            raise ValueError("Four-reservoir tilting mode requires cylindrical reservoir geometry.")
        if adjusted_volume > 4 * reservoir_geometry["volume_ul_max"]:
            raise ValueError(
                "Filling volume does not fit into the four cylindrical reservoirs."
            )
        resistance_1 = get_hydraulic_resistance(primary_geometry)
        return adjusted_volume, resistance_1, None, primary_geometry, None

    if adjusted_volume >= 2 * reservoir_geometry["volume_ul_max"]:
        raise ValueError("Volume too high for reservoir geometry.")

    resistance_1 = get_hydraulic_resistance(primary_geometry)
    resistance_2 = get_hydraulic_resistance(secondary_geometry)
    return adjusted_volume, resistance_1, resistance_2, primary_geometry, secondary_geometry


def build_case(
    mode,
    speed1,
    resistance_1,
    resistance_2,
    primary_geometry,
    secondary_geometry,
    speed2=None,
    slow_start=None,
    slow_duration=None,
):
    if mode == "symmetric":
        return {
            "mode": "symmetric",
            "title": default_mode_settings["symmetric"]["title"],
            "speed1": speed1,
            "speed2": speed1,
            "slow_start": None,
            "slow_duration": None,
            "R1": resistance_1,
            "R2": resistance_1,
            "geometry1": primary_geometry,
            "geometry2": primary_geometry,
        }

    if mode == "assymmetric":
        if speed2 is None or slow_start is None or slow_duration is None:
            raise ValueError(
                "Assymmetric mode requires speed2, slow_start, and slow_duration."
            )
        return {
            "mode": "assymmetric",
            "title": default_mode_settings["assymmetric"]["title"],
            "speed1": speed1,
            "speed2": speed2,
            "slow_start": slow_start,
            "slow_duration": slow_duration,
            "R1": resistance_1,
            "R2": resistance_2,
            "geometry1": primary_geometry,
            "geometry2": secondary_geometry,
        }

    if mode == "single_reservoir":
        return {
            "mode": "single_reservoir",
            "title": default_mode_settings["single_reservoir"]["title"],
            "speed1": speed1,
            "speed2": None,
            "slow_start": None,
            "slow_duration": None,
            "R1": resistance_1,
            "R2": None,
            "geometry1": primary_geometry,
            "geometry2": None,
        }

    if mode == "single_reservoir_assymmetric":
        if speed2 is None or slow_start is None or slow_duration is None:
            raise ValueError(
                "Single-reservoir assymmetric mode requires speed2, slow_start, and slow_duration."
            )
        return {
            "mode": "single_reservoir_assymmetric",
            "title": default_mode_settings["single_reservoir_assymmetric"]["title"],
            "speed1": speed1,
            "speed2": speed2,
            "slow_start": slow_start,
            "slow_duration": slow_duration,
            "R1": resistance_1,
            "R2": None,
            "geometry1": primary_geometry,
            "geometry2": None,
        }

    if mode == "simple_tilting":
        return {
            "mode": "simple_tilting",
            "title": default_mode_settings["simple_tilting"]["title"],
            "speed1": speed1,
            "speed2": None,
            "slow_start": None,
            "slow_duration": None,
            "R1": resistance_1,
            "R2": None,
            "geometry1": primary_geometry,
            "geometry2": None,
        }

    if mode == "four_reservoir_tilting":
        return {
            "mode": "four_reservoir_tilting",
            "title": default_mode_settings["four_reservoir_tilting"]["title"],
            "speed1": speed1,
            "speed2": None,
            "slow_start": None,
            "slow_duration": None,
            "R1": resistance_1,
            "R2": None,
            "geometry1": primary_geometry,
            "geometry2": None,
        }

    raise ValueError(f"Unsupported mode: {mode}")


def calculate_two_reservoir_channel_driving_heads(
    reservoir_1_volume,
    reservoir_2_volume,
    tilt_length,
    tilt_width,
    dh_tilt_1,
    dh_tilt_2,
    reservoir_geometry,
    capillary_head_threshold,
    use_cached_heads=True,
    use_wet_dry_gating=True,
    reservoir_2_geometry=None,
    reservoir_2_tilt_width=None,
    use_geometry_port_driving_heads=False,
    channel_port_positions=None,
    global_tilt_width=None,
):
    """Evaluate the capillary-corrected two-reservoir driving heads in mm."""
    if reservoir_2_geometry is None:
        reservoir_2_geometry = reservoir_geometry
    if reservoir_2_tilt_width is None:
        reservoir_2_tilt_width = tilt_width
    if global_tilt_width is None:
        global_tilt_width = tilt_width
    v1_left, v1_right = get_oriented_port_heads(
        reservoir_1_volume,
        tilt_length,
        tilt_width,
        reservoir_geometry,
        use_cached_heads=use_cached_heads,
    )
    v2_left, v2_right = get_oriented_port_heads(
        reservoir_2_volume,
        tilt_length,
        reservoir_2_tilt_width,
        reservoir_2_geometry,
        use_cached_heads=use_cached_heads,
    )
    if use_geometry_port_driving_heads:
        if channel_port_positions is None:
            raise ValueError("Geometry-based driving heads require channel port positions.")
        slope_length, slope_width, vertical_scale = get_stl_gravity_plane(
            tilt_length, global_tilt_width
        )
        vertical_vector = np.array(
            [
                slope_length * vertical_scale,
                slope_width * vertical_scale,
                vertical_scale,
            ],
            dtype=float,
        )

        def vertical_head(head, geometry, local_tilt_width):
            if uses_stl_footprint_partition(geometry):
                return float(head)
            _, _, local_vertical_scale = get_stl_gravity_plane(
                tilt_length, local_tilt_width
            )
            return float(head) * local_vertical_scale

        v1_left = vertical_head(v1_left, reservoir_geometry, tilt_width)
        v1_right = vertical_head(v1_right, reservoir_geometry, tilt_width)
        v2_left = vertical_head(
            v2_left, reservoir_2_geometry, reservoir_2_tilt_width
        )
        v2_right = vertical_head(
            v2_right, reservoir_2_geometry, reservoir_2_tilt_width
        )
        channel_1_source, channel_1_destination = channel_port_positions["channel_1"]
        channel_2_source, channel_2_destination = channel_port_positions["channel_2"]
        raw_dh_1 = (
            float(np.dot(vertical_vector, channel_1_source - channel_1_destination))
            + v1_right
            - v2_right
        )
        raw_dh_2 = (
            float(np.dot(vertical_vector, channel_2_source - channel_2_destination))
            + v1_left
            - v2_left
        )
    else:
        raw_dh_1 = dh_tilt_1 + v1_right - v2_right
        raw_dh_2 = dh_tilt_2 + v1_left - v2_left
    if use_wet_dry_gating:
        dh_1 = apply_capillary_cutoff(
            raw_dh_1,
            v1_right,
            v2_right,
            capillary_head_threshold,
        )
        dh_2 = apply_capillary_cutoff(
            raw_dh_2,
            v1_left,
            v2_left,
            capillary_head_threshold,
        )
    else:
        dh_1, dh_2 = raw_dh_1, raw_dh_2
    return dh_1, dh_2


def calculate_two_reservoir_channel_flows(
    reservoir_1_volume,
    reservoir_2_volume,
    tilt_length,
    tilt_width,
    dh_tilt_1,
    dh_tilt_2,
    reservoir_geometry,
    channel_1_coefficient,
    channel_2_coefficient,
    capillary_head_threshold,
    use_cached_heads=True,
    use_wet_dry_gating=True,
    reservoir_2_geometry=None,
    reservoir_2_tilt_width=None,
    use_geometry_port_driving_heads=False,
    channel_port_positions=None,
    global_tilt_width=None,
):
    """Evaluate both two-reservoir channel flows for a candidate volume state."""
    dh_1, dh_2 = calculate_two_reservoir_channel_driving_heads(
        reservoir_1_volume,
        reservoir_2_volume,
        tilt_length,
        tilt_width,
        dh_tilt_1,
        dh_tilt_2,
        reservoir_geometry,
        capillary_head_threshold,
        use_cached_heads=use_cached_heads,
        use_wet_dry_gating=use_wet_dry_gating,
        reservoir_2_geometry=reservoir_2_geometry,
        reservoir_2_tilt_width=reservoir_2_tilt_width,
        use_geometry_port_driving_heads=use_geometry_port_driving_heads,
        channel_port_positions=channel_port_positions,
        global_tilt_width=global_tilt_width,
    )
    return dh_1 * channel_1_coefficient, dh_2 * channel_2_coefficient


def solve_two_reservoir_volume_step_implicit(
    reservoir_1_volume,
    total_volume,
    reservoir_capacity,
    tilt_length,
    tilt_width,
    dh_tilt_1,
    dh_tilt_2,
    reservoir_geometry,
    channel_1_coefficient,
    channel_2_coefficient,
    capillary_head_threshold,
    time_step,
    use_wet_dry_gating=True,
    reservoir_2_geometry=None,
    reservoir_2_tilt_width=None,
    use_geometry_port_driving_heads=False,
    channel_port_positions=None,
    global_tilt_width=None,
):
    """Advance the coupled two-reservoir volume balance with bounded bisection.

    The root is resolved against the same quantized STL water-level cache used
    by the fast solver.  Refining below one cached volume bin would not change
    the hydraulic heads, while it would require many unnecessary free-surface
    solves.
    """
    lower_volume = max(0.0, total_volume - reservoir_capacity)
    upper_volume = min(reservoir_capacity, total_volume)

    def residual(candidate_volume):
        candidate_other_volume = total_volume - candidate_volume
        q_1, q_2 = calculate_two_reservoir_channel_flows(
            candidate_volume,
            candidate_other_volume,
            tilt_length,
            tilt_width,
            dh_tilt_1,
            dh_tilt_2,
            reservoir_geometry,
            channel_1_coefficient,
            channel_2_coefficient,
            capillary_head_threshold,
            use_cached_heads=True,
            use_wet_dry_gating=use_wet_dry_gating,
            reservoir_2_geometry=reservoir_2_geometry,
            reservoir_2_tilt_width=reservoir_2_tilt_width,
            use_geometry_port_driving_heads=use_geometry_port_driving_heads,
            channel_port_positions=channel_port_positions,
            global_tilt_width=global_tilt_width,
        )
        return candidate_volume - reservoir_1_volume + time_step * (q_1 + q_2)

    lower_residual = residual(lower_volume)
    upper_residual = residual(upper_volume)
    if lower_residual * upper_residual <= 0.0:
        # Twelve halvings are finer than the 300-bin STL volume lookup.
        for _ in range(12):
            midpoint = 0.5 * (lower_volume + upper_volume)
            midpoint_residual = residual(midpoint)
            if midpoint_residual == 0.0:
                lower_volume = upper_volume = midpoint
                break
            if lower_residual * midpoint_residual <= 0.0:
                upper_volume = midpoint
                upper_residual = midpoint_residual
            else:
                lower_volume = midpoint
                lower_residual = midpoint_residual
        next_reservoir_1_volume = 0.5 * (lower_volume + upper_volume)
    else:
        # A capillary discontinuity can remove the strict sign bracket. Choose
        # the bounded state with the smaller balance residual in that case.
        next_reservoir_1_volume = (
            lower_volume
            if abs(lower_residual) <= abs(upper_residual)
            else upper_volume
        )
    next_reservoir_2_volume = total_volume - next_reservoir_1_volume
    q_1, q_2 = calculate_two_reservoir_channel_flows(
        next_reservoir_1_volume,
        next_reservoir_2_volume,
        tilt_length,
        tilt_width,
        dh_tilt_1,
        dh_tilt_2,
        reservoir_geometry,
        channel_1_coefficient,
        channel_2_coefficient,
        capillary_head_threshold,
        use_cached_heads=True,
        use_wet_dry_gating=use_wet_dry_gating,
        reservoir_2_geometry=reservoir_2_geometry,
        reservoir_2_tilt_width=reservoir_2_tilt_width,
        use_geometry_port_driving_heads=use_geometry_port_driving_heads,
        channel_port_positions=channel_port_positions,
        global_tilt_width=global_tilt_width,
    )
    return next_reservoir_1_volume, next_reservoir_2_volume, q_1, q_2


def calculate_four_reservoir_ring_flows(
    reservoir_volumes,
    tilt_length,
    tilt_width,
    channel_length,
    reservoir_geometry,
    channel_coefficient,
    capillary_head_threshold,
    use_wet_dry_gating=True,
):
    """Compute signed flows around the four-cylinder ring.

    Channels are ordered clockwise: bottom (1 to 2), right (2 to 3), top
    (3 to 4), and left (4 to 1). Positive flow follows that ordering. Each
    gate head is evaluated from the current reservoir volume and local gate
    position before any of the four volumes are changed.
    """
    radius = float(reservoir_geometry["radius"])
    channel_definitions = (
        (0, (radius, 0.0), 1, (-radius, 0.0), (channel_length, 0.0)),
        (1, (0.0, radius), 2, (0.0, -radius), (0.0, channel_length)),
        (2, (-radius, 0.0), 3, (radius, 0.0), (-channel_length, 0.0)),
        (3, (0.0, -radius), 0, (0.0, radius), (0.0, -channel_length)),
    )
    flows = []
    source_heads = []
    destination_heads = []
    raw_driving_heads = []
    for source_index, source_gate, destination_index, destination_gate, direction in channel_definitions:
        source_head = get_cylindrical_gate_head(
            reservoir_volumes[source_index],
            tilt_length,
            tilt_width,
            reservoir_geometry,
            *source_gate,
        )
        destination_head = get_cylindrical_gate_head(
            reservoir_volumes[destination_index],
            tilt_length,
            tilt_width,
            reservoir_geometry,
            *destination_gate,
        )
        tilt_head = (
            float(direction[0]) * math.sin(tilt_width)
            + float(direction[1]) * math.sin(tilt_length)
        )
        raw_head = tilt_head + source_head - destination_head
        driving_head = (
            apply_capillary_cutoff(
                raw_head,
                source_head,
                destination_head,
                capillary_head_threshold,
            )
            if use_wet_dry_gating
            else raw_head
        )
        flows.append(driving_head * channel_coefficient)
        source_heads.append(source_head)
        destination_heads.append(destination_head)
        raw_driving_heads.append(raw_head)
    return (
        np.asarray(flows, dtype=float),
        np.asarray(source_heads, dtype=float),
        np.asarray(destination_heads, dtype=float),
        np.asarray(raw_driving_heads, dtype=float),
    )


def limit_four_reservoir_ring_flows(flows, reservoir_volumes, capacity, time_step):
    """Scale a simultaneous ring update so every reservoir stays feasible."""
    flows = np.asarray(flows, dtype=float)
    volumes = np.asarray(reservoir_volumes, dtype=float)
    net_volume_rate = np.array(
        [
            flows[3] - flows[0],
            flows[0] - flows[1],
            flows[1] - flows[2],
            flows[2] - flows[3],
        ],
        dtype=float,
    )
    scale = 1.0
    for volume, rate in zip(volumes, net_volume_rate):
        delta = rate * time_step
        if delta < 0.0:
            scale = min(scale, volume / -delta)
        elif delta > 0.0:
            scale = min(scale, (capacity - volume) / delta)
    return flows * max(0.0, min(1.0, scale))


def simulate(
    time_points,
    mode,
    speed1,
    alpha,
    WSS_ref,
    R1,
    V_fill,
    reservoir_geometry,
    geometry1,
    cap_force=default_cap_force,
    speed2=None,
    slow_start=None,
    slow_duration=None,
    R2=None,
    geometry2=None,
    reservoir_2_geometry=None,
    use_geometry_port_driving_heads=True,
    time_step=default_time_step,
    use_wet_dry_gating=True,
    stl_wetting_solver="predictor_corrector",
    two_reservoir_stl_solver="cached",
    progress_callback=None,
    progress_base=0,
    progress_total=1,
    progress_message="Running simulation...",
):
    if mode == "symmetric":
        speed2 = speed1
        R2 = R1
        slow_start = None
        slow_duration = None
        geometry2 = geometry1
    elif mode == "assymmetric":
        if (
            speed2 is None
            or slow_start is None
            or slow_duration is None
            or R2 is None
            or geometry2 is None
        ):
            raise ValueError(
                "Assymmetric mode requires speed2, slow_start, slow_duration, R2, and geometry2."
            )
    elif mode == "single_reservoir":
        speed2 = None
        slow_start = None
        slow_duration = None
        R2 = None
        geometry2 = None
    elif mode == "single_reservoir_assymmetric":
        if speed2 is None or slow_start is None or slow_duration is None:
            raise ValueError(
                "Single-reservoir assymmetric mode requires speed2, slow_start, and slow_duration."
            )
        R2 = None
        geometry2 = None
    elif mode == "simple_tilting":
        speed2 = None
        slow_start = None
        slow_duration = None
        R2 = None
        geometry2 = None
    elif mode == "four_reservoir_tilting":
        speed2 = None
        slow_start = None
        slow_duration = None
        R2 = None
        geometry2 = None
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode in ("symmetric", "assymmetric"):
        if reservoir_2_geometry is None:
            reservoir_2_geometry = mirror_reservoir_geometry_across_width(
                reservoir_geometry
            )
    else:
        reservoir_2_geometry = reservoir_geometry
    channel_port_positions = (
        get_two_reservoir_channel_port_positions(
            reservoir_geometry,
            geometry1,
            geometry2,
        )
        if mode in ("symmetric", "assymmetric") and use_geometry_port_driving_heads
        else None
    )
    if two_reservoir_stl_solver not in ("cached", "implicit"):
        raise ValueError("Two-reservoir STL solver must be 'cached' or 'implicit'.")

    q_1 = []
    q_2 = []
    q_3 = []
    q_4 = []
    wall_shear_stress = []
    reservoir_1_volume = []
    reservoir_2_volume = []
    reservoir_3_volume = []
    reservoir_4_volume = []
    left_local_volume_trace = []
    right_local_volume_trace = []
    left_port_head_trace = []
    right_port_head_trace = []
    driving_head_trace = []
    pool_connected_trace = []
    channel_1_midpoint_driving_pressure_trace = []
    channel_2_midpoint_driving_pressure_trace = []
    dh_min = calculate_dh_min(cap_force)

    # gravity pressure / hydraulic resistance
    k1 = density * gravity / (R1 * 0.000001)
    k2 = None if R2 is None else density * gravity / (R2 * 0.000001)

    if mode in ("single_reservoir", "single_reservoir_assymmetric"):
        V_reservoir = V_fill
        V1 = None
        V2 = None
        single_reservoir_volume_offset = 0.0
        stl_local_volumes = None
        stl_topology = "unknown"
        stl_smoothed_equilibrium_left_volume = None
        stl_pressure_difference_pa = 0.0
        stl_compartments = (
            build_single_reservoir_compartments(reservoir_geometry)
            if uses_stl_footprint_partition(reservoir_geometry)
            else None
        )
    else:
        V1 = V_fill / 2
        V2 = V_fill / 2
    if mode == "four_reservoir_tilting":
        ring_volumes = np.full(4, V_fill / 4.0, dtype=float)

    V_max = reservoir_geometry["volume_ul_max"]

    point_count = len(time_points)
    for index, time in enumerate(time_points, start=1):
        tilt_y = (
            -alpha
            / 180
            * np.pi
            * rotation_signal(
                time,
                mode,
                speed1,
                start_angle,
                speed2=speed2,
                slow_start=slow_start,
                slow_duration=slow_duration,
            )
        )
        if mode in ("single_reservoir", "single_reservoir_assymmetric"):
            dh_tilt_1 = get_port_separation_along_length(reservoir_geometry) * np.sin(tilt_y)
        else:
            dh_tilt_1 = geometry1["length"] * np.sin(tilt_y)
        dh_tilt_2 = 0 if geometry2 is None else geometry2["length"] * np.sin(tilt_y)

        tilt_z = (
            -alpha
            / 180
            * np.pi
            * rotation_signal(
                time,
                mode,
                speed1,
                start_angle + phaseDiff,
                speed2=speed2,
                slow_start=slow_start,
                slow_duration=slow_duration,
            )
        )
        isInverted = tilt_z < 0
        pools_connected = False
        channel_1_midpoint_driving_pressure = np.nan
        channel_2_midpoint_driving_pressure = np.nan

        if mode == "four_reservoir_tilting":
            Q_ring, gate_source_heads, gate_destination_heads, raw_ring_heads = (
                calculate_four_reservoir_ring_flows(
                    ring_volumes,
                    tilt_z,
                    tilt_y,
                    geometry1["length"],
                    reservoir_geometry,
                    k1,
                    dh_min,
                    use_wet_dry_gating=use_wet_dry_gating,
                )
            )
            Q_ring = limit_four_reservoir_ring_flows(
                Q_ring,
                ring_volumes,
                V_max,
                time_step,
            )
            ring_volume_rates = np.array(
                [
                    Q_ring[3] - Q_ring[0],
                    Q_ring[0] - Q_ring[1],
                    Q_ring[1] - Q_ring[2],
                    Q_ring[2] - Q_ring[3],
                ],
                dtype=float,
            )
            ring_volumes = np.clip(
                ring_volumes + ring_volume_rates * time_step,
                0.0,
                V_max,
            )
            # Preserve the global balance exactly after harmless floating-point drift.
            ring_volumes[3] += V_fill - float(np.sum(ring_volumes))
            Q1, Q2, Q3, Q4 = Q_ring
            V1, V2, V3, V4 = ring_volumes
            reservoir_1_volume.append(V1)
            reservoir_2_volume.append(V2)
            reservoir_3_volume.append(V3)
            reservoir_4_volume.append(V4)
            left_local_volume_trace.append(V1)
            right_local_volume_trace.append(V2)
            left_port_head_trace.append(gate_source_heads[0])
            right_port_head_trace.append(gate_destination_heads[0])
            driving_head_trace.append(raw_ring_heads[0])
        elif mode == "simple_tilting":
            h_left = get_cylindrical_port_head(V1, tilt_z, tilt_y, reservoir_geometry)
            h_right = get_cylindrical_port_head(V2, tilt_z, tilt_y, reservoir_geometry)
            raw_driving_head = dh_tilt_1 + h_left - h_right
            driving_head = apply_capillary_cutoff(
                raw_driving_head,
                h_left,
                h_right,
                dh_min,
            )
            Q1 = driving_head * k1
            # Positive Q transfers volume from cylinder 1 to cylinder 2.
            max_positive_flow = min(V1, V_max - V2) / time_step
            max_negative_flow = min(V2, V_max - V1) / time_step
            Q1 = float(np.clip(Q1, -max_negative_flow, max_positive_flow))
            V1 -= Q1 * time_step
            V2 = V_fill - V1
            Q2 = 0.0
            reservoir_1_volume.append(V1)
            reservoir_2_volume.append(V2)
            reservoir_3_volume.append(0.0)
            reservoir_4_volume.append(0.0)
            left_local_volume_trace.append(V1)
            right_local_volume_trace.append(V2)
            left_port_head_trace.append(h_left)
            right_port_head_trace.append(h_right)
            driving_head_trace.append(raw_driving_head)
        elif mode in ("single_reservoir", "single_reservoir_assymmetric"):
            if stl_compartments is not None:
                local_topologies = {"left_only", "right_only", "separated"}
                if stl_topology not in local_topologies:
                    global_state = get_stl_global_liquid_topology(
                        V_reservoir,
                        tilt_z,
                        tilt_y,
                        reservoir_geometry,
                    )
                    stl_topology = global_state["topology"]
                    if stl_topology in local_topologies:
                        stl_local_volumes = [
                            global_state["left_volume"],
                            global_state["right_volume"],
                        ]
                    else:
                        stl_local_volumes = None

                if stl_topology in local_topologies:
                    left_local_volume, right_local_volume = stl_local_volumes
                    (
                        left_local_volume,
                        right_local_volume,
                        Q1,
                        final_pool_state,
                    ) = solve_stl_separated_pool_volume_step(
                        V_reservoir,
                        left_local_volume,
                        right_local_volume,
                        tilt_z,
                        tilt_y,
                        reservoir_geometry,
                        stl_compartments,
                        R1,
                        cap_force,
                        time_step,
                        use_wet_dry_gating,
                        wetting_solver=stl_wetting_solver,
                    )
                    h_left = final_pool_state["left_head"]
                    h_right = final_pool_state["right_head"]
                    raw_driving_head = (
                        final_pool_state["right_global_height"]
                        - final_pool_state["left_global_height"]
                    )
                    stl_topology = classify_stl_local_pool_topology(
                        final_pool_state,
                        left_local_volume,
                        right_local_volume,
                    )
                    if stl_topology in local_topologies:
                        stl_local_volumes = [left_local_volume, right_local_volume]
                    else:
                        # A merged pool or a liquid body that no longer wets a
                        # port has no meaningful local-volume split.
                        stl_local_volumes = None
                    pools_connected = stl_topology == "connected"
                else:
                    # A free body and a connected body use one global surface.
                    # Neither has a pressure difference that can drive the
                    # always-filled external channel.
                    left_local_volume = np.nan
                    right_local_volume = np.nan
                    h_left = global_state["left_head"]
                    h_right = global_state["right_head"]
                    raw_driving_head = 0.0
                    Q1 = 0.0
                    pools_connected = stl_topology == "connected"
                stl_pressure_difference_pa = 0.0
                stl_smoothed_equilibrium_left_volume = None
            else:
                (
                    _single_reservoir_compartments,
                    left_local_volume,
                    right_local_volume,
                    single_reservoir_volume_offset,
                ) = get_single_reservoir_local_volumes(
                    V_reservoir,
                    tilt_z,
                    tilt_y,
                    reservoir_geometry,
                    single_reservoir_volume_offset,
                )
                h_left, h_right, single_reservoir_volume_offset = get_single_reservoir_feedback_heads(
                    V_reservoir,
                    tilt_z,
                    tilt_y,
                    reservoir_geometry,
                    single_reservoir_volume_offset,
                )
                raw_driving_head = h_right - h_left
                dh_1 = apply_capillary_cutoff(raw_driving_head, h_right, h_left, dh_min)
                Q1 = limit_single_reservoir_flow_by_volume(
                    dh_1 * k1,
                    left_local_volume,
                    right_local_volume,
                    _single_reservoir_compartments,
                    time_step,
                )
                Q1 = limit_single_reservoir_flow_before_pool_merge(
                    Q1,
                    V_reservoir,
                    tilt_z,
                    tilt_y,
                    reservoir_geometry,
                    single_reservoir_volume_offset,
                    time_step,
                )
                single_reservoir_volume_offset -= Q1 * time_step
                (
                    _single_reservoir_compartments,
                    left_local_volume,
                    right_local_volume,
                    single_reservoir_volume_offset,
                ) = get_single_reservoir_local_volumes(
                    V_reservoir,
                    tilt_z,
                    tilt_y,
                    reservoir_geometry,
                    single_reservoir_volume_offset,
                )
            Q2 = 0.0
            reservoir_1_volume.append(V_reservoir)
            reservoir_2_volume.append(0.0)
            reservoir_3_volume.append(0.0)
            reservoir_4_volume.append(0.0)
            left_local_volume_trace.append(left_local_volume)
            right_local_volume_trace.append(right_local_volume)
            left_port_head_trace.append(h_left)
            right_port_head_trace.append(h_right)
            driving_head_trace.append(raw_driving_head)
        else:
            tilt_width = tilt_y if reservoir_geometry["type"] == "profile" else 0.0
            global_tilt_width = tilt_y
            # Reservoir 2 is reflected across the width axis, so its local
            # roll direction is the opposite of reservoir 1's direction.
            reservoir_2_tilt_width = (
                -tilt_y if reservoir_2_geometry["type"] == "profile" else 0.0
            )
            if (
                reservoir_geometry["type"] == "profile"
                and two_reservoir_stl_solver == "implicit"
            ):
                V1, V2, Q1, Q2 = solve_two_reservoir_volume_step_implicit(
                    V1,
                    V_fill,
                    V_max,
                    tilt_z,
                    tilt_width,
                    dh_tilt_1,
                    dh_tilt_2,
                    reservoir_geometry,
                    k1,
                    k2,
                    dh_min,
                    time_step,
                    use_wet_dry_gating=use_wet_dry_gating,
                    reservoir_2_geometry=reservoir_2_geometry,
                    reservoir_2_tilt_width=reservoir_2_tilt_width,
                    use_geometry_port_driving_heads=use_geometry_port_driving_heads,
                    channel_port_positions=channel_port_positions,
                    global_tilt_width=global_tilt_width,
                )
                channel_driving_head_1, channel_driving_head_2 = (
                    calculate_two_reservoir_channel_driving_heads(
                        V1,
                        V2,
                        tilt_z,
                        tilt_width,
                        dh_tilt_1,
                        dh_tilt_2,
                        reservoir_geometry,
                        dh_min,
                        use_cached_heads=True,
                        use_wet_dry_gating=use_wet_dry_gating,
                        reservoir_2_geometry=reservoir_2_geometry,
                        reservoir_2_tilt_width=reservoir_2_tilt_width,
                        use_geometry_port_driving_heads=use_geometry_port_driving_heads,
                        channel_port_positions=channel_port_positions,
                        global_tilt_width=global_tilt_width,
                    )
                )
            else:
                channel_driving_head_1, channel_driving_head_2 = (
                    calculate_two_reservoir_channel_driving_heads(
                        V1,
                        V2,
                        tilt_z,
                        tilt_width,
                        dh_tilt_1,
                        dh_tilt_2,
                        reservoir_geometry,
                        dh_min,
                        use_cached_heads=True,
                        use_wet_dry_gating=use_wet_dry_gating,
                        reservoir_2_geometry=reservoir_2_geometry,
                        reservoir_2_tilt_width=reservoir_2_tilt_width,
                        use_geometry_port_driving_heads=use_geometry_port_driving_heads,
                        channel_port_positions=channel_port_positions,
                        global_tilt_width=global_tilt_width,
                    )
                )
                Q1 = channel_driving_head_1 * k1
                Q2 = channel_driving_head_2 * k2
                V1 = float(
                    np.clip(
                        V1 - (Q1 + Q2) * time_step,
                        max(0.0, V_fill - V_max),
                        min(V_max, V_fill),
                    )
                )
                V2 = float(V_fill - V1)
            # The end-to-end driving heads are distributed linearly along a
            # uniform channel, so the midpoint contribution is one half.
            channel_1_midpoint_driving_pressure = (
                0.5 * density * gravity * channel_driving_head_1 * 1e-3
            )
            channel_2_midpoint_driving_pressure = (
                0.5 * density * gravity * channel_driving_head_2 * 1e-3
            )
            reservoir_1_volume.append(V1)
            reservoir_2_volume.append(V2)
            reservoir_3_volume.append(0.0)
            reservoir_4_volume.append(0.0)
            left_local_volume_trace.append(0.0)
            right_local_volume_trace.append(0.0)
            left_port_head_trace.append(np.nan)
            right_port_head_trace.append(np.nan)
            driving_head_trace.append(np.nan)

        q1_value = -Q1
        q2_value = -Q2
        q3_value = -Q3 if mode == "four_reservoir_tilting" else 0.0
        q4_value = -Q4 if mode == "four_reservoir_tilting" else 0.0

        q_1.append(q1_value)
        q_2.append(q2_value)
        q_3.append(q3_value)
        q_4.append(q4_value)
        wall_shear_stress.append(calculate_wss_from_q1(q1_value, geometry1))
        pool_connected_trace.append(pools_connected)
        channel_1_midpoint_driving_pressure_trace.append(
            channel_1_midpoint_driving_pressure
        )
        channel_2_midpoint_driving_pressure_trace.append(
            channel_2_midpoint_driving_pressure
        )

        if progress_callback and (
            index == 1
            or index == point_count
            or index % progress_update_every == 0
        ):
            progress_callback(
                progress_base + index,
                progress_total,
                progress_message,
            )

    return (
        np.array(q_1),
        np.array(q_2),
        np.array(q_3),
        np.array(q_4),
        np.array(wall_shear_stress),
        np.array(reservoir_1_volume),
        np.array(reservoir_2_volume),
        np.array(reservoir_3_volume),
        np.array(reservoir_4_volume),
        np.array(left_local_volume_trace),
        np.array(right_local_volume_trace),
        np.array(left_port_head_trace),
        np.array(right_port_head_trace),
        np.array(driving_head_trace),
        np.array(pool_connected_trace, dtype=bool),
        np.array(channel_1_midpoint_driving_pressure_trace),
        np.array(channel_2_midpoint_driving_pressure_trace),
    )


def compute_nested_four_reservoir_loop_data(
    stop_time,
    tilt,
    speed1,
    time_step,
    inner_reservoir_geometry,
    outer_reservoir_geometry,
    inner_channel_geometry,
    outer_channel_geometry,
    cap_force,
    use_wet_dry_gating,
    stl_wetting_solver,
    two_reservoir_stl_solver,
    stl_wetting_volume_bins,
    stl_wetting_tilt_step_deg,
    top_contact_wetting,
    use_geometry_port_driving_heads=True,
    progress_callback=None,
):
    """Solve independent inner and outer two-reservoir recirculating loops.

    The loops share the prescribed tilt only.  Each loop has its own two
    reservoir volumes, two equal channels, and fluidic resistance, so liquid
    cannot be transferred between the inner and outer circuits.
    """
    inner_length = float(inner_reservoir_geometry["length"])
    outer_length = float(outer_reservoir_geometry["length"])
    if outer_length <= inner_length:
        raise ValueError(
            "The outer reservoir length must be larger than the inner reservoir length."
        )
    if float(outer_channel_geometry["length"]) <= float(inner_channel_geometry["length"]):
        raise ValueError(
            "The outer channel length must be larger than the inner channel length."
        )

    if progress_callback:
        progress_callback(0, 2, "Solving inner recirculating loop...")
    inner_data = compute_plot_data(
        stop_time,
        mode="symmetric",
        tilt=tilt,
        speed1=speed1,
        time_step=time_step,
        reservoir_geometry=inner_reservoir_geometry,
        primary_geometry=inner_channel_geometry,
        cap_force=cap_force,
        use_wet_dry_gating=use_wet_dry_gating,
        stl_wetting_solver=stl_wetting_solver,
        two_reservoir_stl_solver=two_reservoir_stl_solver,
        stl_wetting_volume_bins=stl_wetting_volume_bins,
        stl_wetting_tilt_step_deg=stl_wetting_tilt_step_deg,
        top_contact_wetting=top_contact_wetting,
        use_geometry_port_driving_heads=use_geometry_port_driving_heads,
    )
    if progress_callback:
        progress_callback(1, 2, "Solving outer recirculating loop...")
    outer_data = compute_plot_data(
        stop_time,
        mode="symmetric",
        tilt=tilt,
        speed1=speed1,
        time_step=time_step,
        reservoir_geometry=outer_reservoir_geometry,
        primary_geometry=outer_channel_geometry,
        cap_force=cap_force,
        use_wet_dry_gating=use_wet_dry_gating,
        stl_wetting_solver=stl_wetting_solver,
        two_reservoir_stl_solver=two_reservoir_stl_solver,
        stl_wetting_volume_bins=stl_wetting_volume_bins,
        stl_wetting_tilt_step_deg=stl_wetting_tilt_step_deg,
        top_contact_wetting=top_contact_wetting,
        use_geometry_port_driving_heads=use_geometry_port_driving_heads,
    )
    if progress_callback:
        progress_callback(2, 2, "Nested-loop simulation complete.")

    def loop_reservoir_placements(reservoir_geometry, channel_geometry):
        """Return the geometry and centered positions of the V1/V2 loop pair."""
        animation_geometry = build_animation_reservoir_geometry(reservoir_geometry)
        mirrored_geometry = mirror_animation_reservoir_across_width(animation_geometry)
        port_offsets = []
        for port_name in ("left", "right"):
            mirrored_port = get_animation_port_coordinates(mirrored_geometry, port_name)
            original_port = get_animation_port_coordinates(animation_geometry, port_name)
            port_offsets.append(
                float(channel_geometry["length"])
                + mirrored_port[1]
                - original_port[1]
            )
        x_offset = -0.5 * get_animation_plot_extents(animation_geometry)[0]
        separation = 0.5 * float(np.mean(port_offsets))
        return (
            animation_geometry,
            mirrored_geometry,
            (x_offset, separation),
            (x_offset, -separation),
        )

    def port_elevation(
        volume,
        tilt_length,
        tilt_width,
        reservoir_geometry,
        placement_geometry,
        position,
        tilt_width_sign=1.0,
    ):
        """Return water-table elevations relative to the circuit centre."""
        reservoir_tilt_width = (
            float(tilt_width_sign) * tilt_width
            if reservoir_geometry["type"] == "profile"
            else 0.0
        )
        inlet_head, outlet_head = get_oriented_port_heads(
            volume,
            tilt_length,
            reservoir_tilt_width,
            reservoir_geometry,
        )
        slope_length, slope_width, vertical_scale = get_stl_gravity_plane(
            tilt_length, tilt_width
        )
        vertical_vector = np.array(
            [
                slope_length * vertical_scale,
                slope_width * vertical_scale,
                vertical_scale,
            ],
            dtype=float,
        )
        elevations = []
        wet_ports = []
        for port_name, head in (("left", inlet_head), ("right", outlet_head)):
            port = placement_geometry[f"{port_name}_port"]
            width_position = float(
                placement_geometry.get(
                    f"{port_name}_port_width",
                    float(placement_geometry["width"]) / 2.0,
                )
            )
            port_position = np.array(
                [
                    float(position[0]) + float(port["x"]),
                    float(position[1]) + width_position,
                    float(position[2]) + float(port["y"]),
                ],
                dtype=float,
            )
            # STL heads are already gravity-aligned; analytic heads are local-z depths.
            vertical_head = (
                float(head)
                if uses_stl_footprint_partition(reservoir_geometry)
                else float(head) * vertical_scale
            )
            elevations.append(
                float(np.dot(vertical_vector, port_position - circuit_reference) + vertical_head)
            )
            # A dry port has no liquid column, so its free-surface elevation
            # must not be interpreted as a hydrostatic pressure reference.
            wet_ports.append(float(head) > 1e-9)
        return elevations, wet_ports, (float(inlet_head), float(outlet_head))

    (
        inner_animation_geometry,
        inner_mirrored_geometry,
        inner_v1_position,
        inner_v2_position,
    ) = loop_reservoir_placements(
        inner_reservoir_geometry, inner_channel_geometry
    )
    (
        outer_animation_geometry,
        outer_mirrored_geometry,
        outer_v3_position,
        outer_v4_position,
    ) = loop_reservoir_placements(
        outer_reservoir_geometry, outer_channel_geometry
    )
    # Keep an analytic rectangular reservoir analytic for the port-head
    # calculation. The mirrored animation geometry is profile-based and is
    # only needed for placement and drawing.
    inner_mirrored_computation_geometry = mirror_reservoir_geometry_across_width(
        inner_reservoir_geometry
    )
    outer_mirrored_computation_geometry = mirror_reservoir_geometry_across_width(
        outer_reservoir_geometry
    )

    def placed_geometry_points(geometry, position):
        """Return reservoir shell points in the common nested-loop frame."""
        shell_points = np.asarray(geometry["shell_triangles"], dtype=float).reshape(-1, 3)
        return shell_points + np.asarray(position, dtype=float)

    # Rebase every body around the combined shell bounding-box centre.  This
    # makes the fixed common rotation reference explicitly (0, 0, 0) mm.
    circuit_points = np.vstack(
        (
            placed_geometry_points(inner_animation_geometry, (*inner_v1_position, 0.0)),
            placed_geometry_points(inner_mirrored_geometry, (*inner_v2_position, 0.0)),
            placed_geometry_points(outer_animation_geometry, (*outer_v3_position, 0.0)),
            placed_geometry_points(outer_mirrored_geometry, (*outer_v4_position, 0.0)),
        )
    )
    circuit_layout_centre = 0.5 * (
        np.min(circuit_points, axis=0) + np.max(circuit_points, axis=0)
    )

    def rebase_position(position):
        return tuple(np.asarray((*position, 0.0), dtype=float) - circuit_layout_centre)

    inner_v1_position = rebase_position(inner_v1_position)
    inner_v2_position = rebase_position(inner_v2_position)
    outer_v3_position = rebase_position(outer_v3_position)
    outer_v4_position = rebase_position(outer_v4_position)
    circuit_reference = np.zeros(3, dtype=float)

    inlet_difference = []
    outlet_difference = []
    inner_inlet_pressure = []
    inner_outlet_pressure = []
    outer_inlet_pressure = []
    outer_outlet_pressure = []
    channel_1_pressure_difference = []
    channel_2_pressure_difference = []
    channel_1_midpoint_pressure_difference = []
    channel_2_midpoint_pressure_difference = []
    for index, (tilt_y_degrees, tilt_z_degrees) in enumerate(
        zip(inner_data["tilt_y"], inner_data["tilt_z"])
    ):
        tilt_length = math.radians(float(tilt_z_degrees))
        tilt_width = math.radians(float(tilt_y_degrees))
        (inner_v1_left, inner_v1_right), inner_v1_wet, _inner_v1_heads = port_elevation(
            inner_data["reservoir_1_volume_ul"][index],
            tilt_length,
            tilt_width,
            inner_reservoir_geometry,
            inner_animation_geometry,
            inner_v1_position,
        )
        (outer_v3_left, outer_v3_right), outer_v3_wet, _outer_v3_heads = port_elevation(
            outer_data["reservoir_1_volume_ul"][index],
            tilt_length,
            tilt_width,
            outer_reservoir_geometry,
            outer_animation_geometry,
            outer_v3_position,
        )
        (inner_v2_left, inner_v2_right), inner_v2_wet, _inner_v2_heads = port_elevation(
            inner_data["reservoir_2_volume_ul"][index],
            tilt_length,
            tilt_width,
            inner_mirrored_computation_geometry,
            inner_mirrored_geometry,
            inner_v2_position,
            tilt_width_sign=-1.0,
        )
        (outer_v4_left, outer_v4_right), outer_v4_wet, _outer_v4_heads = port_elevation(
            outer_data["reservoir_2_volume_ul"][index],
            tilt_length,
            tilt_width,
            outer_mirrored_computation_geometry,
            outer_mirrored_geometry,
            outer_v4_position,
            tilt_width_sign=-1.0,
        )
        inlet_difference.append(
            outer_v4_left - inner_v2_left
            if inner_v2_wet[0] and outer_v4_wet[0]
            else np.nan
        )
        outlet_difference.append(
            outer_v3_right - inner_v1_right
            if inner_v1_wet[1] and outer_v3_wet[1]
            else np.nan
        )
        # Signed water-table pressure potentials relative to the circuit centre.
        # A dry inlet has no liquid column, so it is intentionally omitted.
        inner_inlet_pressure.append(
            density * gravity * inner_v2_left * 1e-3 if inner_v2_wet[0] else np.nan
        )
        inner_outlet_pressure.append(
            density * gravity * inner_v1_right * 1e-3 if inner_v1_wet[1] else np.nan
        )
        outer_inlet_pressure.append(
            density * gravity * outer_v4_left * 1e-3 if outer_v4_wet[0] else np.nan
        )
        outer_outlet_pressure.append(
            density * gravity * outer_v3_right * 1e-3 if outer_v3_wet[1] else np.nan
        )
        channel_1_pressure_difference.append(
            density * gravity * (outer_v3_right - inner_v1_right) * 1e-3
            if inner_v1_wet[1] and outer_v3_wet[1]
            else np.nan
        )
        channel_2_pressure_difference.append(
            density * gravity * (outer_v4_left - inner_v2_left) * 1e-3
            if inner_v2_wet[0] and outer_v4_wet[0]
            else np.nan
        )
        channel_1_midpoint_pressure_difference.append(
            0.5 * density * gravity
            * ((outer_v3_right + outer_v4_right) - (inner_v1_right + inner_v2_right))
            * 1e-3
            if all((inner_v1_wet[1], inner_v2_wet[1], outer_v3_wet[1], outer_v4_wet[1]))
            else np.nan
        )
        channel_2_midpoint_pressure_difference.append(
            0.5 * density * gravity
            * ((outer_v3_left + outer_v4_left) - (inner_v1_left + inner_v2_left))
            * 1e-3
            if all((inner_v1_wet[0], inner_v2_wet[0], outer_v3_wet[0], outer_v4_wet[0]))
            else np.nan
        )

    inner_case = inner_data["selected_case"]
    outer_case = outer_data["selected_case"]
    selected_case = {
        "mode": "nested_four_reservoir_loops",
        "title": default_mode_settings["nested_four_reservoir_loops"]["title"],
        "speed1": speed1,
        "speed2": None,
        "slow_start": None,
        "slow_duration": None,
        "R1": inner_case["R1"],
        "R2": inner_case["R2"],
        "R3": outer_case["R1"],
        "R4": outer_case["R2"],
        "geometry1": inner_case["geometry1"],
        "geometry2": inner_case["geometry2"],
        "geometry3": outer_case["geometry1"],
        "geometry4": outer_case["geometry2"],
        "inner_reservoir_geometry": inner_reservoir_geometry,
        "outer_reservoir_geometry": outer_reservoir_geometry,
        "circuit_reference_mm": tuple(float(value) for value in circuit_reference),
    }
    return {
        "stop_time": stop_time,
        "time_step_s": time_step,
        "time_points": inner_data["time_points"],
        "selected_case": selected_case,
        "tilt_y": inner_data["tilt_y"],
        "tilt_z": inner_data["tilt_z"],
        "q_1": inner_data["q_1"],
        "q_2": inner_data["q_2"],
        "q_3": outer_data["q_1"],
        "q_4": outer_data["q_2"],
        "wall_shear_stress": inner_data["wall_shear_stress"],
        "reservoir_1_volume_ul": inner_data["reservoir_1_volume_ul"],
        "reservoir_2_volume_ul": inner_data["reservoir_2_volume_ul"],
        "reservoir_3_volume_ul": outer_data["reservoir_1_volume_ul"],
        "reservoir_4_volume_ul": outer_data["reservoir_2_volume_ul"],
        "left_local_volume_ul": inner_data["left_local_volume_ul"],
        "right_local_volume_ul": inner_data["right_local_volume_ul"],
        "left_port_head_mm": inner_data["left_port_head_mm"],
        "right_port_head_mm": inner_data["right_port_head_mm"],
        "driving_head_mm": inner_data["driving_head_mm"],
        "pools_connected": np.logical_or(
            inner_data["pools_connected"], outer_data["pools_connected"]
        ),
        "channel_1_midpoint_driving_pressure_pa": inner_data[
            "channel_1_midpoint_driving_pressure_pa"
        ],
        "channel_2_midpoint_driving_pressure_pa": inner_data[
            "channel_2_midpoint_driving_pressure_pa"
        ],
        "inner_outer_inlet_hydrostatic_difference_mm": np.asarray(
            inlet_difference, dtype=float
        ),
        "inner_outer_outlet_hydrostatic_difference_mm": np.asarray(
            outlet_difference, dtype=float
        ),
        "inner_inlet_hydrostatic_pressure_pa": np.asarray(inner_inlet_pressure, dtype=float),
        "inner_outlet_hydrostatic_pressure_pa": np.asarray(inner_outlet_pressure, dtype=float),
        "outer_inlet_hydrostatic_pressure_pa": np.asarray(outer_inlet_pressure, dtype=float),
        "outer_outlet_hydrostatic_pressure_pa": np.asarray(outer_outlet_pressure, dtype=float),
        "inner_outer_channel_1_pressure_difference_pa": np.asarray(
            channel_1_pressure_difference, dtype=float
        ),
        "inner_outer_channel_2_pressure_difference_pa": np.asarray(
            channel_2_pressure_difference, dtype=float
        ),
        "inner_outer_channel_1_midpoint_pressure_difference_pa": np.asarray(
            channel_1_midpoint_pressure_difference, dtype=float
        ),
        "inner_outer_channel_2_midpoint_pressure_difference_pa": np.asarray(
            channel_2_midpoint_pressure_difference, dtype=float
        ),
    }


def compute_plot_data(
    stop_time,
    mode="symmetric",
    tilt=default_tilt,
    speed1=0.3,
    time_step=default_time_step,
    reservoir_geometry=None,
    primary_geometry=None,
    cap_force=default_cap_force,
    speed2=None,
    slow_start=None,
    slow_duration=None,
    secondary_geometry=None,
    use_wet_dry_gating=True,
    stl_wetting_solver="predictor_corrector",
    two_reservoir_stl_solver="cached",
    stl_wetting_volume_bins=default_stl_wetting_volume_bins,
    stl_wetting_tilt_step_deg=default_stl_wetting_tilt_step_deg,
    top_contact_wetting=False,
    use_geometry_port_driving_heads=True,
    progress_callback=None,
    outer_reservoir_geometry=None,
    outer_channel_geometry=None,
):
    if reservoir_geometry is None:
        reservoir_geometry = build_reservoir_geometry(**default_reservoir_settings)
    if primary_geometry is None:
        primary_geometry = build_geometry(**default_geometry_settings["primary"])
    if mode == "nested_four_reservoir_loops":
        if outer_reservoir_geometry is None or outer_channel_geometry is None:
            raise ValueError(
                "Nested four-reservoir mode requires both an outer reservoir and outer channel geometry."
            )
        return compute_nested_four_reservoir_loop_data(
            stop_time,
            tilt,
            speed1,
            time_step,
            reservoir_geometry,
            outer_reservoir_geometry,
            primary_geometry,
            outer_channel_geometry,
            cap_force,
            use_wet_dry_gating,
            stl_wetting_solver,
            two_reservoir_stl_solver,
            stl_wetting_volume_bins,
            stl_wetting_tilt_step_deg,
            top_contact_wetting,
            use_geometry_port_driving_heads=use_geometry_port_driving_heads,
            progress_callback=progress_callback,
        )
    time_points = build_time_vector(stop_time, time_step=time_step)
    if (mode == "assymmetric") and (secondary_geometry is None):
        secondary_geometry = build_geometry(**default_geometry_settings["secondary"])
    if stl_wetting_solver not in ("cached", "predictor_corrector", "implicit"):
        raise ValueError(
            "STL wetting solver must be 'cached' or 'predictor_corrector'."
        )
    if two_reservoir_stl_solver not in ("cached", "implicit"):
        raise ValueError("Two-reservoir STL solver must be 'cached' or 'implicit'.")
    reservoir_geometry["top_contact_wetting"] = bool(
        top_contact_wetting and reservoir_geometry["type"] == "profile"
    )
    reservoir_geometry["top_contact_wetting_for_two_reservoir"] = bool(
        top_contact_wetting
        and reservoir_geometry["type"] == "profile"
        and mode in ("symmetric", "assymmetric")
    )
    configure_stl_wetting_cache(
        reservoir_geometry,
        stl_wetting_volume_bins,
        stl_wetting_tilt_step_deg,
    )

    V_fill, R1, R2, primary_geometry, secondary_geometry = get_simulation_setup(
        mode,
        reservoir_geometry,
        primary_geometry,
        secondary_geometry,
    )
    selected_case = build_case(
        mode,
        speed1,
        R1,
        R2,
        primary_geometry,
        secondary_geometry,
        speed2=speed2,
        slow_start=slow_start,
        slow_duration=slow_duration,
    )
    total_work = len(time_points) * 2

    if progress_callback:
        progress_callback(0, total_work, "Preparing figure...")

    tilt_y, tilt_z = getTiltFunc(
        time_points,
        selected_case["mode"],
        selected_case["speed1"],
        tilt,
        speed2=selected_case["speed2"],
        slow_start=selected_case["slow_start"],
        slow_duration=selected_case["slow_duration"],
        progress_callback=progress_callback,
        progress_base=0,
        progress_total=total_work,
    )
    (
        q_1,
        q_2,
        q_3,
        q_4,
        wall_shear_stress,
        reservoir_1_volume,
        reservoir_2_volume,
        reservoir_3_volume,
        reservoir_4_volume,
        left_local_volume,
        right_local_volume,
        left_port_head,
        right_port_head,
        driving_head,
        pool_connected,
        channel_1_midpoint_driving_pressure,
        channel_2_midpoint_driving_pressure,
    ) = simulate(
        time_points,
        selected_case["mode"],
        selected_case["speed1"],
        tilt,
        WSS_ref,
        selected_case["R1"],
        V_fill,
        reservoir_geometry,
        selected_case["geometry1"],
        cap_force=cap_force,
        speed2=selected_case["speed2"],
        slow_start=selected_case["slow_start"],
        slow_duration=selected_case["slow_duration"],
        R2=selected_case["R2"],
        geometry2=selected_case["geometry2"],
        use_geometry_port_driving_heads=use_geometry_port_driving_heads,
        time_step=time_step,
        use_wet_dry_gating=use_wet_dry_gating,
        stl_wetting_solver=stl_wetting_solver,
        two_reservoir_stl_solver=two_reservoir_stl_solver,
        progress_callback=progress_callback,
        progress_base=len(time_points),
        progress_total=total_work,
        progress_message="Computing flow and WSS...",
    )

    return {
        "stop_time": stop_time,
        "time_step_s": time_step,
        "time_points": time_points,
        "selected_case": selected_case,
        "tilt_y": tilt_y,
        "tilt_z": tilt_z,
        "q_1": q_1,
        "q_2": q_2,
        "q_3": q_3,
        "q_4": q_4,
        "wall_shear_stress": wall_shear_stress,
        "reservoir_1_volume_ul": reservoir_1_volume,
        "reservoir_2_volume_ul": reservoir_2_volume,
        "reservoir_3_volume_ul": reservoir_3_volume,
        "reservoir_4_volume_ul": reservoir_4_volume,
        "left_local_volume_ul": left_local_volume,
        "right_local_volume_ul": right_local_volume,
        "left_port_head_mm": left_port_head,
        "right_port_head_mm": right_port_head,
        "driving_head_mm": driving_head,
        "pools_connected": pool_connected,
        "channel_1_midpoint_driving_pressure_pa": channel_1_midpoint_driving_pressure,
        "channel_2_midpoint_driving_pressure_pa": channel_2_midpoint_driving_pressure,
    }


def shade_connected_pool_intervals(axes, time_points, connected_states, stop_time):
    """Shade each time range in which the imported-reservoir pools touch."""
    time_values = np.asarray(time_points, dtype=float)
    connected = np.asarray(connected_states, dtype=bool)
    if len(time_values) == 0 or len(connected) != len(time_values):
        return False

    has_connected_interval = False
    start_index = None
    for index, is_connected in enumerate(connected):
        if is_connected and start_index is None:
            start_index = index
        is_last_sample = index == len(connected) - 1
        if start_index is not None and (not is_connected or is_last_sample):
            end_index = index if not is_connected else index + 1
            start_time = time_values[start_index]
            end_time = (
                time_values[end_index]
                if end_index < len(time_values)
                else float(stop_time)
            )
            for axis in axes:
                axis.axvspan(start_time, end_time, color="#75b798", alpha=0.18, zorder=0)
            has_connected_interval = True
            start_index = None
    return has_connected_interval


def apply_zero_phase_low_pass_filter(time_points, values, cutoff_hz):
    """Smooth a plotted signal without adding a phase shift to its features."""
    time_values = np.asarray(time_points, dtype=float)
    signal = np.asarray(values, dtype=float)
    if len(signal) < 3:
        return signal.copy()
    sample_step = float(np.median(np.diff(time_values)))
    if not math.isfinite(sample_step) or sample_step <= 0.0:
        raise ValueError("Low-pass filtering requires increasing, evenly spaced time values.")
    nyquist_hz = 0.5 / sample_step
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError("Low-pass cutoff frequency must be greater than zero.")
    if cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"Low-pass cutoff must be below the Nyquist frequency ({nyquist_hz:.4g} Hz)."
        )

    alpha = math.exp(-2.0 * math.pi * cutoff_hz * sample_step)

    def filter_forward(samples):
        filtered = np.empty_like(samples)
        filtered[0] = samples[0]
        for index in range(1, len(samples)):
            filtered[index] = alpha * filtered[index - 1] + (1.0 - alpha) * samples[index]
        return filtered

    return filter_forward(filter_forward(signal)[::-1])[::-1]


def autoscale_y_to_visible_range(axis, start_time, stop_time):
    """Fit y limits to the visible time range, including horizontal references."""
    visible_values = []
    for line in axis.get_lines():
        x_values = np.asarray(line.get_xdata(), dtype=float)
        y_values = np.asarray(line.get_ydata(), dtype=float)
        if len(x_values) != len(y_values) or len(y_values) == 0:
            continue
        # axhline uses axes-relative x coordinates from 0 to 1. Its y value
        # should remain visible regardless of the displayed time interval.
        is_horizontal_reference = (
            len(x_values) == 2
            and np.all((x_values >= 0.0) & (x_values <= 1.0))
            and np.allclose(y_values, y_values[0])
        )
        if is_horizontal_reference:
            candidates = y_values
        else:
            candidates = y_values[(x_values >= start_time) & (x_values <= stop_time)]
        candidates = candidates[np.isfinite(candidates)]
        if len(candidates):
            visible_values.append(candidates)
    if not visible_values:
        return

    values = np.concatenate(visible_values)
    lower_limit = float(np.min(values))
    upper_limit = float(np.max(values))
    span = upper_limit - lower_limit
    if span <= 1e-12:
        padding = max(abs(upper_limit) * 0.08, 1e-6)
    else:
        padding = 0.08 * span
    axis.set_ylim(lower_limit - padding, upper_limit + padding)


def create_figure_from_data(
    plot_data,
    plot_tilts=True,
    plot_q1=True,
    plot_q2=False,
    plot_q3=True,
    plot_q4=True,
    plot_local_volumes=True,
    plot_midpoint_gravity_pressure=False,
    plot_port_heads=False,
    plot_inner_outer_inlet_hydrostatic_difference=False,
    plot_inner_outer_outlet_hydrostatic_difference=False,
    plot_inner_outer_raw_hydrostatic_pressures=False,
    plot_wss=False,
    plot_wss_reference=False,
    reference_wss_pa=default_reference_wss_pa,
    reference_flow_uL_s=default_reference_flow_uL_s,
    flow_low_pass_enabled=False,
    flow_low_pass_cutoff_hz=0.1,
    ring_global_direction_convention=False,
    plot_start_time=0.0,
    save_path=None,
    progress_callback=None,
):
    plot_tilts = bool(plot_tilts)
    plot_q1 = bool(plot_q1)
    plot_q2 = bool(plot_q2)
    plot_q3 = bool(plot_q3)
    plot_q4 = bool(plot_q4)
    plot_local_volumes = bool(plot_local_volumes)
    plot_midpoint_gravity_pressure = bool(plot_midpoint_gravity_pressure)
    plot_port_heads = bool(plot_port_heads)
    plot_inner_outer_inlet_hydrostatic_difference = bool(
        plot_inner_outer_inlet_hydrostatic_difference
    )
    plot_inner_outer_outlet_hydrostatic_difference = bool(
        plot_inner_outer_outlet_hydrostatic_difference
    )
    plot_inner_outer_raw_hydrostatic_pressures = bool(
        plot_inner_outer_raw_hydrostatic_pressures
    )
    plot_wss = bool(plot_wss)
    plot_wss_reference = bool(plot_wss_reference)
    flow_low_pass_enabled = bool(flow_low_pass_enabled)
    ring_global_direction_convention = bool(ring_global_direction_convention)
    is_single_reservoir_mode = plot_data["selected_case"]["mode"] in (
        "single_reservoir",
        "single_reservoir_assymmetric",
        "simple_tilting",
    )
    is_four_reservoir_mode = plot_data["selected_case"]["mode"] in (
        "four_reservoir_tilting",
        "nested_four_reservoir_loops",
    )
    is_four_reservoir_ring_mode = (
        plot_data["selected_case"]["mode"] == "four_reservoir_tilting"
    )
    is_nested_four_reservoir_mode = (
        plot_data["selected_case"]["mode"] == "nested_four_reservoir_loops"
    )
    if is_single_reservoir_mode:
        plot_q2 = False
        plot_q3 = False
        plot_q4 = False
        plot_midpoint_gravity_pressure = False
    else:
        plot_port_heads = False
    if not is_four_reservoir_mode:
        plot_q3 = False
        plot_q4 = False
    if not is_nested_four_reservoir_mode:
        plot_inner_outer_inlet_hydrostatic_difference = False
        plot_inner_outer_outlet_hydrostatic_difference = False
        plot_inner_outer_raw_hydrostatic_pressures = False
    plot_start_time = float(plot_start_time)
    if not 0.0 <= plot_start_time < float(plot_data["stop_time"]):
        raise ValueError("Plot start time must be at least 0 and below the computation time.")
    row_count = (
        int(plot_tilts)
        + (
            int(plot_q1 or plot_q3) + int(plot_q2 or plot_q4)
            if is_four_reservoir_ring_mode
            else int(plot_q1 or plot_q2)
        )
        + (
            int(plot_q3 or plot_q4)
            if is_nested_four_reservoir_mode
            else 0 if is_four_reservoir_ring_mode else int(plot_q3) + int(plot_q4)
        )
        + int(plot_local_volumes)
        + int(plot_midpoint_gravity_pressure)
        + int(plot_port_heads)
        + int(plot_inner_outer_inlet_hydrostatic_difference)
        + int(plot_inner_outer_outlet_hydrostatic_difference)
        + int(plot_inner_outer_raw_hydrostatic_pressures)
        + int(plot_wss)
        + int(plot_wss_reference)
    )
    if row_count < 1:
        raise ValueError("Select at least one quantity to plot.")
    q1_for_plot = np.asarray(plot_data["q_1"], dtype=float)
    q2_for_plot = np.asarray(plot_data["q_2"], dtype=float)
    q3_for_plot = np.asarray(plot_data["q_3"], dtype=float)
    q4_for_plot = np.asarray(plot_data["q_4"], dtype=float)
    if flow_low_pass_enabled:
        q1_for_plot = apply_zero_phase_low_pass_filter(
            plot_data["time_points"], q1_for_plot, flow_low_pass_cutoff_hz
        )
        q2_for_plot = apply_zero_phase_low_pass_filter(
            plot_data["time_points"], q2_for_plot, flow_low_pass_cutoff_hz
        )
        q3_for_plot = apply_zero_phase_low_pass_filter(
            plot_data["time_points"], q3_for_plot, flow_low_pass_cutoff_hz
        )
        q4_for_plot = apply_zero_phase_low_pass_filter(
            plot_data["time_points"], q4_for_plot, flow_low_pass_cutoff_hz
        )
    if is_four_reservoir_ring_mode and ring_global_direction_convention:
        # The solver uses one positive direction around the ring.  For a
        # fixed global x/y view, channels 3 and 4 point in -x and -y instead.
        q3_for_plot = -q3_for_plot
        q4_for_plot = -q4_for_plot
    flow_line_label = (
        f"Low-pass {flow_low_pass_cutoff_hz:g} Hz"
        if flow_low_pass_enabled
        else None
    )
    reference_wss = calculate_reference_wss_from_q1(
        plot_data["q_1"], reference_wss_pa, reference_flow_uL_s
    )
    q1_period_mean, q_period_seconds, q_period_start, q_period_end = calculate_last_period_mean(
        plot_data["time_points"],
        q1_for_plot,
        plot_data["selected_case"],
    )
    q2_period_mean, _, _, _ = calculate_last_period_mean(
        plot_data["time_points"],
        q2_for_plot,
        plot_data["selected_case"],
    )
    q3_period_mean, _, _, _ = calculate_last_period_mean(
        plot_data["time_points"],
        q3_for_plot,
        plot_data["selected_case"],
    )
    q4_period_mean, _, _, _ = calculate_last_period_mean(
        plot_data["time_points"],
        q4_for_plot,
        plot_data["selected_case"],
    )

    fig, axs = plt.subplots(row_count, 1, sharex=True, squeeze=False)
    fig.set_size_inches(7, 3.8 * row_count)

    row_index = 0
    tilt_axis = None
    if plot_tilts:
        tilt_axis = axs[row_index, 0]
        tilt_axis.set_ylabel("tilt [deg]")
        tilt_axis.set_title(plot_data["selected_case"]["title"])
        tilt_axis.plot(plot_data["time_points"], plot_data["tilt_y"], label="pitch")
        tilt_axis.plot(plot_data["time_points"], plot_data["tilt_z"], label="roll")
        tilt_axis.margins(y=0.08)
        row_index += 1
    else:
        axs[0, 0].set_title(plot_data["selected_case"]["title"])

    if not is_four_reservoir_ring_mode and (plot_q1 or plot_q2):
        flow_axes = axs[row_index, 0]
        flow_axes._simulation_series_keys = {
            key for key, enabled in (("q_1", plot_q1), ("q_2", plot_q2)) if enabled
        }
        flow_axes.set_ylabel("Q [uL/s]")
        for channel_name, enabled, series, mean_value, color in (
            ("Channel 1" if not is_single_reservoir_mode else "Flow", plot_q1, q1_for_plot, q1_period_mean, plot_colors[0]),
            ("Channel 2", plot_q2, q2_for_plot, q2_period_mean, plot_colors[1]),
        ):
            if not enabled:
                continue
            line_label = channel_name
            if flow_line_label is not None:
                line_label += f" ({flow_line_label})"
            flow_axes.plot(plot_data["time_points"], series, color=color, label=line_label)
            flow_axes.axhline(
                mean_value,
                color=color,
                linestyle="--",
                label=f"Flow average: {mean_value:.2f}",
            )
        flow_axes.set_xlim(0, plot_data["stop_time"])
        flow_axes.margins(y=0.08)
        row_index += 1

    if is_four_reservoir_ring_mode:
        ring_labels = (
            ("Channel 1 (+x)", "Channel 3 (+x; -Q3)")
            if ring_global_direction_convention
            else ("Channel 1", "Channel 3")
        )
        ring_vertical_labels = (
            ("Channel 2 (+y)", "Channel 4 (+y; -Q4)")
            if ring_global_direction_convention
            else ("Channel 2", "Channel 4")
        )
        for keys, names, enabled, series, means, colors in (
            (("q_1", "q_3"), ring_labels, (plot_q1, plot_q3), (q1_for_plot, q3_for_plot), (q1_period_mean, q3_period_mean), (plot_colors[0], plot_colors[2])),
            (("q_2", "q_4"), ring_vertical_labels, (plot_q2, plot_q4), (q2_for_plot, q4_for_plot), (q2_period_mean, q4_period_mean), (plot_colors[1], plot_colors[3])),
        ):
            if not any(enabled):
                continue
            axis = axs[row_index, 0]
            axis._simulation_series_keys = {key for key, selected in zip(keys, enabled) if selected}
            axis.set_ylabel("Q [uL/s]")
            for name, selected, values, mean_value, color in zip(names, enabled, series, means, colors):
                if selected:
                    axis.plot(plot_data["time_points"], values, color=color, label=name)
                    axis.axhline(
                        mean_value,
                        color=color,
                        linestyle="--",
                        label=f"Flow average: {mean_value:.2f}",
                    )
            axis.margins(y=0.08)
            row_index += 1
    elif is_nested_four_reservoir_mode and (plot_q3 or plot_q4):
        outer_flow_axes = axs[row_index, 0]
        outer_flow_axes._simulation_series_keys = {
            key for key, enabled in (("q_3", plot_q3), ("q_4", plot_q4)) if enabled
        }
        outer_flow_axes.set_ylabel("Q outer [uL/s]")
        for channel_name, enabled, series, mean_value, color in (
            ("Outer channel 1", plot_q3, q3_for_plot, q3_period_mean, plot_colors[2]),
            ("Outer channel 2", plot_q4, q4_for_plot, q4_period_mean, plot_colors[3]),
        ):
            if not enabled:
                continue
            label = channel_name if flow_line_label is None else f"{channel_name} ({flow_line_label})"
            outer_flow_axes.plot(plot_data["time_points"], series, color=color, label=label)
            outer_flow_axes.axhline(
                mean_value,
                color=color,
                linestyle="--",
                label=f"Flow average: {mean_value:.2f}",
            )
        outer_flow_axes.set_xlim(0, plot_data["stop_time"])
        outer_flow_axes.margins(y=0.08)
        row_index += 1
    elif not is_four_reservoir_ring_mode:
        for channel_index, enabled, series, mean_value in (
            (3, plot_q3, q3_for_plot, q3_period_mean),
            (4, plot_q4, q4_for_plot, q4_period_mean),
        ):
            if not enabled:
                continue
            channel_axes = axs[row_index]
            channel_axes[0]._simulation_series_key = f"q_{channel_index}"
            channel_axes[0].set_ylabel(f"Q_{channel_index} [uL/s]")
            channel_plot_kwargs = {"color": plot_colors[channel_index - 1]}
            if flow_line_label is not None:
                channel_plot_kwargs["label"] = flow_line_label
            channel_axes[0].plot(plot_data["time_points"], series, **channel_plot_kwargs)
            channel_axes[0].axhline(
                mean_value,
                color=plot_colors[channel_index - 1],
                linestyle="--",
                label=(
                    f"Last-period mean ({q_period_start:.1f}-{q_period_end:.1f} s, "
                    f"T={q_period_seconds:.2f} s): {mean_value:.2f} uL/s"
                ),
            )
            channel_axes[0].set_xlim(0, plot_data["stop_time"])
            channel_axes[0].margins(y=0.08)
            row_index += 1

    if plot_local_volumes:
        local_volume_axes = axs[row_index]
        if is_four_reservoir_mode:
            local_volume_axes[0].set_ylabel("V reservoir [uL]")
            for reservoir_index, color in enumerate(plot_colors[:4], start=1):
                local_volume_axes[0].plot(
                    plot_data["time_points"],
                    plot_data[f"reservoir_{reservoir_index}_volume_ul"],
                    color=color,
                    label=f"V{reservoir_index}",
                )
        elif is_single_reservoir_mode:
            local_volume_axes[0].set_ylabel("V local [uL]")
            local_volume_axes[0].plot(
                plot_data["time_points"],
                plot_data["left_local_volume_ul"],
                color=plot_colors[1],
                label="V_left_local",
            )
            local_volume_axes[0].plot(
                plot_data["time_points"],
                plot_data["right_local_volume_ul"],
                color=plot_colors[2],
                label="V_right_local",
            )
        else:
            local_volume_axes[0].set_ylabel("V reservoir [uL]")
            local_volume_axes[0].plot(
                plot_data["time_points"],
                plot_data["reservoir_1_volume_ul"],
                color=plot_colors[1],
                label="V1",
            )
            local_volume_axes[0].plot(
                plot_data["time_points"],
                plot_data["reservoir_2_volume_ul"],
                color=plot_colors[2],
                label="V2",
            )
        local_volume_axes[0].set_xlim(0, plot_data["stop_time"])
        local_volume_axes[0].margins(y=0.08)
        local_volume_axes[0].legend(loc="upper right")
        row_index += 1

    if plot_midpoint_gravity_pressure:
        gravity_pressure_axes = axs[row_index]
        gravity_pressure_axes[0].set_ylabel("p drive [Pa]")
        gravity_pressure_axes[0].plot(
            plot_data["time_points"],
            plot_data["channel_1_midpoint_driving_pressure_pa"],
            color=plot_colors[1],
            label="Channel 1 midpoint driving pressure",
        )
        gravity_pressure_axes[0].plot(
            plot_data["time_points"],
            plot_data["channel_2_midpoint_driving_pressure_pa"],
            color=plot_colors[2],
            label="Channel 2 midpoint driving pressure",
        )
        gravity_pressure_axes[0].axhline(0.0, color="0.4", linewidth=0.7)
        gravity_pressure_axes[0].set_xlim(0, plot_data["stop_time"])
        gravity_pressure_axes[0].margins(y=0.08)
        gravity_pressure_axes[0].legend(loc="upper right")
        row_index += 1

    if plot_port_heads:
        head_axes = axs[row_index]
        head_axes[0].set_ylabel("head [mm]")
        head_axes[0].plot(
            plot_data["time_points"],
            plot_data["left_port_head_mm"],
            color=plot_colors[1],
            label="inlet water-table head",
        )
        head_axes[0].plot(
            plot_data["time_points"],
            plot_data["right_port_head_mm"],
            color=plot_colors[2],
            label="outlet water-table head",
        )
        head_axes[0].plot(
            plot_data["time_points"],
            plot_data["driving_head_mm"],
            color=plot_colors[0],
            linestyle="--",
            label="channel-driving head (global free-surface difference)",
        )
        head_axes[0].axhline(0.0, color="0.4", linewidth=0.7)
        head_axes[0].set_xlim(0, plot_data["stop_time"])
        head_axes[0].margins(y=0.08)
        head_axes[0].legend(loc="upper right")
        row_index += 1

    if plot_inner_outer_inlet_hydrostatic_difference:
        hydrostatic_axes = axs[row_index]
        hydrostatic_axes[0].set_ylabel("outer - inner [Pa]")
        hydrostatic_axes[0].plot(
            plot_data["time_points"],
            plot_data["inner_outer_channel_2_midpoint_pressure_difference_pa"],
            color="#457b9d",
            label="Channel 2 midpoint pressure difference",
        )
        hydrostatic_axes[0].axhline(0.0, color="0.4", linewidth=0.7)
        hydrostatic_axes[0].set_xlim(0, plot_data["stop_time"])
        hydrostatic_axes[0].margins(y=0.08)
        hydrostatic_axes[0].legend(loc="upper right")
        row_index += 1

    if plot_inner_outer_outlet_hydrostatic_difference:
        hydrostatic_axes = axs[row_index]
        hydrostatic_axes[0].set_ylabel("outer - inner [Pa]")
        hydrostatic_axes[0].plot(
            plot_data["time_points"],
            plot_data["inner_outer_channel_1_midpoint_pressure_difference_pa"],
            color="#e76f51",
            label="Channel 1 midpoint pressure difference",
        )
        hydrostatic_axes[0].axhline(0.0, color="0.4", linewidth=0.7)
        hydrostatic_axes[0].set_xlim(0, plot_data["stop_time"])
        hydrostatic_axes[0].margins(y=0.08)
        hydrostatic_axes[0].legend(loc="upper right")
        row_index += 1

    if plot_inner_outer_raw_hydrostatic_pressures:
        pressure_axes = axs[row_index]
        pressure_axes[0].set_ylabel("water-table pressure relative to centre [Pa]")
        for key, color, label in (
            ("inner_outlet_hydrostatic_pressure_pa", "#457b9d", "Inner channel 1 inlet"),
            ("outer_outlet_hydrostatic_pressure_pa", "#e76f51", "Outer channel 1 inlet"),
            ("inner_inlet_hydrostatic_pressure_pa", "#2a9d8f", "Inner channel 2 inlet"),
            ("outer_inlet_hydrostatic_pressure_pa", "#9d4edd", "Outer channel 2 inlet"),
        ):
            pressure_axes[0].plot(plot_data["time_points"], plot_data[key], color=color, label=label)
        pressure_axes[0].set_xlim(0, plot_data["stop_time"])
        pressure_axes[0].margins(y=0.08)
        pressure_axes[0].legend(loc="upper right")
        row_index += 1

    if plot_wss:
        wss_axes = axs[row_index]
        wss_axes[0].set_ylabel("WSS [Pa]")
        wss_axes[0].plot(
            plot_data["time_points"], plot_data["wall_shear_stress"], color=plot_colors[1]
        )
        wss_axes[0].axhline(
            np.mean(plot_data["wall_shear_stress"]),
            color=plot_colors[1],
            linestyle="--",
            label=f"{np.mean(plot_data['wall_shear_stress']):.4f} Pa",
        )
        wss_axes[0].set_xlim(0, plot_data["stop_time"])
        wss_axes[0].margins(y=0.08)
        wss_axes[0].legend(loc="upper right")
        row_index += 1

    if plot_wss_reference:
        wss_ref_axes = axs[row_index]
        wss_ref_axes[0].set_ylabel("WSS [Pa]")
        wss_ref_axes[0].plot(plot_data["time_points"], reference_wss, color=plot_colors[2])
        wss_ref_axes[0].axhline(
            np.mean(reference_wss),
            color=plot_colors[2],
            linestyle="--",
            label=f"{np.mean(reference_wss):.4f} Pa",
        )
        wss_ref_axes[0].set_xlim(0, plot_data["stop_time"])
        wss_ref_axes[0].margins(y=0.08)
        wss_ref_axes[0].legend(loc="upper right")

    if is_single_reservoir_mode and shade_connected_pool_intervals(
        axs[:, 0],
        plot_data["time_points"],
        plot_data.get("pools_connected", []),
        plot_data["stop_time"],
    ):
        title_axis = tilt_axis if tilt_axis is not None else axs[0, 0]
        title_axis.set_title(
            f"{plot_data['selected_case']['title']}\nGreen shading: pools connected"
        )

    for axis in axs[:, 0]:
        axis.set_xlim(plot_start_time, plot_data["stop_time"])
        autoscale_y_to_visible_range(axis, plot_start_time, plot_data["stop_time"])
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2,
                borderaxespad=0.0,
            )
    axs[-1, 0].set_xlabel("time [s]")
    fig.tight_layout(pad=2.0, h_pad=4.0)
    for axis in axs[:, 0]:
        title = axis.get_title()
        if title:
            axis.set_title(title, pad=36)

    output_path = None
    if save_path is not None:
        output_path = Path(save_path)
        fig.savefig(output_path, bbox_inches="tight")
        if progress_callback:
            progress_callback(1, 1, "Saving image...")
    elif progress_callback:
        progress_callback(1, 1, "Rendering plot...")

    return fig, output_path


def build_plot_animation_data(plot_data, frame_times):
    """Resample a plot-data dictionary onto the final animation period."""
    source_times = np.asarray(plot_data["time_points"], dtype=float)
    if len(source_times) < 2:
        raise ValueError("Plot animation requires at least two simulation time points.")

    period_seconds = get_actuation_period_seconds(plot_data["selected_case"])
    end_time = float(frame_times[0] + period_seconds)
    # Include the period endpoint for the static axes and reference statistics.
    # GIF frames stay endpoint-free, so playback duration remains one period.
    display_times = np.append(np.asarray(frame_times, dtype=float), end_time)
    animation_data = copy.copy(plot_data)
    animation_data["time_points"] = display_times
    animation_data["stop_time"] = end_time

    for key, values in plot_data.items():
        if key == "time_points":
            continue
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != len(source_times):
            continue
        if np.issubdtype(array.dtype, np.bool_):
            # Connection-state shading is sampled as a discrete state.
            sample_indices = np.searchsorted(source_times, display_times, side="left")
            sample_indices = np.clip(sample_indices, 0, len(source_times) - 1)
            animation_data[key] = array[sample_indices]
        elif np.issubdtype(array.dtype, np.number):
            animation_data[key] = np.interp(
                display_times, source_times, np.asarray(array, dtype=float)
            )
    return animation_data


def save_plot_animation(
    plot_data,
    output_path,
    *,
    plot_tilts=True,
    plot_q1=True,
    plot_q2=True,
    plot_q3=True,
    plot_q4=True,
    plot_local_volumes=False,
    plot_midpoint_gravity_pressure=False,
    plot_port_heads=False,
    plot_inner_outer_inlet_hydrostatic_difference=False,
    plot_inner_outer_outlet_hydrostatic_difference=False,
    plot_inner_outer_raw_hydrostatic_pressures=False,
    plot_wss=False,
    plot_wss_reference=False,
    reference_wss_pa=default_reference_wss_pa,
    reference_flow_uL_s=default_reference_flow_uL_s,
    flow_low_pass_enabled=False,
    flow_low_pass_cutoff_hz=0.1,
    ring_global_direction_convention=False,
    fps=24,
    frame_count=None,
):
    """Save one final actuation period as a progressively drawn plot GIF."""
    frame_times = get_realtime_animation_time_values(
        plot_data, fps=fps, frame_count=frame_count
    )
    animation_data = build_plot_animation_data(plot_data, frame_times)
    figure, _ = create_figure_from_data(
        animation_data,
        plot_tilts=plot_tilts,
        plot_q1=plot_q1,
        plot_q2=plot_q2,
        plot_q3=plot_q3,
        plot_q4=plot_q4,
        plot_local_volumes=plot_local_volumes,
        plot_midpoint_gravity_pressure=plot_midpoint_gravity_pressure,
        plot_port_heads=plot_port_heads,
        plot_inner_outer_inlet_hydrostatic_difference=plot_inner_outer_inlet_hydrostatic_difference,
        plot_inner_outer_outlet_hydrostatic_difference=plot_inner_outer_outlet_hydrostatic_difference,
        plot_inner_outer_raw_hydrostatic_pressures=plot_inner_outer_raw_hydrostatic_pressures,
        plot_wss=plot_wss,
        plot_wss_reference=plot_wss_reference,
        reference_wss_pa=reference_wss_pa,
        reference_flow_uL_s=reference_flow_uL_s,
        flow_low_pass_enabled=flow_low_pass_enabled,
        flow_low_pass_cutoff_hz=flow_low_pass_cutoff_hz,
        ring_global_direction_convention=ring_global_direction_convention,
        plot_start_time=float(frame_times[0]),
    )

    animated_lines = []
    time_count = len(animation_data["time_points"])
    for axis in figure.axes:
        for line in axis.get_lines():
            x_values = np.asarray(line.get_xdata(), dtype=float)
            y_values = np.asarray(line.get_ydata(), dtype=float)
            if (
                len(x_values) == time_count
                and len(y_values) == time_count
                and np.allclose(x_values, animation_data["time_points"])
            ):
                animated_lines.append((line, x_values, y_values))
                line.set_data(x_values[:1], y_values[:1])

    time_markers = [
        axis.axvline(frame_times[0], color="0.25", linewidth=1.0, alpha=0.7)
        for axis in figure.axes
    ]

    def update(frame_index):
        sample_count = frame_index + 1
        for line, x_values, y_values in animated_lines:
            line.set_data(x_values[:sample_count], y_values[:sample_count])
        for marker in time_markers:
            marker.set_xdata([frame_times[frame_index], frame_times[frame_index]])
        return [line for line, _, _ in animated_lines] + time_markers

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_times),
        interval=1000 / fps,
        blit=False,
    )
    output_path = Path(output_path).with_suffix(".gif").resolve()
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=110)
    finally:
        plt.close(figure)
    return output_path


def build_export_columns(
    plot_data,
    plot_q1=True,
    plot_q2=True,
    plot_q3=True,
    plot_q4=True,
    plot_local_volumes=False,
    plot_midpoint_gravity_pressure=False,
    plot_port_heads=False,
    plot_inner_outer_inlet_hydrostatic_difference=False,
    plot_inner_outer_outlet_hydrostatic_difference=False,
    plot_inner_outer_raw_hydrostatic_pressures=False,
    plot_wss=False,
    plot_wss_reference=False,
    reference_wss_pa=default_reference_wss_pa,
    reference_flow_uL_s=default_reference_flow_uL_s,
):
    is_single_reservoir_mode = plot_data["selected_case"]["mode"] in (
        "single_reservoir",
        "single_reservoir_assymmetric",
        "simple_tilting",
    )
    is_four_reservoir_mode = plot_data["selected_case"]["mode"] in (
        "four_reservoir_tilting",
        "nested_four_reservoir_loops",
    )
    is_nested_four_reservoir_mode = (
        plot_data["selected_case"]["mode"] == "nested_four_reservoir_loops"
    )
    if is_single_reservoir_mode:
        plot_q2 = False
        plot_q3 = False
        plot_q4 = False
        plot_midpoint_gravity_pressure = False
    else:
        plot_port_heads = False
    if not is_four_reservoir_mode:
        plot_q3 = False
        plot_q4 = False
    if not is_nested_four_reservoir_mode:
        plot_inner_outer_inlet_hydrostatic_difference = False
        plot_inner_outer_outlet_hydrostatic_difference = False
        plot_inner_outer_raw_hydrostatic_pressures = False
    columns = [
        ("time_s", plot_data["time_points"]),
        ("pitch_deg", plot_data["tilt_y"]),
        ("roll_deg", plot_data["tilt_z"]),
    ]

    if plot_q1:
        columns.append(
            (("Q_uL_per_s" if is_single_reservoir_mode else "Q_1_uL_per_s"), plot_data["q_1"])
        )
    if plot_q2:
        columns.append(("Q_2_uL_per_s", plot_data["q_2"]))
    if plot_q3:
        columns.append(("Q_3_uL_per_s", plot_data["q_3"]))
    if plot_q4:
        columns.append(("Q_4_uL_per_s", plot_data["q_4"]))
    if plot_local_volumes:
        if is_four_reservoir_mode:
            for reservoir_index in range(1, 5):
                columns.append(
                    (
                        f"V{reservoir_index}_uL",
                        plot_data[f"reservoir_{reservoir_index}_volume_ul"],
                    )
                )
        elif is_single_reservoir_mode:
            columns.append(("V_left_local_uL", plot_data["left_local_volume_ul"]))
            columns.append(("V_right_local_uL", plot_data["right_local_volume_ul"]))
        else:
            columns.append(("V1_uL", plot_data["reservoir_1_volume_ul"]))
            columns.append(("V2_uL", plot_data["reservoir_2_volume_ul"]))
    if plot_midpoint_gravity_pressure:
        columns.append(
            (
                "P_drive_channel_1_midpoint_Pa",
                plot_data["channel_1_midpoint_driving_pressure_pa"],
            )
        )
        columns.append(
            (
                "P_drive_channel_2_midpoint_Pa",
                plot_data["channel_2_midpoint_driving_pressure_pa"],
            )
        )
    if plot_port_heads:
        columns.append(("h_inlet_mm", plot_data["left_port_head_mm"]))
        columns.append(("h_outlet_mm", plot_data["right_port_head_mm"]))
        columns.append(("channel_driving_head_mm", plot_data["driving_head_mm"]))
    if plot_inner_outer_inlet_hydrostatic_difference:
        columns.append(
            (
                "pressure_difference_channel_2_inlet_outer_minus_inner_Pa",
                plot_data["inner_outer_channel_2_pressure_difference_pa"],
            )
        )
        columns.append(
            (
                "pressure_difference_channel_2_midpoint_outer_minus_inner_Pa",
                plot_data["inner_outer_channel_2_midpoint_pressure_difference_pa"],
            )
        )
    if plot_inner_outer_outlet_hydrostatic_difference:
        columns.append(
            (
                "pressure_difference_channel_1_inlet_outer_minus_inner_Pa",
                plot_data["inner_outer_channel_1_pressure_difference_pa"],
            )
        )
        columns.append(
            (
                "pressure_difference_channel_1_midpoint_outer_minus_inner_Pa",
                plot_data["inner_outer_channel_1_midpoint_pressure_difference_pa"],
            )
        )
    if plot_inner_outer_raw_hydrostatic_pressures:
        for key, label in (
            ("inner_outlet_hydrostatic_pressure_pa", "P_water_table_inner_channel_1_inlet_relative_centre_Pa"),
            ("outer_outlet_hydrostatic_pressure_pa", "P_water_table_outer_channel_1_inlet_relative_centre_Pa"),
            ("inner_inlet_hydrostatic_pressure_pa", "P_water_table_inner_channel_2_inlet_relative_centre_Pa"),
            ("outer_inlet_hydrostatic_pressure_pa", "P_water_table_outer_channel_2_inlet_relative_centre_Pa"),
        ):
            columns.append((label, plot_data[key]))
    if plot_wss:
        columns.append(("WSS_Pa", plot_data["wall_shear_stress"]))
    if plot_wss_reference:
        columns.append(
            (
                "WSS_reference_Pa",
                calculate_reference_wss_from_q1(
                    plot_data["q_1"],
                    reference_wss_pa,
                    reference_flow_uL_s,
                ),
            )
        )

    return columns


def validate_stop_time(stop_time_text):
    try:
        stop_time = int(stop_time_text)
    except ValueError as exc:
        raise ValueError("Stop time must be an integer between 60 and 600 seconds.") from exc

    if not 60 <= stop_time <= 600:
        raise ValueError("Stop time must be between 60 and 600 seconds.")

    return stop_time


def validate_float_range(value_text, label, minimum, maximum):
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number between {minimum} and {maximum}.") from exc

    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")

    return value


def validate_integer_range(value_text, label, minimum, maximum):
    value = validate_float_range(value_text, label, minimum, maximum)
    if not value.is_integer():
        raise ValueError(f"{label} must be a whole number between {minimum} and {maximum}.")
    return int(value)


def parse_port_coordinates(value_text, label):
    parts = [part.strip() for part in value_text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{label} must be provided as x,y in millimeters.")

    try:
        x_value = float(parts[0])
        y_value = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{label} must be provided as x,y in millimeters.") from exc

    return [x_value, y_value]


def parse_extrusion_axis(axis_text):
    if axis_text is None:
        return None
    axis_value = axis_text.strip().lower()
    if axis_value == "auto":
        return None
    if axis_value not in stl_axis_names:
        raise ValueError("Extrusion axis must be one of auto, x, y, or z.")
    return stl_axis_names.index(axis_value)


def set_figure_window_geometry(figure, left, top, width, height):
    manager = getattr(figure.canvas, "manager", None)
    window = getattr(manager, "window", None)
    if window is None:
        return

    try:
        window.setGeometry(left, top, width, height)
    except Exception:
        return


def format_xyz_coordinates(port_world):
    if port_world is None:
        return "unknown"
    return (
        f"({port_world['x']:.3f}, {port_world['z']:.3f}, {port_world['y']:.3f}) mm"
    )


def build_world_port_coordinates(profile_template, plan_point, height_value):
    world_values = dict(zip(stl_axis_names, profile_template["axis_mins"]))
    top_view_axis_names = profile_template["top_view_axes"]
    profile_axis_names = profile_template["profile_axes"]

    world_values[top_view_axis_names[0]] = profile_template["axis_mins"][
        profile_template["top_view_axis_indices"][0]
    ] + float(plan_point[0])
    world_values[top_view_axis_names[1]] = profile_template["axis_mins"][
        profile_template["top_view_axis_indices"][1]
    ] + float(plan_point[1])
    world_values[profile_axis_names[1]] = profile_template["axis_mins"][
        profile_template["profile_axis_indices"][1]
    ] + float(height_value)

    return {axis_name: float(world_values[axis_name]) for axis_name in stl_axis_names}


def describe_reservoir_geometry(reservoir_geometry):
    if reservoir_geometry["type"] == "profile":
        return (
            f"Imported STL reservoir dimensions in use: "
            f"width={reservoir_geometry['width']:.3f} mm, "
            f"length={reservoir_geometry['length']:.3f} mm, "
            f"height={reservoir_geometry['height']:.3f} mm."
        )

    return (
        f"Rectangular reservoir dimensions in use: "
        f"width={reservoir_geometry['width']:.3f} mm, "
        f"length={reservoir_geometry['length']:.3f} mm, "
        f"height={reservoir_geometry['height']:.3f} mm."
    )


def uses_stl_footprint_partition(reservoir_geometry):
    """Return whether a reservoir has a real imported plan-view footprint."""
    return (
        reservoir_geometry.get("type") == "profile"
        and reservoir_geometry.get("shell_triangles") is not None
        and reservoir_geometry.get("source_path") not in (None, "rectangular-default")
    )


def find_nearest_valid_grid_cell(x_grid, width_grid, valid_mask, port_x, port_width):
    if not np.any(valid_mask):
        raise ValueError("The imported STL footprint has no valid liquid cells.")
    squared_distance = (x_grid - float(port_x)) ** 2 + (width_grid - float(port_width)) ** 2
    squared_distance = np.where(valid_mask, squared_distance, np.inf)
    return tuple(np.unravel_index(np.argmin(squared_distance), squared_distance.shape))


def compute_footprint_geodesic_distances(x_grid, width_grid, valid_mask, start_index):
    """Shortest in-footprint distances without crossing a corner or wall gap."""
    distances = np.full(x_grid.shape, np.inf, dtype=float)
    distances[start_index] = 0.0
    pending = [(0.0, start_index)]
    row_count, column_count = valid_mask.shape

    while pending:
        current_distance, (row_index, column_index) = heapq.heappop(pending)
        if current_distance != distances[row_index, column_index]:
            continue
        for row_step, column_step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row = row_index + row_step
            next_column = column_index + column_step
            if (
                next_row < 0
                or next_row >= row_count
                or next_column < 0
                or next_column >= column_count
                or not valid_mask[next_row, next_column]
            ):
                continue
            step_length = math.hypot(
                float(x_grid[next_row, next_column] - x_grid[row_index, column_index]),
                float(width_grid[next_row, next_column] - width_grid[row_index, column_index]),
            )
            next_distance = current_distance + step_length
            if next_distance < distances[next_row, next_column]:
                distances[next_row, next_column] = next_distance
                heapq.heappush(pending, (next_distance, (next_row, next_column)))
    return distances


def get_port_connected_footprint_partition(
    reservoir_geometry,
    x_grid,
    width_grid,
    valid_mask,
):
    """Partition the footprint by geodesic distance from its selected ports."""
    left_start = find_nearest_valid_grid_cell(
        x_grid,
        width_grid,
        valid_mask,
        reservoir_geometry["left_port"]["x"],
        reservoir_geometry["left_port_width"],
    )
    right_start = find_nearest_valid_grid_cell(
        x_grid,
        width_grid,
        valid_mask,
        reservoir_geometry["right_port"]["x"],
        reservoir_geometry["right_port_width"],
    )
    left_distance = compute_footprint_geodesic_distances(
        x_grid,
        width_grid,
        valid_mask,
        left_start,
    )
    right_distance = compute_footprint_geodesic_distances(
        x_grid,
        width_grid,
        valid_mask,
        right_start,
    )
    if not np.all(np.isfinite(left_distance[valid_mask])) or not np.all(
        np.isfinite(right_distance[valid_mask])
    ):
        raise ValueError(
            "The imported reservoir footprint is disconnected between the selected ports."
        )
    left_region = valid_mask & (left_distance <= right_distance)
    right_region = valid_mask & ~left_region
    return left_distance, right_distance, left_region, right_region


def configure_stl_wetting_cache(
    reservoir_geometry,
    volume_bins=default_stl_wetting_volume_bins,
    tilt_step_deg=default_stl_wetting_tilt_step_deg,
):
    """Store the user-selected resolution for cached STL wetting states."""
    reservoir_geometry["_stl_wetting_cache_settings"] = {
        "volume_bins": max(1, int(volume_bins)),
        "tilt_step_deg": max(float(tilt_step_deg), 1e-6),
    }


def get_stl_wetting_cache_settings(reservoir_geometry):
    settings = reservoir_geometry.get("_stl_wetting_cache_settings", {})
    return (
        max(1, int(settings.get("volume_bins", default_stl_wetting_volume_bins))),
        max(
            float(settings.get("tilt_step_deg", default_stl_wetting_tilt_step_deg)),
            1e-6,
        ),
    )


def get_stl_gravity_plane(signed_tilt_length, signed_tilt_width):
    """Return a world-horizontal free-surface plane in reservoir coordinates.

    The reservoir first pitches about its width axis, then rolls about its
    length axis.  The returned slopes describe the horizontal laboratory-frame
    water table in local coordinates; ``vertical_scale`` converts a local
    depth below that plane into a real vertical pressure head.
    """
    pitch = float(signed_tilt_length)
    roll = float(signed_tilt_width)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)

    # R = R_x(roll) R_y(-pitch), so R.T [0, 0, 1] is the laboratory vertical
    # expressed in the rotating reservoir frame.
    vertical_in_local = np.array(
        [sin_pitch * cos_roll, sin_roll, cos_pitch * cos_roll], dtype=float
    )
    vertical_scale = float(vertical_in_local[2])
    if vertical_scale <= 1e-9:
        raise ValueError("Reservoir rotation cannot reach a vertical orientation.")
    return (
        float(vertical_in_local[0] / vertical_scale),
        float(vertical_in_local[1] / vertical_scale),
        vertical_scale,
    )


def get_stl_footprint_context(reservoir_geometry):
    cached_context = reservoir_geometry.get("_stl_footprint_context")
    if cached_context is not None:
        return cached_context

    context = build_animation_surface_context(
        reservoir_geometry,
        x_count=70,
        width_count=50,
    )
    valid_mask = (
        context["point_mask"]
        & np.isfinite(context["bottom_grid"])
        & np.isfinite(context["top_grid"])
    )
    (
        left_distance,
        right_distance,
        left_region,
        right_region,
    ) = get_port_connected_footprint_partition(
        reservoir_geometry,
        context["surface_x_grid"],
        context["surface_width_grid"],
        valid_mask,
    )
    cell_area = float(context["surface_cell_area"])
    cell_capacity = np.where(
        valid_mask,
        np.maximum(0.0, context["top_grid"] - context["bottom_grid"]),
        0.0,
    ) * cell_area
    context.update(
        {
            "valid_mask": valid_mask,
            "left_distance": left_distance,
            "right_distance": right_distance,
            "left_region": left_region,
            "right_region": right_region,
            "total_capacity_ul": float(np.sum(cell_capacity)),
            "equilibrium_cache": {},
        }
    )
    # The distance regions are retained only to define the initial equilibrium
    # split.  Each dry pool itself may grow through any port-connected wet path.
    context["left_capacity_ul"] = context["total_capacity_ul"]
    context["right_capacity_ul"] = context["total_capacity_ul"]
    reservoir_geometry["_stl_footprint_context"] = context
    return context


def get_stl_footprint_equilibrium(
    total_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
):
    """Return port heads and port-connected equilibrium volumes for an STL."""
    context = get_stl_footprint_context(reservoir_geometry)
    top_contact_wetting = bool(
        reservoir_geometry.get("top_contact_wetting_for_two_reservoir", False)
    )
    capacity = context["total_capacity_ul"]
    target_volume = float(np.clip(total_volume, 0.0, capacity))
    volume_bin_count, tilt_step_deg = get_stl_wetting_cache_settings(
        reservoir_geometry
    )
    cache_resolution = max(capacity / volume_bin_count, 0.1)
    tilt_resolution = math.radians(tilt_step_deg)
    cache_key = (
        volume_bin_count,
        tilt_step_deg,
        top_contact_wetting,
        int(round(float(total_volume) / cache_resolution)),
        int(round(float(signed_tilt_length) / tilt_resolution)),
        int(round(float(signed_tilt_width) / tilt_resolution)),
    )
    cached_state = context["equilibrium_cache"].get(cache_key)
    if cached_state is not None:
        return {
            "left_head": cached_state["left_head"],
            "right_head": cached_state["right_head"],
            "left_volume": target_volume * cached_state["left_fraction"],
            "right_volume": target_volume * cached_state["right_fraction"],
        }

    quantized_volume = float(np.clip(cache_key[3] * cache_resolution, 0.0, capacity))
    quantized_tilt_length = cache_key[4] * tilt_resolution
    quantized_tilt_width = cache_key[5] * tilt_resolution
    slope_length, slope_width, vertical_scale = get_stl_gravity_plane(
        quantized_tilt_length,
        quantized_tilt_width,
    )
    intercept = solve_animation_free_surface_3d(
        context,
        quantized_volume,
        slope_length,
        slope_width,
    )
    liquid_top = np.minimum(
        intercept - slope_length * context["surface_x_grid"] - slope_width * context["surface_width_grid"],
        context["top_grid"],
    )
    liquid_height = np.where(
        context["valid_mask"],
        np.maximum(0.0, liquid_top - context["bottom_grid"]),
        0.0,
    )
    cell_volume = liquid_height * float(context["surface_cell_area"])
    equilibrium_left = float(np.sum(cell_volume[context["left_region"]]))
    equilibrium_right = float(np.sum(cell_volume[context["right_region"]]))
    equilibrium_total = equilibrium_left + equilibrium_right
    if equilibrium_total > 0.0:
        scale_factor = quantized_volume / equilibrium_total
        equilibrium_left *= scale_factor
        equilibrium_right *= scale_factor

    left_port = reservoir_geometry["left_port"]
    right_port = reservoir_geometry["right_port"]
    left_head = vertical_scale * max(
        0.0,
        float(
            intercept
            - slope_length * left_port["x"]
            - slope_width * reservoir_geometry["left_port_width"]
            - left_port["y"]
        ),
    )
    right_head = vertical_scale * max(
        0.0,
        float(
            intercept
            - slope_length * right_port["x"]
            - slope_width * reservoir_geometry["right_port_width"]
            - right_port["y"]
        ),
    )
    if top_contact_wetting:
        # In the two-reservoir model, split the local reservoir volume using
        # the free-surface equilibrium, then apply the same port-anchored
        # top-contact rule as the one-reservoir model to both port pools.
        left_intercept, left_wet, left_depth = solve_port_connected_free_surface_3d(
            context,
            equilibrium_left,
            slope_length,
            slope_width,
            left_port,
            reservoir_geometry["left_port_width"],
            top_contact_wetting=True,
        )
        right_intercept, right_wet, right_depth = solve_port_connected_free_surface_3d(
            context,
            equilibrium_right,
            slope_length,
            slope_width,
            right_port,
            reservoir_geometry["right_port_width"],
            top_contact_wetting=True,
        )
        if not wet_regions_touch(left_wet, right_wet):
            left_head = vertical_scale * max(
                0.0,
                float(
                    left_intercept
                    - slope_length * left_port["x"]
                    - slope_width * reservoir_geometry["left_port_width"]
                    - left_port["y"]
                ),
            )
            right_head = vertical_scale * max(
                0.0,
                float(
                    right_intercept
                    - slope_length * right_port["x"]
                    - slope_width * reservoir_geometry["right_port_width"]
                    - right_port["y"]
                ),
            )
    if quantized_volume > 0.0:
        left_fraction = equilibrium_left / quantized_volume
    else:
        left_fraction = 0.5
    state = {
        "left_head": left_head,
        "right_head": right_head,
        "left_fraction": float(np.clip(left_fraction, 0.0, 1.0)),
        "right_fraction": float(np.clip(1.0 - left_fraction, 0.0, 1.0)),
    }
    context["equilibrium_cache"][cache_key] = state
    return {
        "left_head": left_head,
        "right_head": right_head,
        "left_volume": target_volume * state["left_fraction"],
        "right_volume": target_volume * state["right_fraction"],
    }


def get_stl_global_liquid_topology(
    total_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
):
    """Classify a global STL free surface without imposing a local-volume cut."""
    context = get_stl_footprint_context(reservoir_geometry)
    capacity = float(context["total_capacity_ul"])
    target_volume = float(np.clip(total_volume, 0.0, capacity))
    volume_bin_count, tilt_step_deg = get_stl_wetting_cache_settings(
        reservoir_geometry
    )
    volume_step = max(capacity / volume_bin_count, 0.1)
    tilt_step = math.radians(tilt_step_deg)
    cache_key = (
        volume_bin_count,
        tilt_step_deg,
        int(round(target_volume / volume_step)),
        int(round(float(signed_tilt_length) / tilt_step)),
        int(round(float(signed_tilt_width) / tilt_step)),
    )
    topology_cache = context.setdefault("global_topology_cache", {})
    cached_state = topology_cache.get(cache_key)
    if cached_state is not None:
        return {
            "topology": cached_state["topology"],
            "left_head": cached_state["left_head"],
            "right_head": cached_state["right_head"],
            "left_volume": target_volume * cached_state["left_fraction"],
            "right_volume": target_volume * cached_state["right_fraction"],
        }

    solve_volume = float(np.clip(cache_key[2] * volume_step, 0.0, capacity))
    slope_length, slope_width, vertical_scale = get_stl_gravity_plane(
        cache_key[3] * tilt_step,
        cache_key[4] * tilt_step,
    )
    intercept = solve_animation_free_surface_3d(
        context,
        solve_volume,
        slope_length,
        slope_width,
    )
    liquid_depth = get_masked_liquid_height(
        context,
        intercept,
        slope_length,
        slope_width,
        context["valid_mask"],
    )
    x_grid = context["surface_x_grid"]
    width_grid = context["surface_width_grid"]
    cell_dx = max(float(np.median(np.abs(np.diff(x_grid, axis=1)))), 1e-6)
    cell_width = max(float(np.median(np.abs(np.diff(width_grid, axis=0)))), 1e-6)
    hydraulic_depth = 0.25 * min(cell_dx, cell_width)
    hydraulic_mask = context["valid_mask"] & (liquid_depth >= hydraulic_depth)

    left_port = reservoir_geometry["left_port"]
    right_port = reservoir_geometry["right_port"]
    left_port_width = reservoir_geometry["left_port_width"]
    right_port_width = reservoir_geometry["right_port_width"]
    left_head = vertical_scale * max(
        0.0,
        float(
            intercept
            - slope_length * left_port["x"]
            - slope_width * left_port_width
            - left_port["y"]
        ),
    )
    right_head = vertical_scale * max(
        0.0,
        float(
            intercept
            - slope_length * right_port["x"]
            - slope_width * right_port_width
            - right_port["y"]
        ),
    )

    def port_component(port, port_width):
        if not np.any(hydraulic_mask):
            return np.zeros_like(hydraulic_mask, dtype=bool)
        start_index = find_nearest_valid_grid_cell(
            x_grid,
            width_grid,
            hydraulic_mask,
            port["x"],
            port_width,
        )
        distance = math.hypot(
            float(x_grid[start_index] - port["x"]),
            float(width_grid[start_index] - port_width),
        )
        if distance > 2.0 * max(cell_dx, cell_width):
            return np.zeros_like(hydraulic_mask, dtype=bool)
        return get_port_connected_component_mask(hydraulic_mask, start_index)

    left_wet = left_head > 1e-6
    right_wet = right_head > 1e-6
    left_component = port_component(left_port, left_port_width) if left_wet else None
    right_component = port_component(right_port, right_port_width) if right_wet else None
    cell_volume = liquid_depth * float(context["surface_cell_area"])
    left_component_volume = (
        float(np.sum(cell_volume[left_component])) if left_component is not None else 0.0
    )
    right_component_volume = (
        float(np.sum(cell_volume[right_component])) if right_component is not None else 0.0
    )
    volume_tolerance = max(0.01 * solve_volume, 2.0 * hydraulic_depth * context["surface_cell_area"])

    if not left_wet and not right_wet:
        topology = "free"
        left_fraction = right_fraction = 0.0
    elif left_wet and not right_wet:
        if solve_volume - left_component_volume > volume_tolerance:
            topology = "free"
            left_fraction = right_fraction = 0.0
        else:
            topology = "left_only"
            left_fraction, right_fraction = 1.0, 0.0
    elif right_wet and not left_wet:
        if solve_volume - right_component_volume > volume_tolerance:
            topology = "free"
            left_fraction = right_fraction = 0.0
        else:
            topology = "right_only"
            left_fraction, right_fraction = 0.0, 1.0
    elif (
        left_component is not None
        and right_component is not None
        and np.any(left_component & right_component)
    ):
        topology = "connected"
        left_fraction = right_fraction = 0.0
    else:
        assigned_volume = left_component_volume + right_component_volume
        if assigned_volume <= 1e-12 or solve_volume - assigned_volume > volume_tolerance:
            topology = "free"
            left_fraction = right_fraction = 0.0
        else:
            topology = "separated"
            left_fraction = float(
                np.clip(left_component_volume / assigned_volume, 0.0, 1.0)
            )
            right_fraction = float(1.0 - left_fraction)

    state = {
        "topology": topology,
        "left_head": left_head,
        "right_head": right_head,
        "left_fraction": left_fraction,
        "right_fraction": right_fraction,
    }
    topology_cache[cache_key] = state
    return {
        "topology": topology,
        "left_head": left_head,
        "right_head": right_head,
        "left_volume": target_volume * left_fraction,
        "right_volume": target_volume * right_fraction,
    }


def solve_masked_free_surface_3d(
    surface_context,
    target_volume,
    slope_length,
    slope_width,
    region_mask,
):
    """Solve a free surface in one port-connected footprint region."""
    valid_mask = (
        region_mask
        & np.isfinite(surface_context["bottom_grid"])
        & np.isfinite(surface_context["top_grid"])
    )
    if not np.any(valid_mask) or target_volume <= 0.0:
        return float(np.min(surface_context["bottom_grid"][valid_mask])) - 1.0

    x_grid = surface_context["surface_x_grid"]
    width_grid = surface_context["surface_width_grid"]
    bottom_grid = surface_context["bottom_grid"]
    top_grid = surface_context["top_grid"]
    plane_offset = slope_length * x_grid + slope_width * width_grid
    lower_bound = float(np.min((bottom_grid + plane_offset)[valid_mask]))
    upper_bound = float(np.max((top_grid + plane_offset)[valid_mask]))
    cell_area = float(surface_context["surface_cell_area"])

    def liquid_volume(intercept):
        liquid_top = np.minimum(intercept - plane_offset, top_grid)
        liquid_height = np.where(
            valid_mask,
            np.maximum(0.0, liquid_top - bottom_grid),
            0.0,
        )
        return float(np.sum(liquid_height) * cell_area)

    if target_volume >= liquid_volume(upper_bound):
        return upper_bound
    for _ in range(60):
        midpoint = 0.5 * (lower_bound + upper_bound)
        if liquid_volume(midpoint) < target_volume:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    return 0.5 * (lower_bound + upper_bound)


def get_masked_liquid_height(
    surface_context,
    intercept,
    slope_length,
    slope_width,
    region_mask,
):
    liquid_top = np.minimum(
        intercept
        - slope_length * surface_context["surface_x_grid"]
        - slope_width * surface_context["surface_width_grid"],
        surface_context["top_grid"],
    )
    return np.where(
        region_mask
        & np.isfinite(surface_context["bottom_grid"])
        & np.isfinite(surface_context["top_grid"]),
        np.maximum(0.0, liquid_top - surface_context["bottom_grid"]),
        0.0,
    )


def wet_regions_touch(left_wet_mask, right_wet_mask):
    """True when two port pools touch through a shared footprint edge."""
    return bool(
        np.any(left_wet_mask & right_wet_mask)
        or np.any(left_wet_mask[1:, :] & right_wet_mask[:-1, :])
        or np.any(left_wet_mask[:-1, :] & right_wet_mask[1:, :])
        or np.any(left_wet_mask[:, 1:] & right_wet_mask[:, :-1])
        or np.any(left_wet_mask[:, :-1] & right_wet_mask[:, 1:])
    )


def get_wet_pool_connection_metrics(
    surface_context,
    left_wet_mask,
    right_wet_mask,
    left_liquid_depth,
    right_liquid_depth,
    left_global_height,
    right_global_height,
    vertical_scale,
):
    """Measure whether two independently solved pools form a physical neck."""
    x_grid = surface_context["surface_x_grid"]
    width_grid = surface_context["surface_width_grid"]
    cell_dx = float(np.min(np.diff(x_grid[0, :])))
    cell_width = float(np.min(np.diff(width_grid[:, 0])))
    smallest_cell_size = min(cell_dx, cell_width)
    minimum_local_depth = 0.25 * smallest_cell_size
    minimum_neck_width = 2.0 * smallest_cell_size
    level_tolerance = max(1e-4, 0.01 * smallest_cell_size * vertical_scale)

    contact_length = 0.0
    contact_depths = []

    def add_contact(left_depth, right_depth, left_mask, right_mask, edge_length):
        nonlocal contact_length
        shared_depth = np.minimum(left_depth, right_depth)
        valid_contact = left_mask & right_mask & (shared_depth >= minimum_local_depth)
        if np.any(valid_contact):
            contact_length += float(np.count_nonzero(valid_contact)) * edge_length
            contact_depths.append(float(np.min(shared_depth[valid_contact])))

    # Overlap can occur only because the two pools were solved independently.
    # Treat it as a potential neck only after the global levels agree.
    add_contact(
        left_liquid_depth,
        right_liquid_depth,
        left_wet_mask,
        right_wet_mask,
        smallest_cell_size,
    )
    add_contact(
        left_liquid_depth[1:, :],
        right_liquid_depth[:-1, :],
        left_wet_mask[1:, :],
        right_wet_mask[:-1, :],
        cell_width,
    )
    add_contact(
        left_liquid_depth[:-1, :],
        right_liquid_depth[1:, :],
        left_wet_mask[:-1, :],
        right_wet_mask[1:, :],
        cell_width,
    )
    add_contact(
        left_liquid_depth[:, 1:],
        right_liquid_depth[:, :-1],
        left_wet_mask[:, 1:],
        right_wet_mask[:, :-1],
        cell_dx,
    )
    add_contact(
        left_liquid_depth[:, :-1],
        right_liquid_depth[:, 1:],
        left_wet_mask[:, :-1],
        right_wet_mask[:, 1:],
        cell_dx,
    )

    level_difference = abs(left_global_height - right_global_height)
    minimum_contact_depth = min(contact_depths) if contact_depths else 0.0
    return {
        "connected": bool(
            level_difference <= level_tolerance
            and contact_length >= minimum_neck_width
            and minimum_contact_depth >= minimum_local_depth
        ),
        "level_difference_mm": level_difference,
        "level_tolerance_mm": level_tolerance,
        "neck_width_mm": contact_length,
        "minimum_neck_width_mm": minimum_neck_width,
        "neck_depth_mm": minimum_contact_depth,
        "minimum_neck_depth_mm": minimum_local_depth,
    }


def get_port_connected_component_mask(candidate_mask, start_index):
    """Return the four-neighbour wet component containing a selected port."""
    component = np.zeros_like(candidate_mask, dtype=bool)
    if not candidate_mask[start_index]:
        return component

    # Propagate the wet component with vectorized neighbour operations.  This
    # is substantially faster than a Python queue inside every bisection step.
    component[start_index] = True
    while True:
        expanded = component.copy()
        expanded[1:, :] |= component[:-1, :]
        expanded[:-1, :] |= component[1:, :]
        expanded[:, 1:] |= component[:, :-1]
        expanded[:, :-1] |= component[:, 1:]
        expanded &= candidate_mask
        if np.array_equal(expanded, component):
            break
        component = expanded
    return component


def get_grid_geodesic_distances(valid_mask, seed_mask):
    """Return four-neighbour distances through a valid STL footprint."""
    distances = np.full(valid_mask.shape, np.inf, dtype=float)
    queue = [tuple(index) for index in np.argwhere(valid_mask & seed_mask)]
    for row, column in queue:
        distances[row, column] = 0.0

    queue_index = 0
    row_count, column_count = valid_mask.shape
    while queue_index < len(queue):
        row, column = queue[queue_index]
        queue_index += 1
        next_distance = distances[row, column] + 1.0
        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                0 <= next_row < row_count
                and 0 <= next_column < column_count
                and valid_mask[next_row, next_column]
                and next_distance < distances[next_row, next_column]
            ):
                distances[next_row, next_column] = next_distance
                queue.append((next_row, next_column))
    return distances


def solve_port_connected_liquid_pool(
    x_grid,
    width_grid,
    bottom_grid,
    top_grid,
    valid_mask,
    cell_area,
    port,
    port_width,
    target_volume,
    slope_length,
    slope_width,
    top_contact_wetting=False,
):
    """Solve a tilted pool that can wet only from its selected inlet/outlet."""
    if not np.any(valid_mask):
        raise ValueError("The imported STL footprint has no valid liquid cells.")

    start_index = find_nearest_valid_grid_cell(
        x_grid,
        width_grid,
        valid_mask,
        port["x"],
        port_width,
    )
    plane_offset = slope_length * x_grid + slope_width * width_grid
    lower_bound = float(np.min((bottom_grid + plane_offset)[valid_mask]))
    upper_bound = float(np.max((top_grid + plane_offset)[valid_mask]))

    def pool_at_intercept(intercept):
        liquid_top = np.minimum(intercept - plane_offset, top_grid)
        liquid_height = np.where(
            valid_mask,
            np.maximum(0.0, liquid_top - bottom_grid),
            0.0,
        )
        wet_component = get_port_connected_component_mask(
            liquid_height > 1e-9,
            start_index,
        )
        return (
            float(np.sum(liquid_height[wet_component]) * cell_area),
            wet_component,
            liquid_height,
        )

    def has_top_contact(wet_component, liquid_height):
        full_height = np.maximum(top_grid - bottom_grid, 0.0)
        top_tolerance = max(1e-6, 1e-3 * float(np.max(full_height[valid_mask])))
        return bool(
            np.any(
                wet_component
                & (liquid_height >= full_height - top_tolerance)
            )
        )

    if target_volume <= 0.0:
        _, wet_component, liquid_height = pool_at_intercept(lower_bound)
        return lower_bound, wet_component, liquid_height

    max_volume, _, _ = pool_at_intercept(upper_bound)
    if target_volume >= max_volume:
        _, wet_component, liquid_height = pool_at_intercept(upper_bound)
        return upper_bound, wet_component, liquid_height

    if top_contact_wetting:
        _, lower_wet, lower_height = pool_at_intercept(lower_bound)
        _, upper_wet, upper_height = pool_at_intercept(upper_bound)
        if (
            not has_top_contact(lower_wet, lower_height)
            and has_top_contact(upper_wet, upper_height)
        ):
            contact_lower = lower_bound
            contact_upper = upper_bound
            for _ in range(18):
                midpoint = 0.5 * (contact_lower + contact_upper)
                _, contact_wet, contact_height = pool_at_intercept(midpoint)
                if has_top_contact(contact_wet, contact_height):
                    contact_upper = midpoint
                else:
                    contact_lower = midpoint
            contact_intercept = contact_upper
            contact_volume, contact_wet, contact_height = pool_at_intercept(
                contact_intercept
            )
            if target_volume > contact_volume + 1e-9:
                # Once the ceiling is reached, approximate 90-degree wetting
                # by completing contacted columns and advancing a connected,
                # full-height plug through the STL footprint.
                reachable_mask = get_port_connected_component_mask(
                    valid_mask,
                    start_index,
                )
                base_height = np.where(contact_wet, contact_height, 0.0)
                column_capacity = np.where(
                    reachable_mask,
                    np.maximum(top_grid - bottom_grid - base_height, 0.0),
                    0.0,
                )
                distances = get_grid_geodesic_distances(
                    reachable_mask,
                    contact_wet,
                )
                # Finish the existing wetted region before the plug advances.
                priority = np.where(contact_wet, -1.0, distances)
                candidate_indices = np.flatnonzero(
                    reachable_mask & (column_capacity > 1e-12)
                )
                ordered_indices = candidate_indices[
                    np.argsort(priority.flat[candidate_indices], kind="stable")
                ]
                additional_volume = target_volume - contact_volume
                expanded_height = base_height.copy()
                for flat_index in ordered_indices:
                    row, column = np.unravel_index(flat_index, valid_mask.shape)
                    available_volume = column_capacity[row, column] * cell_area
                    added_volume = min(additional_volume, available_volume)
                    expanded_height[row, column] += added_volume / cell_area
                    additional_volume -= added_volume
                    if additional_volume <= 1e-9:
                        break
                wet_component = get_port_connected_component_mask(
                    expanded_height > 1e-9,
                    start_index,
                )
                return contact_intercept, wet_component, expanded_height

    for _ in range(18):
        midpoint = 0.5 * (lower_bound + upper_bound)
        midpoint_volume, _, _ = pool_at_intercept(midpoint)
        if midpoint_volume < target_volume:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    intercept = 0.5 * (lower_bound + upper_bound)
    _, wet_component, liquid_height = pool_at_intercept(intercept)
    return intercept, wet_component, liquid_height


def solve_port_connected_free_surface_3d(
    surface_context,
    target_volume,
    slope_length,
    slope_width,
    port,
    port_width,
    top_contact_wetting=False,
):
    """Solve one dry STL pool without an artificial midpoint partition."""
    valid_mask = (
        surface_context["point_mask"]
        & np.isfinite(surface_context["bottom_grid"])
        & np.isfinite(surface_context["top_grid"])
    )
    return solve_port_connected_liquid_pool(
        surface_context["surface_x_grid"],
        surface_context["surface_width_grid"],
        surface_context["bottom_grid"],
        surface_context["top_grid"],
        valid_mask,
        float(surface_context["surface_cell_area"]),
        port,
        port_width,
        target_volume,
        slope_length,
        slope_width,
        top_contact_wetting=top_contact_wetting,
    )


def get_stl_port_pool_state_for_volumes(
    total_volume,
    left_volume,
    right_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    use_cache=True,
):
    """Return STL port-pool heads for the explicitly transported volumes."""
    footprint_context = get_stl_footprint_context(reservoir_geometry)
    top_contact_wetting = bool(reservoir_geometry.get("top_contact_wetting", False))
    target_total_volume = float(
        np.clip(total_volume, 0.0, footprint_context["total_capacity_ul"])
    )
    left_volume = float(np.clip(left_volume, 0.0, target_total_volume))
    # Keep the mass balance exact even if a caller accumulates floating-point
    # round-off in the separately stored right-hand volume.
    right_volume = target_total_volume - left_volume
    left_capacity = target_total_volume
    right_capacity = target_total_volume

    if use_cache:
        volume_bin_count, tilt_step_deg = get_stl_wetting_cache_settings(
            reservoir_geometry
        )
        volume_step = max(
            footprint_context["total_capacity_ul"] / volume_bin_count, 0.1
        )
        tilt_step = math.radians(tilt_step_deg)
        pool_cache_key = (
            volume_bin_count,
            tilt_step_deg,
            top_contact_wetting,
            int(round(left_volume / volume_step)),
            int(round(right_volume / volume_step)),
            int(round(float(signed_tilt_length) / tilt_step)),
            int(round(float(signed_tilt_width) / tilt_step)),
        )
        pool_cache = footprint_context.setdefault("pool_state_cache", {})
        cached_pool_state = pool_cache.get(pool_cache_key)
        solve_left_volume = np.clip(pool_cache_key[3] * volume_step, 0.0, left_capacity)
        solve_right_volume = np.clip(pool_cache_key[4] * volume_step, 0.0, right_capacity)
        slope_length, slope_width, vertical_scale = get_stl_gravity_plane(
            pool_cache_key[5] * tilt_step,
            pool_cache_key[6] * tilt_step,
        )
    else:
        pool_cache = None
        pool_cache_key = None
        cached_pool_state = None
        solve_left_volume = left_volume
        solve_right_volume = right_volume
        slope_length, slope_width, vertical_scale = get_stl_gravity_plane(
            signed_tilt_length,
            signed_tilt_width,
        )
    if cached_pool_state is None:
        left_port = reservoir_geometry["left_port"]
        right_port = reservoir_geometry["right_port"]
        left_intercept, left_wet, left_liquid_depth = solve_port_connected_free_surface_3d(
            footprint_context,
            solve_left_volume,
            slope_length,
            slope_width,
            left_port,
            reservoir_geometry["left_port_width"],
            top_contact_wetting=top_contact_wetting,
        )
        right_intercept, right_wet, right_liquid_depth = solve_port_connected_free_surface_3d(
            footprint_context,
            solve_right_volume,
            slope_length,
            slope_width,
            right_port,
            reservoir_geometry["right_port_width"],
            top_contact_wetting=top_contact_wetting,
        )
        connection_metrics = get_wet_pool_connection_metrics(
            footprint_context,
            left_wet,
            right_wet,
            left_liquid_depth,
            right_liquid_depth,
            vertical_scale * left_intercept,
            vertical_scale * right_intercept,
            vertical_scale,
        )
        cached_pool_state = {
            "connected": connection_metrics["connected"],
            "left_global_height": vertical_scale * left_intercept,
            "right_global_height": vertical_scale * right_intercept,
            "connection_metrics": connection_metrics,
            "left_head": vertical_scale * max(
                0.0,
                float(
                    left_intercept
                    - slope_length * left_port["x"]
                    - slope_width * reservoir_geometry["left_port_width"]
                    - left_port["y"]
                ),
            ),
            "right_head": vertical_scale * max(
                0.0,
                float(
                    right_intercept
                    - slope_length * right_port["x"]
                    - slope_width * reservoir_geometry["right_port_width"]
                    - right_port["y"]
                ),
            ),
        }
        if use_cache:
            pool_cache[pool_cache_key] = cached_pool_state

    if cached_pool_state["connected"]:
        # Once the pools touch, the reservoir has one water table again.
        equilibrium = get_stl_footprint_equilibrium(
            target_total_volume,
            signed_tilt_length,
            signed_tilt_width,
            reservoir_geometry,
        )
        return {
            "connected": True,
            "left_head": equilibrium["left_head"],
            "right_head": equilibrium["right_head"],
            "left_volume": left_volume,
            "right_volume": right_volume,
            "left_global_height": cached_pool_state["left_global_height"],
            "right_global_height": cached_pool_state["right_global_height"],
            "connection_metrics": cached_pool_state["connection_metrics"],
        }
    return {
        "connected": False,
        "left_head": cached_pool_state["left_head"],
        "right_head": cached_pool_state["right_head"],
        "left_volume": left_volume,
        "right_volume": right_volume,
        "left_global_height": cached_pool_state["left_global_height"],
        "right_global_height": cached_pool_state["right_global_height"],
        "connection_metrics": cached_pool_state["connection_metrics"],
    }


def classify_stl_local_pool_topology(pool_state, left_volume, right_volume):
    """Classify explicit port pools without reconstructing a midpoint split."""
    volume_tolerance = 1e-6
    left_wet = left_volume > volume_tolerance and pool_state["left_head"] > 1e-6
    right_wet = right_volume > volume_tolerance and pool_state["right_head"] > 1e-6
    if left_wet and right_wet:
        return "connected" if pool_state["connected"] else "separated"
    if left_wet:
        return "left_only"
    if right_wet:
        return "right_only"
    return "free"


def get_stl_port_pool_state(
    total_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    volume_offset,
):
    """Legacy offset wrapper for callers that have not yet stored pool volumes."""
    footprint_context = get_stl_footprint_context(reservoir_geometry)
    equilibrium = get_stl_footprint_equilibrium(
        total_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
    )
    target_total_volume = float(
        np.clip(total_volume, 0.0, footprint_context["total_capacity_ul"])
    )
    min_offset = -equilibrium["right_volume"]
    max_offset = equilibrium["left_volume"]
    clamped_offset = float(np.clip(volume_offset, min_offset, max_offset))
    state = get_stl_port_pool_state_for_volumes(
        target_total_volume,
        equilibrium["left_volume"] - clamped_offset,
        equilibrium["right_volume"] + clamped_offset,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
    )
    state["volume_offset"] = 0.0 if state["connected"] else clamped_offset
    return state


def get_stl_reservoir_resistance_context(reservoir_geometry):
    """Return a coarse footprint grid used only for the internal bypass solve."""
    cached_context = reservoir_geometry.get("_stl_resistance_context")
    if cached_context is not None:
        return cached_context

    # This grid is deliberately coarser than the pool grid.  The resistance is
    # cached by liquid state, so it captures slots and bottlenecks without
    # making every time step as expensive as a full 3-D CFD calculation.
    context = build_animation_surface_context(
        reservoir_geometry,
        x_count=32,
        width_count=22,
    )
    context["valid_mask"] = (
        context["point_mask"]
        & np.isfinite(context["bottom_grid"])
        & np.isfinite(context["top_grid"])
    )
    context["total_capacity_ul"] = float(
        np.sum(
            np.where(
                context["valid_mask"],
                np.maximum(0.0, context["top_grid"] - context["bottom_grid"]),
                0.0,
            )
            * float(context["surface_cell_area"])
        )
    )
    context["resistance_cache"] = {}
    reservoir_geometry["_stl_resistance_context"] = context
    return context


def get_stl_port_boundary_mask(context, port, port_width):
    """Use a small wetted patch around a selected STL port as a pressure node."""
    valid_mask = context["valid_mask"]
    x_grid = context["surface_x_grid"]
    width_grid = context["surface_width_grid"]
    x_spacing = max(
        float(np.median(np.abs(np.diff(x_grid, axis=1)))),
        1e-6,
    )
    width_spacing = max(
        float(np.median(np.abs(np.diff(width_grid, axis=0)))),
        1e-6,
    )
    patch_radius = 1.6 * max(x_spacing, width_spacing)
    distance_squared = (
        (x_grid - float(port["x"])) ** 2
        + (width_grid - float(port_width)) ** 2
    )
    boundary_mask = valid_mask & (distance_squared <= patch_radius**2)
    if np.any(boundary_mask):
        return boundary_mask

    nearest = find_nearest_valid_grid_cell(
        x_grid,
        width_grid,
        valid_mask,
        port["x"],
        port_width,
    )
    boundary_mask = np.zeros_like(valid_mask, dtype=bool)
    boundary_mask[nearest] = True
    return boundary_mask


def solve_stl_wet_network_resistance(context, liquid_depth_mm, left_port, left_port_width, right_port, right_port_width):
    """Solve the effective resistance of the currently wetted STL footprint.

    Adjacent wet cells form a lubrication-network edge.  Solving the pressure
    field between the two port patches includes parallel paths around bends and
    through widened regions, unlike a shortest-path approximation.
    """
    wet_mask = context["valid_mask"] & (liquid_depth_mm > 1e-4)
    if not np.any(wet_mask):
        return math.inf

    active_context = dict(context)
    active_context["valid_mask"] = wet_mask
    left_boundary = get_stl_port_boundary_mask(
        active_context,
        left_port,
        left_port_width,
    )
    right_boundary = get_stl_port_boundary_mask(
        active_context,
        right_port,
        right_port_width,
    )
    if np.any(left_boundary & right_boundary):
        return 0.0

    # The free surface is a very thin film at its edge.  Excluding it avoids a
    # numerically huge resistance that is not a meaningful hydraulic bridge.
    depth_m = np.maximum(liquid_depth_mm, 0.0) * 1e-3
    x_grid = context["surface_x_grid"]
    width_grid = context["surface_width_grid"]
    dx_m = max(float(np.median(np.abs(np.diff(x_grid, axis=1)))) * 1e-3, 1e-9)
    dy_m = max(float(np.median(np.abs(np.diff(width_grid, axis=0)))) * 1e-3, 1e-9)

    # Conductance of a shallow liquid sheet: G = h^3 * face_width / (12 mu ds).
    horizontal_wet = wet_mask[:, :-1] & wet_mask[:, 1:]
    vertical_wet = wet_mask[:-1, :] & wet_mask[1:, :]
    horizontal_depth = np.minimum(depth_m[:, :-1], depth_m[:, 1:])
    vertical_depth = np.minimum(depth_m[:-1, :], depth_m[1:, :])
    horizontal_g = np.where(
        horizontal_wet,
        horizontal_depth**3 * dy_m / (12.0 * viscosity * dx_m),
        0.0,
    )
    vertical_g = np.where(
        vertical_wet,
        vertical_depth**3 * dx_m / (12.0 * viscosity * dy_m),
        0.0,
    )

    total_g = np.zeros_like(depth_m)
    total_g[:, :-1] += horizontal_g
    total_g[:, 1:] += horizontal_g
    total_g[:-1, :] += vertical_g
    total_g[1:, :] += vertical_g
    fixed_mask = left_boundary | right_boundary
    solvable_mask = wet_mask & ~fixed_mask & (total_g > 0.0)
    if not np.any(solvable_mask):
        return math.inf

    # Unit pressure drop.  A damped Jacobi solve is robust for irregular masks
    # and avoids adding SciPy as a runtime dependency to the standalone app.
    pressure = np.full(depth_m.shape, 0.5, dtype=float)
    pressure[left_boundary] = 1.0
    pressure[right_boundary] = 0.0
    for _ in range(90):
        weighted_pressure = np.zeros_like(pressure)
        weighted_pressure[:, :-1] += horizontal_g * pressure[:, 1:]
        weighted_pressure[:, 1:] += horizontal_g * pressure[:, :-1]
        weighted_pressure[:-1, :] += vertical_g * pressure[1:, :]
        weighted_pressure[1:, :] += vertical_g * pressure[:-1, :]
        next_pressure = pressure.copy()
        next_pressure[solvable_mask] = (
            weighted_pressure[solvable_mask] / total_g[solvable_mask]
        )
        next_pressure[left_boundary] = 1.0
        next_pressure[right_boundary] = 0.0
        maximum_change = float(np.max(np.abs(next_pressure[solvable_mask] - pressure[solvable_mask])))
        pressure = 0.65 * next_pressure + 0.35 * pressure
        pressure[left_boundary] = 1.0
        pressure[right_boundary] = 0.0
        if maximum_change < 3e-5:
            break

    # Sum the conductance leaving the inlet pressure patch for Delta p = 1 Pa.
    flow_per_pa = 0.0
    left_to_right = left_boundary[:, :-1] & ~left_boundary[:, 1:]
    right_to_left = left_boundary[:, 1:] & ~left_boundary[:, :-1]
    flow_per_pa += float(np.sum(horizontal_g[left_to_right] * (1.0 - pressure[:, 1:][left_to_right])))
    flow_per_pa += float(np.sum(horizontal_g[right_to_left] * (1.0 - pressure[:, :-1][right_to_left])))
    top_to_bottom = left_boundary[:-1, :] & ~left_boundary[1:, :]
    bottom_to_top = left_boundary[1:, :] & ~left_boundary[:-1, :]
    flow_per_pa += float(np.sum(vertical_g[top_to_bottom] * (1.0 - pressure[1:, :][top_to_bottom])))
    flow_per_pa += float(np.sum(vertical_g[bottom_to_top] * (1.0 - pressure[:-1, :][bottom_to_top])))
    return math.inf if flow_per_pa <= 1e-18 else 1.0 / flow_per_pa


def get_stl_dynamic_reservoir_resistance(
    total_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    use_cache=True,
):
    """Return the wet-state internal bypass resistance for an imported STL."""
    context = get_stl_reservoir_resistance_context(reservoir_geometry)
    capacity = max(float(context["total_capacity_ul"]), 1e-6)
    if use_cache:
        volume_step = max(capacity / 120.0, 0.1)
        tilt_step = math.radians(0.75)
        cache_key = (
            int(round(float(total_volume) / volume_step)),
            int(round(float(signed_tilt_length) / tilt_step)),
            int(round(float(signed_tilt_width) / tilt_step)),
        )
        cached_resistance = context["resistance_cache"].get(cache_key)
        if cached_resistance is not None:
            return cached_resistance
        solve_volume = float(np.clip(cache_key[0] * volume_step, 0.0, capacity))
        slope_length, slope_width, _ = get_stl_gravity_plane(
            cache_key[1] * tilt_step,
            cache_key[2] * tilt_step,
        )
    else:
        cache_key = None
        solve_volume = float(np.clip(total_volume, 0.0, capacity))
        slope_length, slope_width, _ = get_stl_gravity_plane(
            signed_tilt_length,
            signed_tilt_width,
        )
    intercept = solve_animation_free_surface_3d(
        context,
        solve_volume,
        slope_length,
        slope_width,
    )
    liquid_depth = get_masked_liquid_height(
        context,
        intercept,
        slope_length,
        slope_width,
        context["valid_mask"],
    )
    resistance = solve_stl_wet_network_resistance(
        context,
        liquid_depth,
        reservoir_geometry["left_port"],
        reservoir_geometry["left_port_width"],
        reservoir_geometry["right_port"],
        reservoir_geometry["right_port_width"],
    )
    if use_cache:
        context["resistance_cache"][cache_key] = resistance
    return resistance


def limit_stl_flow_before_pool_merge(
    flow_rate,
    total_volume,
    left_volume,
    right_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    time_step,
):
    """Limit a transported volume to the first dry-pool contact event."""
    if abs(flow_rate) <= 1e-12:
        return flow_rate

    current_state = get_stl_port_pool_state_for_volumes(
        total_volume,
        left_volume,
        right_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
    )
    if current_state["connected"]:
        return 0.0

    proposed_state = get_stl_port_pool_state_for_volumes(
        total_volume,
        left_volume + flow_rate * time_step,
        right_volume - flow_rate * time_step,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
    )
    if not proposed_state["connected"]:
        return flow_rate

    lower_fraction = 0.0
    upper_fraction = 1.0
    for _ in range(12):
        midpoint = 0.5 * (lower_fraction + upper_fraction)
        midpoint_state = get_stl_port_pool_state_for_volumes(
            total_volume,
            left_volume + flow_rate * time_step * midpoint,
            right_volume - flow_rate * time_step * midpoint,
            signed_tilt_length,
            signed_tilt_width,
            reservoir_geometry,
        )
        if midpoint_state["connected"]:
            upper_fraction = midpoint
        else:
            lower_fraction = midpoint
    return flow_rate * max(0.0, lower_fraction * (1.0 - 1e-6))


def calculate_stl_single_reservoir_flow(
    total_volume,
    left_volume,
    right_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    compartments,
    flow_coefficient,
    channel_resistance,
    capillary_head_threshold,
    time_step,
):
    """Calculate one volume-limited port-to-port flow update for an STL cell."""
    pool_state = get_stl_port_pool_state_for_volumes(
        total_volume,
        left_volume,
        right_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        use_cache=False,
    )
    h_left = pool_state["left_head"]
    h_right = pool_state["right_head"]
    raw_driving_head = h_right - h_left
    if pool_state["connected"]:
        reservoir_resistance = get_stl_dynamic_reservoir_resistance(
            total_volume,
            signed_tilt_length,
            signed_tilt_width,
            reservoir_geometry,
            use_cache=False,
        )
        if not math.isfinite(reservoir_resistance):
            # A connected pool without a resolved wet network is treated as a
            # hydrostatic body rather than falling back to an unrealistically
            # large external-channel flow.
            return 0.0, h_left, h_right, raw_driving_head, True
        bypass_fraction = reservoir_resistance / (
            max(channel_resistance, 1e-18) + reservoir_resistance
        )
        driving_head = apply_capillary_cutoff(
            raw_driving_head,
            h_right,
            h_left,
            capillary_head_threshold,
        )
        # The previous channel-only result represents the tilt-driven
        # redistribution.  The internal wet reservoir path shunts the part
        # that can flow through the reservoir instead of the perfusion channel.
        return (
            driving_head * flow_coefficient * bypass_fraction,
            h_left,
            h_right,
            raw_driving_head,
            True,
        )

    driving_head = apply_capillary_cutoff(
        raw_driving_head,
        h_right,
        h_left,
        capillary_head_threshold,
    )
    flow_rate = limit_single_reservoir_flow_by_volume(
        driving_head * flow_coefficient,
        left_volume,
        right_volume,
        compartments,
        time_step,
    )
    flow_rate = limit_stl_flow_before_pool_merge(
        flow_rate,
        total_volume,
        left_volume,
        right_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        time_step,
    )
    return flow_rate, h_left, h_right, raw_driving_head, False


def get_stl_pressure_compliance_ul_per_pa(compartments):
    """Approximate differential port compliance from the two local volumes."""
    left_compliance_ul_mm = max(float(compartments["left_head_compliance"]), 1e-6)
    right_compliance_ul_mm = max(float(compartments["right_head_compliance"]), 1e-6)
    differential_compliance_ul_mm = 1.0 / (
        1.0 / left_compliance_ul_mm + 1.0 / right_compliance_ul_mm
    )
    return max(
        differential_compliance_ul_mm / (density * gravity * 1e-3),
        1e-6,
    )


def get_stl_separated_pool_compliance_ul_per_pa(
    total_volume,
    left_volume,
    right_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    fallback_compliance,
):
    """Estimate local port-pool compliance from the imported free surfaces."""
    footprint_context = get_stl_footprint_context(reservoir_geometry)
    capacity = max(float(footprint_context["total_capacity_ul"]), 1e-6)
    available_positive = min(
        max(right_volume, 0.0),
        max(total_volume - left_volume, 0.0),
    )
    available_negative = min(
        max(left_volume, 0.0),
        max(total_volume - right_volume, 0.0),
    )
    sample_volume = min(
        max(capacity / 500.0, 0.05),
        0.45 * available_positive,
        0.45 * available_negative,
    )
    if sample_volume <= 1e-6:
        return fallback_compliance

    positive_state = get_stl_port_pool_state_for_volumes(
        total_volume,
        left_volume + sample_volume,
        right_volume - sample_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        use_cache=False,
    )
    negative_state = get_stl_port_pool_state_for_volumes(
        total_volume,
        left_volume - sample_volume,
        right_volume + sample_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        use_cache=False,
    )
    if positive_state["connected"] or negative_state["connected"]:
        return fallback_compliance
    positive_pressure = density * gravity * (
        positive_state["right_head"] - positive_state["left_head"]
    ) * 1e-3
    negative_pressure = density * gravity * (
        negative_state["right_head"] - negative_state["left_head"]
    ) * 1e-3
    pressure_stiffness = -(
        positive_pressure - negative_pressure
    ) / (2.0 * sample_volume)
    if not math.isfinite(pressure_stiffness) or pressure_stiffness <= 1e-9:
        return fallback_compliance
    return float(np.clip(1.0 / pressure_stiffness, 1e-6, fallback_compliance * 20.0))


def get_stl_separated_pool_channel_flow(
    total_volume,
    left_volume,
    right_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    channel_resistance,
    capillary_pressure_pa,
    use_wet_dry_gating,
    use_cache=True,
):
    """Return channel flow from the global-level difference of separated pools."""
    pool_state = get_stl_port_pool_state_for_volumes(
        total_volume,
        left_volume,
        right_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        use_cache=use_cache,
    )
    h_left = pool_state["left_head"]
    h_right = pool_state["right_head"]
    # Port pressure includes the hydrostatic elevation of each port.  The
    # matching gravity term along a rigid connecting channel cancels that
    # elevation, leaving only the difference in global free-surface height.
    pressure_difference = density * gravity * (
        pool_state["right_global_height"] - pool_state["left_global_height"]
    ) * 1e-3
    channel_conductance = 1e9 / max(channel_resistance, 1e-18)
    if pressure_difference > 0.0:
        upstream_head, downstream_head = h_right, h_left
    elif pressure_difference < 0.0:
        upstream_head, downstream_head = h_left, h_right
    else:
        upstream_head = downstream_head = max(h_left, h_right)
    if use_wet_dry_gating and (
        upstream_head <= 1e-9
        or (
            downstream_head <= 1e-9
            and abs(pressure_difference) < capillary_pressure_pa
        )
    ):
        channel_conductance = 0.0
    return (
        channel_conductance * pressure_difference,
        pool_state,
        channel_conductance,
        pressure_difference,
    )


def solve_stl_separated_pool_volume_step_cached(
    total_volume,
    left_volume,
    right_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    compartments,
    channel_resistance,
    capillary_pressure_pa,
    time_step,
    use_wet_dry_gating,
):
    """Advance separated pools with one cached, volume-limited flow update."""
    initial_flow, initial_state, _, _ = get_stl_separated_pool_channel_flow(
        total_volume,
        left_volume,
        right_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        channel_resistance,
        capillary_pressure_pa,
        use_wet_dry_gating,
    )
    if initial_state["connected"] or abs(initial_flow) <= 1e-12:
        return left_volume, right_volume, 0.0, initial_state

    # This is intentionally one forward-Euler update.  The cache makes the
    # water-table solve practical at fine user-selected time steps, while the
    # limiter retains the exact total liquid volume and prevents dry-pool
    # overshoot.
    bounded_flow = limit_single_reservoir_flow_by_volume(
        initial_flow,
        left_volume,
        right_volume,
        compartments,
        time_step,
    )
    next_left_volume = float(
        np.clip(
            left_volume + bounded_flow * time_step,
            0.0,
            total_volume,
        )
    )
    next_right_volume = float(total_volume - next_left_volume)
    return next_left_volume, next_right_volume, bounded_flow, initial_state


def solve_stl_separated_pool_volume_step_predictor_corrector(
    total_volume,
    left_volume,
    right_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    compartments,
    channel_resistance,
    capillary_pressure_pa,
    time_step,
    use_wet_dry_gating,
):
    """Use exact STL states in a faster two-stage volume update."""
    initial_flow, initial_state, _, _ = get_stl_separated_pool_channel_flow(
        total_volume,
        left_volume,
        right_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        channel_resistance,
        capillary_pressure_pa,
        use_wet_dry_gating,
        use_cache=False,
    )
    if initial_state["connected"] or abs(initial_flow) <= 1e-12:
        return left_volume, right_volume, 0.0, initial_state

    predicted_left_volume = float(
        np.clip(left_volume + initial_flow * time_step, 0.0, total_volume)
    )
    predicted_flow, predicted_state, _, _ = get_stl_separated_pool_channel_flow(
        total_volume,
        predicted_left_volume,
        total_volume - predicted_left_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        channel_resistance,
        capillary_pressure_pa,
        use_wet_dry_gating,
        use_cache=False,
    )
    if predicted_state["connected"]:
        # Do not step through a topological merge with the extrapolated state.
        bounded_flow = limit_single_reservoir_flow_by_volume(
            initial_flow,
            left_volume,
            right_volume,
            compartments,
            time_step,
        )
        next_left_volume = float(
            np.clip(left_volume + bounded_flow * time_step, 0.0, total_volume)
        )
        return (
            next_left_volume,
            float(total_volume - next_left_volume),
            bounded_flow,
            initial_state,
        )

    next_left_volume = float(
        np.clip(
            left_volume + 0.5 * (initial_flow + predicted_flow) * time_step,
            0.0,
            total_volume,
        )
    )
    next_right_volume = float(total_volume - next_left_volume)
    solved_flow = (next_left_volume - left_volume) / time_step
    _, final_state, _, _ = get_stl_separated_pool_channel_flow(
        total_volume,
        next_left_volume,
        next_right_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        channel_resistance,
        capillary_pressure_pa,
        use_wet_dry_gating,
        use_cache=False,
    )
    return next_left_volume, next_right_volume, solved_flow, final_state


def solve_stl_separated_pool_volume_step(*args, wetting_solver="cached"):
    """Dispatch the selected STL two-pool wetting/transport solver."""
    if wetting_solver == "cached":
        return solve_stl_separated_pool_volume_step_cached(*args)
    if wetting_solver in ("predictor_corrector", "implicit"):
        return solve_stl_separated_pool_volume_step_predictor_corrector(*args)
    raise ValueError(f"Unsupported STL wetting solver: {wetting_solver}")


def calculate_stl_rate_driven_channel_flow(
    total_volume,
    left_volume,
    right_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    compartments,
    channel_resistance,
    capillary_pressure_pa,
    time_step,
    required_redistribution_flow,
    pressure_difference_pa,
    use_wet_dry_gating=True,
):
    """Advance a tilt-rate driven channel/reservoir parallel circuit.

    Positive flow moves liquid from the outlet (right) port to the inlet (left)
    port.  The rotation creates a requested geometric redistribution; the
    external channel and wet reservoir path share it through their conductance.
    """
    pool_state = get_stl_port_pool_state_for_volumes(
        total_volume,
        left_volume,
        right_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
    )
    h_left = pool_state["left_head"]
    h_right = pool_state["right_head"]
    # Separate port pools have a real pressure difference from their distinct
    # free surfaces.  A common pool does not: its depth difference is part of
    # the hydrostatic tilted surface and must not be counted as a second pump.
    pool_level_pressure_pa = (
        0.0
        if pool_state["connected"]
        else density * gravity * (h_right - h_left) * 1e-3
    )
    reservoir_conductance_ul_pa_s = 0.0
    if pool_state["connected"]:
        reservoir_resistance = get_stl_dynamic_reservoir_resistance(
            total_volume,
            signed_tilt_length,
            signed_tilt_width,
            reservoir_geometry,
        )
        if math.isfinite(reservoir_resistance) and reservoir_resistance > 0.0:
            reservoir_conductance_ul_pa_s = 1e9 / reservoir_resistance

    if pool_state["connected"]:
        direction_signal = (
            pressure_difference_pa
            if abs(pressure_difference_pa) > 1e-9
            else required_redistribution_flow
        )
        available_channel_pressure = pressure_difference_pa
    else:
        # With a dry gap, the free-surface difference is the physical source.
        # Tilt-rate forcing is already expressed through that changing surface.
        direction_signal = pool_level_pressure_pa
        available_channel_pressure = pool_level_pressure_pa
    channel_conductance_ul_pa_s = 1e9 / max(channel_resistance, 1e-18)
    if direction_signal > 0.0:
        upstream_head, downstream_head = h_right, h_left
    elif direction_signal < 0.0:
        upstream_head, downstream_head = h_left, h_right
    else:
        upstream_head = downstream_head = max(h_left, h_right)

    # An unprimed channel cannot drain a dry donor port.  A dry receiver also
    # needs enough dynamic pressure to overcome the configured capillary entry
    # threshold before it becomes a conductive branch.
    if use_wet_dry_gating and (
        upstream_head <= 1e-9
        or (
            downstream_head <= 1e-9
            and abs(available_channel_pressure) < capillary_pressure_pa
        )
    ):
        channel_conductance_ul_pa_s = 0.0

    total_conductance = (
        channel_conductance_ul_pa_s + reservoir_conductance_ul_pa_s
    )
    compliance = get_stl_pressure_compliance_ul_per_pa(compartments)
    if not pool_state["connected"]:
        compliance = get_stl_separated_pool_compliance_ul_per_pa(
            total_volume,
            left_volume,
            right_volume,
            signed_tilt_length,
            signed_tilt_width,
            reservoir_geometry,
            compliance,
        )
    if pool_state["connected"]:
        # Backward Euler is stable even when a large channel makes the
        # connected-pool hydraulic time constant shorter than the output step.
        next_pressure_difference = (
            pressure_difference_pa
            + time_step * required_redistribution_flow / compliance
        ) / (1.0 + time_step * total_conductance / compliance)
        _, _, vertical_scale = get_stl_gravity_plane(
            signed_tilt_length,
            signed_tilt_width,
        )
        max_pressure_difference = (
            density * gravity * reservoir_geometry["height"] * vertical_scale * 1e-3
        )
        next_pressure_difference = float(
            np.clip(
                next_pressure_difference,
                -max_pressure_difference,
                max_pressure_difference,
            )
        )
        channel_pressure_difference = next_pressure_difference
        raw_total_redistribution_flow = total_conductance * next_pressure_difference
        raw_channel_flow = channel_conductance_ul_pa_s * next_pressure_difference
    else:
        # Do not carry the connected-pool pressure state into two separated
        # pools.  Their port-level pressure is computed directly from volume.
        next_pressure_difference = 0.0
        channel_pressure_difference = pool_level_pressure_pa
        # A transferred volume lowers the donor level and raises the receiver
        # level.  Solve this local free-surface feedback implicitly so a low
        # channel resistance cannot overshoot and reverse the two pool volumes.
        raw_channel_flow = channel_conductance_ul_pa_s * channel_pressure_difference / (
            1.0 + channel_conductance_ul_pa_s * time_step / compliance
        )
        raw_total_redistribution_flow = raw_channel_flow
    total_redistribution_flow = limit_single_reservoir_flow_by_volume(
        raw_total_redistribution_flow,
        left_volume,
        right_volume,
        compartments,
        time_step,
    )
    if abs(raw_total_redistribution_flow) > 1e-12:
        flow_scale = total_redistribution_flow / raw_total_redistribution_flow
    else:
        flow_scale = 0.0
    channel_flow = raw_channel_flow * flow_scale
    if channel_conductance_ul_pa_s > 0.0:
        channel_pressure_difference = channel_flow / channel_conductance_ul_pa_s
    driving_head_mm = channel_pressure_difference / (density * gravity) * 1e3
    return (
        channel_flow,
        total_redistribution_flow,
        h_left,
        h_right,
        driving_head_mm,
        pool_state["connected"],
        next_pressure_difference,
    )


def build_single_reservoir_compartments(reservoir_geometry):
    cached_compartments = reservoir_geometry.get("_single_reservoir_compartments")
    if cached_compartments is not None:
        return cached_compartments

    control_geometry = (
        reservoir_geometry
        if reservoir_geometry["type"] == "profile"
        else build_animation_reservoir_geometry(reservoir_geometry)
    )
    if uses_stl_footprint_partition(control_geometry):
        footprint_context = get_stl_footprint_context(control_geometry)
        cached_compartments = {
            "control_geometry": control_geometry,
            "uses_footprint_partition": True,
            "left_capacity_ul": footprint_context["left_capacity_ul"],
            "right_capacity_ul": footprint_context["right_capacity_ul"],
            "left_head_compliance": max(
                footprint_context["left_capacity_ul"] / max(control_geometry["height"], 1e-6),
                1e-6,
            ),
            "right_head_compliance": max(
                footprint_context["right_capacity_ul"] / max(control_geometry["height"], 1e-6),
                1e-6,
            ),
        }
        reservoir_geometry["_single_reservoir_compartments"] = cached_compartments
        return cached_compartments

    profile_points = np.asarray(control_geometry["profile_points"], dtype=float)
    left_port = control_geometry["left_port"]
    right_port = control_geometry["right_port"]
    split_x = 0.5 * (float(left_port["x"]) + float(right_port["x"]))
    min_x = float(np.min(profile_points[:, 0]))
    max_x = float(np.max(profile_points[:, 0]))
    split_x = float(np.clip(split_x, min_x + 1e-6, max_x - 1e-6))

    left_profile = clip_polygon_against_vertical_line(profile_points, split_x, keep_left=True)
    right_profile = clip_polygon_against_vertical_line(profile_points, split_x, keep_left=False)
    if len(left_profile) < 3 or len(right_profile) < 3:
        raise ValueError("Could not split the reservoir into left/right local compartments.")
    top_view_points = np.asarray(control_geometry["top_view_points"], dtype=float)
    left_top_view = clip_polygon_against_vertical_line(top_view_points, split_x, keep_left=True)
    right_top_view = clip_polygon_against_vertical_line(top_view_points, split_x, keep_left=False)
    if len(left_top_view) < 3 or len(right_top_view) < 3:
        raise ValueError("Could not split the reservoir top view into left/right local compartments.")
    source_path = control_geometry.get("source_path")
    extrusion_axis = control_geometry.get("extrusion_axis", "y")
    profile_axes = control_geometry.get("profile_axes")
    axis_mins = control_geometry.get("axis_mins")
    axis_maxs = control_geometry.get("axis_maxs")
    axis_names = control_geometry.get("axis_names")
    width = control_geometry["width"]
    v_fill_ml = control_geometry["v_fill_ml"]
    shared_height = 0.5 * (float(left_port["y"]) + float(right_port["y"]))
    left_dummy_port = [split_x, shared_height]
    right_dummy_port = [split_x, shared_height]

    left_geometry = build_profile_reservoir_geometry(
        left_profile,
        width,
        v_fill_ml,
        [left_port["x"], left_port["y"]],
        left_dummy_port,
        source_path=source_path,
        extrusion_axis=extrusion_axis,
        profile_axes=profile_axes,
        top_view_points=left_top_view,
        left_port_world=control_geometry.get("left_port_world"),
        right_port_world=control_geometry.get("left_port_world"),
        axis_mins=axis_mins,
        axis_maxs=axis_maxs,
        axis_names=axis_names,
    )
    right_geometry = build_profile_reservoir_geometry(
        right_profile,
        width,
        v_fill_ml,
        right_dummy_port,
        [right_port["x"], right_port["y"]],
        source_path=source_path,
        extrusion_axis=extrusion_axis,
        profile_axes=profile_axes,
        top_view_points=right_top_view,
        left_port_world=control_geometry.get("right_port_world"),
        right_port_world=control_geometry.get("right_port_world"),
        axis_mins=axis_mins,
        axis_maxs=axis_maxs,
        axis_names=axis_names,
    )
    left_geometry["left_port_width"] = control_geometry["left_port_width"]
    right_geometry["right_port_width"] = control_geometry["right_port_width"]

    cached_compartments = {
        "control_geometry": control_geometry,
        "uses_footprint_partition": False,
        "split_x_global": split_x,
        "left_x_offset_global": float(left_geometry.get("profile_offset_x", 0.0)),
        "right_x_offset_global": float(right_geometry.get("profile_offset_x", 0.0)),
        "left_width_offset_global": float(left_geometry.get("top_view_offset_y", 0.0)),
        "right_width_offset_global": float(right_geometry.get("top_view_offset_y", 0.0)),
        "left_z_offset_global": float(left_geometry.get("profile_offset_y", 0.0)),
        "right_z_offset_global": float(right_geometry.get("profile_offset_y", 0.0)),
        "left_profile_global": np.asarray(left_profile, dtype=float),
        "right_profile_global": np.asarray(right_profile, dtype=float),
        "left_top_view_global": np.asarray(left_top_view, dtype=float),
        "right_top_view_global": np.asarray(right_top_view, dtype=float),
        "left_geometry": left_geometry,
        "right_geometry": right_geometry,
        "left_head_compliance": max(
            left_geometry["volume_ul_max"] / max(left_geometry["height"], 1e-6),
            1e-6,
        ),
        "right_head_compliance": max(
            right_geometry["volume_ul_max"] / max(right_geometry["height"], 1e-6),
            1e-6,
        ),
        "equilibrium_volume_cache": {},
    }
    reservoir_geometry["_single_reservoir_compartments"] = cached_compartments
    return cached_compartments


def get_single_reservoir_feedback_heads(
    total_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    volume_offset,
):
    compartments = build_single_reservoir_compartments(reservoir_geometry)
    control_geometry = compartments["control_geometry"]
    if compartments["uses_footprint_partition"]:
        pool_state = get_stl_port_pool_state(
            total_volume,
            signed_tilt_length,
            signed_tilt_width,
            control_geometry,
            volume_offset,
        )
        compartments["pools_connected"] = pool_state["connected"]
        return (
            pool_state["left_head"],
            pool_state["right_head"],
            pool_state["volume_offset"],
        )
    else:
        base_left_head, base_right_head = get_physical_port_heads(
            total_volume,
            signed_tilt_length,
            signed_tilt_width,
            control_geometry,
        )
        left_capacity = compartments["left_geometry"]["volume_ul_max"]
        right_capacity = compartments["right_geometry"]["volume_ul_max"]
    clamped_offset = float(
        np.clip(
            volume_offset,
            -right_capacity,
            left_capacity,
        )
    )
    left_head = max(
        0.0,
        base_left_head - clamped_offset / compartments["left_head_compliance"],
    )
    right_head = max(
        0.0,
        base_right_head + clamped_offset / compartments["right_head_compliance"],
    )
    return left_head, right_head, clamped_offset


def get_single_reservoir_local_volumes(
    total_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
    volume_offset,
):
    compartments = build_single_reservoir_compartments(reservoir_geometry)
    control_geometry = compartments["control_geometry"]
    if compartments["uses_footprint_partition"]:
        pool_state = get_stl_port_pool_state(
            total_volume,
            signed_tilt_length,
            signed_tilt_width,
            control_geometry,
            volume_offset,
        )
        compartments["pools_connected"] = pool_state["connected"]
        return (
            compartments,
            pool_state["left_volume"],
            pool_state["right_volume"],
            pool_state["volume_offset"],
        )

    left_geometry = compartments["left_geometry"]
    right_geometry = compartments["right_geometry"]
    left_profile_global = compartments["left_profile_global"]
    right_profile_global = compartments["right_profile_global"]
    cache_resolution = control_geometry["_head_cache_resolution"]
    volume_step = cache_resolution["volume_ul"]
    tilt_length_step = cache_resolution["tilt_length_rad"]
    tilt_width_step = cache_resolution["tilt_width_rad"]
    equilibrium_cache_key = (
        int(round(float(total_volume) / volume_step)),
        int(round(float(signed_tilt_length) / tilt_length_step)),
        int(round(float(signed_tilt_width) / tilt_width_step)),
    )
    cached_equilibrium = compartments["equilibrium_volume_cache"].get(equilibrium_cache_key)
    if cached_equilibrium is None:
        quantized_volume = min(
            control_geometry["volume_ul_max"],
            equilibrium_cache_key[0] * volume_step,
        )
        quantized_tilt_length = equilibrium_cache_key[1] * tilt_length_step
        quantized_tilt_width = equilibrium_cache_key[2] * tilt_width_step
        slope_length = math.tan(quantized_tilt_length)
        slope_width = math.tan(quantized_tilt_width)
        intercept = solve_profile_free_surface_3d(
            control_geometry,
            quantized_volume,
            slope_length,
            slope_width,
        )
        equilibrium_left = integrate_profile_volume(
            left_profile_global,
            control_geometry["width"],
            slope_length,
            slope_width,
            intercept,
        )
        equilibrium_right = integrate_profile_volume(
            right_profile_global,
            control_geometry["width"],
            slope_length,
            slope_width,
            intercept,
        )
        equilibrium_total = equilibrium_left + equilibrium_right
        if equilibrium_total > 0:
            scale_factor = float(quantized_volume) / float(equilibrium_total)
            equilibrium_left *= scale_factor
            equilibrium_right *= scale_factor
        cached_equilibrium = (equilibrium_left, equilibrium_right)
        compartments["equilibrium_volume_cache"][equilibrium_cache_key] = cached_equilibrium
    else:
        equilibrium_left, equilibrium_right = cached_equilibrium

    # Cached free-surface states are quantized for speed.  Preserve the exact
    # simulated total when converting that cached split back to local volumes.
    equilibrium_total = equilibrium_left + equilibrium_right
    target_volume = float(
        np.clip(total_volume, 0.0, control_geometry["volume_ul_max"])
    )
    if equilibrium_total > 0.0:
        scale_factor = target_volume / equilibrium_total
        equilibrium_left *= scale_factor
        equilibrium_right *= scale_factor

    min_offset = max(
        equilibrium_left - left_geometry["volume_ul_max"],
        -equilibrium_right,
    )
    max_offset = min(
        equilibrium_left,
        right_geometry["volume_ul_max"] - equilibrium_right,
    )
    clamped_offset = float(np.clip(volume_offset, min_offset, max_offset))
    left_volume = equilibrium_left - clamped_offset
    right_volume = equilibrium_right + clamped_offset

    return compartments, left_volume, right_volume, clamped_offset


def initialize_single_reservoir_half_volumes(
    total_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
):
    (
        _compartments,
        left_volume,
        right_volume,
        _clamped_offset,
    ) = get_single_reservoir_local_volumes(
        total_volume,
        signed_tilt_length,
        signed_tilt_width,
        reservoir_geometry,
        0.0,
    )
    return left_volume, right_volume


def project_single_reservoir_half_volumes(total_volume, left_volume, compartments):
    left_max = compartments["left_geometry"]["volume_ul_max"]
    right_max = compartments["right_geometry"]["volume_ul_max"]
    left_volume = float(np.clip(left_volume, 0.0, left_max))
    right_volume = float(total_volume - left_volume)

    if right_volume < 0.0:
        right_volume = 0.0
        left_volume = float(total_volume)
    if right_volume > right_max:
        right_volume = float(right_max)
        left_volume = float(total_volume - right_volume)

    left_volume = float(np.clip(left_volume, 0.0, left_max))
    right_volume = float(total_volume - left_volume)
    return left_volume, right_volume


def get_single_reservoir_half_port_heads(
    left_volume,
    right_volume,
    signed_tilt_length,
    signed_tilt_width,
    reservoir_geometry,
):
    compartments = build_single_reservoir_compartments(reservoir_geometry)
    left_geometry = compartments["left_geometry"]
    right_geometry = compartments["right_geometry"]
    left_head, _left_internal_head = get_physical_port_heads(
        left_volume,
        signed_tilt_length,
        signed_tilt_width,
        left_geometry,
    )
    _right_internal_head, right_head = get_physical_port_heads(
        right_volume,
        signed_tilt_length,
        signed_tilt_width,
        right_geometry,
    )
    return left_head, right_head


def build_animation_reservoir_geometry(reservoir_geometry):
    if reservoir_geometry["type"] == "profile":
        return reservoir_geometry

    profile_points = np.asarray(
        [
            [0.0, 0.0],
            [reservoir_geometry["length"], 0.0],
            [reservoir_geometry["length"], reservoir_geometry["height"]],
            [0.0, reservoir_geometry["height"]],
        ],
        dtype=float,
    )
    mid_width = reservoir_geometry["width"] / 2
    top_view_points = np.asarray(
        [
            [0.0, 0.0],
            [reservoir_geometry["length"], 0.0],
            [reservoir_geometry["length"], reservoir_geometry["width"]],
            [0.0, reservoir_geometry["width"]],
        ],
        dtype=float,
    )
    top_view_triangles = np.asarray(
        [
            [[0.0, 0.0], [reservoir_geometry["length"], 0.0], [reservoir_geometry["length"], reservoir_geometry["width"]]],
            [[0.0, 0.0], [reservoir_geometry["length"], reservoir_geometry["width"]], [0.0, reservoir_geometry["width"]]],
        ],
        dtype=float,
    )
    shell_triangles = np.asarray(
        [
            [[0.0, 0.0, 0.0], [reservoir_geometry["length"], 0.0, 0.0], [reservoir_geometry["length"], reservoir_geometry["width"], 0.0]],
            [[0.0, 0.0, 0.0], [reservoir_geometry["length"], reservoir_geometry["width"], 0.0], [0.0, reservoir_geometry["width"], 0.0]],
            [[0.0, 0.0, reservoir_geometry["height"]], [reservoir_geometry["length"], 0.0, reservoir_geometry["height"]], [reservoir_geometry["length"], reservoir_geometry["width"], reservoir_geometry["height"]]],
            [[0.0, 0.0, reservoir_geometry["height"]], [reservoir_geometry["length"], reservoir_geometry["width"], reservoir_geometry["height"]], [0.0, reservoir_geometry["width"], reservoir_geometry["height"]]],
            [[0.0, 0.0, 0.0], [reservoir_geometry["length"], 0.0, 0.0], [reservoir_geometry["length"], 0.0, reservoir_geometry["height"]]],
            [[0.0, 0.0, 0.0], [reservoir_geometry["length"], 0.0, reservoir_geometry["height"]], [0.0, 0.0, reservoir_geometry["height"]]],
            [[0.0, reservoir_geometry["width"], 0.0], [reservoir_geometry["length"], reservoir_geometry["width"], 0.0], [reservoir_geometry["length"], reservoir_geometry["width"], reservoir_geometry["height"]]],
            [[0.0, reservoir_geometry["width"], 0.0], [reservoir_geometry["length"], reservoir_geometry["width"], reservoir_geometry["height"]], [0.0, reservoir_geometry["width"], reservoir_geometry["height"]]],
            [[0.0, 0.0, 0.0], [0.0, reservoir_geometry["width"], 0.0], [0.0, reservoir_geometry["width"], reservoir_geometry["height"]]],
            [[0.0, 0.0, 0.0], [0.0, reservoir_geometry["width"], reservoir_geometry["height"]], [0.0, 0.0, reservoir_geometry["height"]]],
            [[reservoir_geometry["length"], 0.0, 0.0], [reservoir_geometry["length"], reservoir_geometry["width"], 0.0], [reservoir_geometry["length"], reservoir_geometry["width"], reservoir_geometry["height"]]],
            [[reservoir_geometry["length"], 0.0, 0.0], [reservoir_geometry["length"], reservoir_geometry["width"], reservoir_geometry["height"]], [reservoir_geometry["length"], 0.0, reservoir_geometry["height"]]],
        ],
        dtype=float,
    )
    return build_profile_reservoir_geometry(
        profile_points,
        reservoir_geometry["width"],
        reservoir_geometry["v_fill_ml"],
        [reservoir_geometry["left_port"]["x"], 0.0],
        [reservoir_geometry["right_port"]["x"], 0.0],
        source_path="rectangular-default",
        extrusion_axis="y",
        profile_axes=("x", "z"),
        top_view_points=top_view_points,
        top_view_triangles=top_view_triangles,
        shell_triangles=shell_triangles,
        left_port_world={"x": 0.0, "y": mid_width, "z": 0.0},
        right_port_world={"x": reservoir_geometry["length"], "y": mid_width, "z": 0.0},
        axis_mins=[0.0, 0.0, 0.0],
        axis_maxs=[
            reservoir_geometry["length"],
            reservoir_geometry["width"],
            reservoir_geometry["height"],
        ],
        axis_names=stl_axis_names,
    )


def mirror_reservoir_geometry_across_width(reservoir_geometry):
    """Reflect a reservoir and its selected ports across its width centreline."""
    mirrored_geometry = copy.deepcopy(reservoir_geometry)
    if "left_port_width" not in mirrored_geometry:
        # Analytic rectangular reservoirs have centred width ports and are
        # unchanged by a width reflection.
        return mirrored_geometry
    footprint_width = get_animation_plot_extents(reservoir_geometry)[1]
    for key in ("top_view_points", "top_view_triangles", "shell_triangles"):
        if mirrored_geometry.get(key) is not None:
            values = np.asarray(mirrored_geometry[key], dtype=float).copy()
            values[..., 1] = footprint_width - values[..., 1]
            mirrored_geometry[key] = values
    for port_name in ("left", "right"):
        width_key = f"{port_name}_port_width"
        mirrored_geometry[width_key] = (
            footprint_width - float(reservoir_geometry[width_key])
        )
    # A reflected STL needs a new footprint and water-level cache. The cache
    # settings themselves are kept so both reservoir sides use the same grid.
    for cache_key in (
        "_stl_footprint_context",
        "_free_surface_cache",
    ):
        mirrored_geometry.pop(cache_key, None)
    if mirrored_geometry.get("type") == "profile":
        mirrored_geometry["_head_cache"] = {}
        mirrored_geometry["_physical_head_cache"] = {}
    return mirrored_geometry


def mirror_animation_reservoir_across_width(reservoir_geometry):
    """Reflect an animation reservoir so its interior faces the first reservoir."""
    return mirror_reservoir_geometry_across_width(reservoir_geometry)


def get_animation_plot_extents(reservoir_geometry):
    shell_triangles = reservoir_geometry.get("shell_triangles")
    if shell_triangles is not None:
        shell_points = np.asarray(shell_triangles, dtype=float).reshape(-1, 3)
        return (
            float(np.max(shell_points[:, 0])),
            float(np.max(shell_points[:, 1])),
            float(np.max(shell_points[:, 2])),
        )

    top_view_points = np.asarray(reservoir_geometry["top_view_points"], dtype=float)
    profile_points = np.asarray(reservoir_geometry["profile_points"], dtype=float)
    return (
        float(np.max(top_view_points[:, 0])),
        float(np.max(top_view_points[:, 1])),
        float(np.max(profile_points[:, 1])),
    )


def get_profile_vertical_bounds(profile_points, x_value, tolerance=1e-9):
    intersections = []
    point_count = len(profile_points)
    for index in range(point_count):
        start = np.asarray(profile_points[index], dtype=float)
        end = np.asarray(profile_points[(index + 1) % point_count], dtype=float)
        min_x = min(start[0], end[0])
        max_x = max(start[0], end[0])
        if (x_value < min_x - tolerance) or (x_value > max_x + tolerance):
            continue

        delta_x = end[0] - start[0]
        if abs(delta_x) <= tolerance:
            if abs(x_value - start[0]) <= tolerance:
                intersections.extend((start[1], end[1]))
            continue

        t_value = (x_value - start[0]) / delta_x
        if -tolerance <= t_value <= 1.0 + tolerance:
            intersections.append(start[1] + t_value * (end[1] - start[1]))

    if len(intersections) < 2:
        return None

    unique_intersections = []
    for intersection in sorted(intersections):
        if (not unique_intersections) or abs(intersection - unique_intersections[-1]) > tolerance:
            unique_intersections.append(intersection)

    if len(unique_intersections) < 2:
        return None
    return float(unique_intersections[0]), float(unique_intersections[-1])


def get_shell_vertical_bounds(shell_triangles, x_value, y_value, tolerance=1e-9):
    intersections = []
    for triangle in np.asarray(shell_triangles, dtype=float):
        point_a = triangle[0]
        point_b = triangle[1]
        point_c = triangle[2]
        denominator = (
            (point_b[1] - point_c[1]) * (point_a[0] - point_c[0])
            + (point_c[0] - point_b[0]) * (point_a[1] - point_c[1])
        )
        if abs(denominator) <= tolerance:
            continue

        alpha = (
            (point_b[1] - point_c[1]) * (x_value - point_c[0])
            + (point_c[0] - point_b[0]) * (y_value - point_c[1])
        ) / denominator
        beta = (
            (point_c[1] - point_a[1]) * (x_value - point_c[0])
            + (point_a[0] - point_c[0]) * (y_value - point_c[1])
        ) / denominator
        gamma = 1.0 - alpha - beta
        if (alpha < -tolerance) or (beta < -tolerance) or (gamma < -tolerance):
            continue

        z_value = alpha * point_a[2] + beta * point_b[2] + gamma * point_c[2]
        intersections.append(float(z_value))

    if len(intersections) < 2:
        return None

    unique_intersections = []
    for intersection in sorted(intersections):
        if (not unique_intersections) or abs(intersection - unique_intersections[-1]) > tolerance:
            unique_intersections.append(intersection)

    if len(unique_intersections) < 2:
        return None
    return float(unique_intersections[0]), float(unique_intersections[-1])


def build_top_view_point_mask(reservoir_geometry, x_grid, width_grid):
    top_view_triangles = reservoir_geometry.get("top_view_triangles")
    if top_view_triangles is None:
        top_view_path = MplPath(reservoir_geometry["top_view_points"])
        return top_view_path.contains_points(
            np.column_stack([x_grid.ravel(), width_grid.ravel()]),
            radius=1e-9,
        ).reshape(x_grid.shape)

    flat_points = np.column_stack([x_grid.ravel(), width_grid.ravel()])
    point_mask = np.zeros(len(flat_points), dtype=bool)
    tolerance = 1e-9
    for triangle in np.asarray(top_view_triangles, dtype=float):
        point_a = triangle[0]
        point_b = triangle[1]
        point_c = triangle[2]
        denominator = (
            (point_b[1] - point_c[1]) * (point_a[0] - point_c[0])
            + (point_c[0] - point_b[0]) * (point_a[1] - point_c[1])
        )
        if abs(denominator) <= tolerance:
            continue

        alpha = (
            (point_b[1] - point_c[1]) * (flat_points[:, 0] - point_c[0])
            + (point_c[0] - point_b[0]) * (flat_points[:, 1] - point_c[1])
        ) / denominator
        beta = (
            (point_c[1] - point_a[1]) * (flat_points[:, 0] - point_c[0])
            + (point_a[0] - point_c[0]) * (flat_points[:, 1] - point_c[1])
        ) / denominator
        gamma = 1.0 - alpha - beta
        point_mask |= (
            (alpha >= -tolerance)
            & (beta >= -tolerance)
            & (gamma >= -tolerance)
        )

    return point_mask.reshape(x_grid.shape)


def build_shell_vertical_bounds_mesh(reservoir_geometry, x_grid, width_grid, point_mask):
    shell_triangles = reservoir_geometry.get("shell_triangles")
    bottom_grid = np.full_like(x_grid, np.nan, dtype=float)
    top_grid = np.full_like(x_grid, np.nan, dtype=float)

    if shell_triangles is None:
        for x_index, x_value in enumerate(x_grid[0]):
            bounds = get_profile_vertical_bounds(reservoir_geometry["profile_points"], x_value)
            if bounds is None:
                continue
            bottom_z, top_z = bounds
            bottom_grid[:, x_index] = np.where(point_mask[:, x_index], bottom_z, np.nan)
            top_grid[:, x_index] = np.where(point_mask[:, x_index], top_z, np.nan)
        return bottom_grid, top_grid

    row_count, column_count = x_grid.shape
    for row_index in range(row_count):
        for column_index in range(column_count):
            if not point_mask[row_index, column_index]:
                continue
            bounds = get_shell_vertical_bounds(
                shell_triangles,
                float(x_grid[row_index, column_index]),
                float(width_grid[row_index, column_index]),
            )
            if bounds is None:
                continue
            bottom_grid[row_index, column_index], top_grid[row_index, column_index] = bounds

    return bottom_grid, top_grid


def build_water_surface_mesh(
    reservoir_geometry,
    intercept,
    slope_length,
    slope_width,
    x_grid,
    width_grid,
    point_mask,
    bottom_grid,
    top_grid,
    x_count=70,
    width_count=30,
):
    z_grid = np.full_like(x_grid, np.nan, dtype=float)

    plane_heights = intercept - slope_length * x_grid - slope_width * width_grid
    visible = point_mask & np.isfinite(bottom_grid) & np.isfinite(top_grid) & (plane_heights > bottom_grid)
    z_grid = np.where(visible, np.minimum(plane_heights, top_grid), np.nan)

    return x_grid, width_grid, z_grid


def draw_reservoir_wireframe(
    axis,
    reservoir_geometry,
    x_offset=0.0,
    width_offset=0.0,
    z_offset=0.0,
):
    shell_triangles = reservoir_geometry.get("shell_triangles")
    if shell_triangles is not None:
        translated_triangles = np.asarray(shell_triangles, dtype=float).copy()
        translated_triangles[:, :, 0] += float(x_offset)
        translated_triangles[:, :, 1] += float(width_offset)
        translated_triangles[:, :, 2] += float(z_offset)
        shell_collection = Poly3DCollection(
            translated_triangles,
            facecolors=(0, 0, 0, 0),
            edgecolors=(0.3, 0.3, 0.3, 0.7),
            linewidths=0.45,
        )
        axis.add_collection3d(shell_collection)
        return

    top_view_triangles = reservoir_geometry.get("top_view_triangles")
    if top_view_triangles is not None:
        boundary_loops = extract_projected_boundaries(top_view_triangles)
    else:
        boundary_loops = [reservoir_geometry["top_view_points"]]

    for top_view in boundary_loops:
        closed_top_view = np.vstack([top_view, top_view[0]])
        top_boundary_z = []
        bottom_boundary_z = []
        for point in closed_top_view:
            bounds = get_profile_vertical_bounds(reservoir_geometry["profile_points"], float(point[0]))
            if bounds is None:
                bottom_boundary_z.append(np.nan)
                top_boundary_z.append(np.nan)
                continue
            z_bottom, z_top = bounds
            bottom_boundary_z.append(z_bottom)
            top_boundary_z.append(z_top)

        axis.plot(
            closed_top_view[:, 0] + float(x_offset),
            closed_top_view[:, 1] + float(width_offset),
            np.asarray(bottom_boundary_z) + float(z_offset),
            color="0.4",
            linewidth=0.8,
        )
        axis.plot(
            closed_top_view[:, 0] + float(x_offset),
            closed_top_view[:, 1] + float(width_offset),
            np.asarray(top_boundary_z) + float(z_offset),
            color="0.35",
            linewidth=0.9,
        )

        connector_indices = np.linspace(0, len(top_view) - 1, min(len(top_view), 24), dtype=int)
        for index in connector_indices:
            axis.plot(
                [top_view[index, 0] + float(x_offset), top_view[index, 0] + float(x_offset)],
                [top_view[index, 1] + float(width_offset), top_view[index, 1] + float(width_offset)],
                [bottom_boundary_z[index] + float(z_offset), top_boundary_z[index] + float(z_offset)],
                color="0.55",
                linewidth=0.6,
            )


def interpolate_series(time_points, values, sample_times):
    return np.interp(sample_times, time_points, values)


def build_animation_surface_context(reservoir_geometry, x_count=70, width_count=30):
    plot_length, footprint_width, plot_height = get_animation_plot_extents(reservoir_geometry)
    # Sample at cell centres.  This makes the numerical volume integral include
    # irregular STL footprints without over-counting the outer boundary cells.
    surface_x_edges = np.linspace(0.0, plot_length, x_count + 1)
    surface_width_edges = np.linspace(0.0, footprint_width, width_count + 1)
    x_values = 0.5 * (surface_x_edges[:-1] + surface_x_edges[1:])
    width_values = 0.5 * (surface_width_edges[:-1] + surface_width_edges[1:])
    surface_x_grid, surface_width_grid = np.meshgrid(x_values, width_values)
    point_mask = build_top_view_point_mask(
        reservoir_geometry,
        surface_x_grid,
        surface_width_grid,
    )
    bottom_grid, top_grid = build_shell_vertical_bounds_mesh(
        reservoir_geometry,
        surface_x_grid,
        surface_width_grid,
        point_mask,
    )
    # Curved or slotted STL footprints need more lateral samples than a
    # rectangular reservoir.  The cap keeps GIF generation responsive.
    voxel_x_count = int(np.clip(round(plot_length * 1.8), 26, 44))
    voxel_width_count = int(np.clip(round(footprint_width * 1.8), 12, 44))
    voxel_z_count = int(np.clip(round(plot_height * 3.0), 18, 28))
    voxel_x_edges = np.linspace(0.0, plot_length, voxel_x_count + 1)
    voxel_width_edges = np.linspace(0.0, footprint_width, voxel_width_count + 1)
    voxel_z_edges = np.linspace(0.0, plot_height, voxel_z_count + 1)
    voxel_x_centers = 0.5 * (voxel_x_edges[:-1] + voxel_x_edges[1:])
    voxel_width_centers = 0.5 * (voxel_width_edges[:-1] + voxel_width_edges[1:])
    voxel_z_centers = 0.5 * (voxel_z_edges[:-1] + voxel_z_edges[1:])
    voxel_x_grid, voxel_width_grid = np.meshgrid(
        voxel_x_centers,
        voxel_width_centers,
        indexing="ij",
    )
    voxel_point_mask = build_top_view_point_mask(
        reservoir_geometry,
        voxel_x_grid,
        voxel_width_grid,
    )
    voxel_bottom_grid, voxel_top_grid = build_shell_vertical_bounds_mesh(
        reservoir_geometry,
        voxel_x_grid,
        voxel_width_grid,
        voxel_point_mask,
    )
    voxel_x_corners, voxel_width_corners, voxel_z_corners = np.meshgrid(
        voxel_x_edges,
        voxel_width_edges,
        voxel_z_edges,
        indexing="ij",
    )
    return {
        "plot_length": plot_length,
        "footprint_width": footprint_width,
        "plot_height": plot_height,
        "surface_x_grid": surface_x_grid,
        "surface_width_grid": surface_width_grid,
        "surface_cell_area": (
            float(surface_x_edges[1] - surface_x_edges[0])
            * float(surface_width_edges[1] - surface_width_edges[0])
        ),
        "point_mask": point_mask,
        "bottom_grid": bottom_grid,
        "top_grid": top_grid,
        "voxel_x_grid": voxel_x_grid,
        "voxel_width_grid": voxel_width_grid,
        "voxel_z_centers": voxel_z_centers,
        "voxel_z_lower": voxel_z_edges[:-1],
        "voxel_z_upper": voxel_z_edges[1:],
        "voxel_point_mask": voxel_point_mask,
        "voxel_bottom_grid": voxel_bottom_grid,
        "voxel_top_grid": voxel_top_grid,
        "voxel_x_corners": voxel_x_corners,
        "voxel_width_corners": voxel_width_corners,
        "voxel_z_corners": voxel_z_corners,
    }


def solve_animation_free_surface_3d(
    animation_surface,
    target_volume,
    slope_length,
    slope_width,
):
    """Solve a liquid plane against the actual sampled STL footprint.

    This is shared by the STL animation and the single-reservoir pool model so
    slots, arcs, and C-shaped cells use their imported shell footprint.
    """
    x_grid = animation_surface["surface_x_grid"]
    width_grid = animation_surface["surface_width_grid"]
    bottom_grid = animation_surface["bottom_grid"]
    top_grid = animation_surface["top_grid"]
    valid = (
        animation_surface["point_mask"]
        & np.isfinite(bottom_grid)
        & np.isfinite(top_grid)
    )
    if not np.any(valid):
        raise ValueError("Imported STL has no valid cells for animation.")

    plane_offset = slope_length * x_grid + slope_width * width_grid
    lower_bound = float(np.min((bottom_grid + plane_offset)[valid]))
    upper_bound = float(np.max((top_grid + plane_offset)[valid]))
    cell_area = float(animation_surface["surface_cell_area"])

    def liquid_volume(intercept):
        liquid_top = np.minimum(intercept - plane_offset, top_grid)
        liquid_height = np.where(
            valid,
            np.maximum(0.0, liquid_top - bottom_grid),
            0.0,
        )
        return float(np.sum(liquid_height) * cell_area)

    max_volume = liquid_volume(upper_bound)
    if target_volume <= 0.0:
        return lower_bound
    if target_volume >= max_volume:
        return upper_bound

    for _ in range(60):
        midpoint = 0.5 * (lower_bound + upper_bound)
        if liquid_volume(midpoint) < target_volume:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    return 0.5 * (lower_bound + upper_bound)


def compute_animation_voxel_fill(
    reservoir_geometry,
    animation_surface,
    target_volume,
    slope_length,
    slope_width,
):
    intercept = solve_animation_free_surface_3d(
        animation_surface,
        float(target_volume),
        slope_length,
        slope_width,
    )

    voxel_x_grid = animation_surface["voxel_x_grid"]
    voxel_width_grid = animation_surface["voxel_width_grid"]
    voxel_z_lower = animation_surface["voxel_z_lower"]
    voxel_z_upper = animation_surface["voxel_z_upper"]
    voxel_point_mask = animation_surface["voxel_point_mask"]
    voxel_bottom_grid = animation_surface["voxel_bottom_grid"]
    voxel_top_grid = animation_surface["voxel_top_grid"]

    voxel_plane_heights = intercept - slope_length * voxel_x_grid - slope_width * voxel_width_grid
    voxel_liquid_top = np.minimum(voxel_plane_heights, voxel_top_grid)
    z_lower_grid = voxel_z_lower[np.newaxis, np.newaxis, :]
    z_upper_grid = voxel_z_upper[np.newaxis, np.newaxis, :]
    cell_bottom = np.maximum(voxel_bottom_grid[:, :, np.newaxis], z_lower_grid)
    cell_top = np.minimum(voxel_liquid_top[:, :, np.newaxis], z_upper_grid)
    fill_height = np.maximum(0.0, cell_top - cell_bottom)
    cell_height = np.maximum(z_upper_grid - z_lower_grid, 1e-12)
    fill_fraction = np.nan_to_num(
        fill_height / cell_height,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    filled = (
        voxel_point_mask[:, :, np.newaxis]
        & np.isfinite(voxel_bottom_grid)[:, :, np.newaxis]
        & np.isfinite(voxel_top_grid)[:, :, np.newaxis]
        & (fill_fraction > 1e-6)
    )
    return {
        "filled": filled,
        "fill_fraction": fill_fraction,
    }


def get_animation_port_partitions(reservoir_geometry, animation_surface):
    partition_key = (
        float(reservoir_geometry["left_port"]["x"]),
        float(reservoir_geometry["left_port_width"]),
        float(reservoir_geometry["right_port"]["x"]),
        float(reservoir_geometry["right_port_width"]),
    )
    if animation_surface.get("_animation_port_partition_key") == partition_key:
        return animation_surface["_animation_port_partitions"]

    surface_valid = (
        animation_surface["point_mask"]
        & np.isfinite(animation_surface["bottom_grid"])
        & np.isfinite(animation_surface["top_grid"])
    )
    voxel_valid = (
        animation_surface["voxel_point_mask"]
        & np.isfinite(animation_surface["voxel_bottom_grid"])
        & np.isfinite(animation_surface["voxel_top_grid"])
    )
    (
        _surface_left_distance,
        _surface_right_distance,
        surface_left_region,
        surface_right_region,
    ) = get_port_connected_footprint_partition(
        reservoir_geometry,
        animation_surface["surface_x_grid"],
        animation_surface["surface_width_grid"],
        surface_valid,
    )
    (
        _voxel_left_distance,
        _voxel_right_distance,
        voxel_left_region,
        voxel_right_region,
    ) = get_port_connected_footprint_partition(
        reservoir_geometry,
        animation_surface["voxel_x_grid"],
        animation_surface["voxel_width_grid"],
        voxel_valid,
    )
    partitions = {
        "surface_left_region": surface_left_region,
        "surface_right_region": surface_right_region,
        "voxel_left_region": voxel_left_region,
        "voxel_right_region": voxel_right_region,
    }
    animation_surface["_animation_port_partition_key"] = partition_key
    animation_surface["_animation_port_partitions"] = partitions
    return partitions


def compute_partition_voxel_fill(
    animation_surface,
    intercept,
    slope_length,
    slope_width,
    voxel_region,
):
    voxel_plane_heights = (
        intercept
        - slope_length * animation_surface["voxel_x_grid"]
        - slope_width * animation_surface["voxel_width_grid"]
    )
    voxel_liquid_top = np.minimum(
        voxel_plane_heights,
        animation_surface["voxel_top_grid"],
    )
    z_lower_grid = animation_surface["voxel_z_lower"][np.newaxis, np.newaxis, :]
    z_upper_grid = animation_surface["voxel_z_upper"][np.newaxis, np.newaxis, :]
    cell_bottom = np.maximum(
        animation_surface["voxel_bottom_grid"][:, :, np.newaxis],
        z_lower_grid,
    )
    cell_top = np.minimum(voxel_liquid_top[:, :, np.newaxis], z_upper_grid)
    fill_fraction = np.nan_to_num(
        np.maximum(0.0, cell_top - cell_bottom)
        / np.maximum(z_upper_grid - z_lower_grid, 1e-12),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    filled = voxel_region[:, :, np.newaxis] & (fill_fraction > 1e-6)
    return filled, fill_fraction


def compute_voxel_fill_from_liquid_depth(
    animation_surface,
    liquid_depth,
    voxel_region,
):
    """Convert a solved column-height fill into voxel fractions."""
    bottom_grid = animation_surface["voxel_bottom_grid"]
    top_grid = animation_surface["voxel_top_grid"]
    liquid_top = np.minimum(bottom_grid + liquid_depth, top_grid)
    z_lower_grid = animation_surface["voxel_z_lower"][np.newaxis, np.newaxis, :]
    z_upper_grid = animation_surface["voxel_z_upper"][np.newaxis, np.newaxis, :]
    cell_bottom = np.maximum(bottom_grid[:, :, np.newaxis], z_lower_grid)
    cell_top = np.minimum(liquid_top[:, :, np.newaxis], z_upper_grid)
    fill_fraction = np.nan_to_num(
        np.maximum(0.0, cell_top - cell_bottom)
        / np.maximum(z_upper_grid - z_lower_grid, 1e-12),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    filled = voxel_region[:, :, np.newaxis] & (fill_fraction > 1e-6)
    return filled, fill_fraction


def solve_partition_voxel_intercept(
    animation_surface,
    target_volume,
    slope_length,
    slope_width,
    voxel_region,
):
    """Match a port pool's displayed voxel volume to its model volume."""
    valid_mask = (
        voxel_region
        & np.isfinite(animation_surface["voxel_bottom_grid"])
        & np.isfinite(animation_surface["voxel_top_grid"])
    )
    if not np.any(valid_mask):
        raise ValueError("The animation grid has no valid cells for a port pool.")

    x_grid = animation_surface["voxel_x_grid"]
    width_grid = animation_surface["voxel_width_grid"]
    bottom_grid = animation_surface["voxel_bottom_grid"]
    top_grid = animation_surface["voxel_top_grid"]
    plane_offset = slope_length * x_grid + slope_width * width_grid
    lower_bound = float(np.min((bottom_grid + plane_offset)[valid_mask]))
    upper_bound = float(np.max((top_grid + plane_offset)[valid_mask]))
    cell_dx = float(
        animation_surface["voxel_x_corners"][1, 0, 0]
        - animation_surface["voxel_x_corners"][0, 0, 0]
    )
    cell_dy = float(
        animation_surface["voxel_width_corners"][0, 1, 0]
        - animation_surface["voxel_width_corners"][0, 0, 0]
    )
    cell_area = cell_dx * cell_dy

    def liquid_volume(intercept):
        liquid_top = np.minimum(intercept - plane_offset, top_grid)
        liquid_height = np.where(
            valid_mask,
            np.maximum(0.0, liquid_top - bottom_grid),
            0.0,
        )
        return float(np.sum(liquid_height) * cell_area)

    if target_volume <= 0.0:
        return lower_bound
    if target_volume >= liquid_volume(upper_bound):
        return upper_bound
    for _ in range(60):
        midpoint = 0.5 * (lower_bound + upper_bound)
        if liquid_volume(midpoint) < target_volume:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    return 0.5 * (lower_bound + upper_bound)


def plot_stl_port_pools(
    axis,
    reservoir_geometry,
    animation_surface,
    left_target_volume,
    right_target_volume,
    slope_length,
    slope_width,
    left_color="#ff006e",
    right_color="#00b4d8",
    alpha=0.9,
):
    """Render separate port pools until their wetted regions meet."""
    top_contact_wetting = bool(reservoir_geometry.get("top_contact_wetting", False))
    left_port = reservoir_geometry["left_port"]
    right_port = reservoir_geometry["right_port"]
    left_intercept, left_wet, _ = solve_port_connected_free_surface_3d(
        animation_surface,
        max(float(left_target_volume), 0.0),
        slope_length,
        slope_width,
        left_port,
        reservoir_geometry["left_port_width"],
        top_contact_wetting=top_contact_wetting,
    )
    right_intercept, right_wet, _ = solve_port_connected_free_surface_3d(
        animation_surface,
        max(float(right_target_volume), 0.0),
        slope_length,
        slope_width,
        right_port,
        reservoir_geometry["right_port_width"],
        top_contact_wetting=top_contact_wetting,
    )
    if wet_regions_touch(left_wet, right_wet):
        return False

    voxel_valid = (
        animation_surface["voxel_point_mask"]
        & np.isfinite(animation_surface["voxel_bottom_grid"])
        & np.isfinite(animation_surface["voxel_top_grid"])
    )
    voxel_cell_area = float(
        animation_surface["voxel_x_corners"][1, 0, 0]
        - animation_surface["voxel_x_corners"][0, 0, 0]
    ) * float(
        animation_surface["voxel_width_corners"][0, 1, 0]
        - animation_surface["voxel_width_corners"][0, 0, 0]
    )
    for target_volume, port, port_width, color in (
        (
            max(float(left_target_volume), 0.0),
            left_port,
            reservoir_geometry["left_port_width"],
            left_color,
        ),
        (
            max(float(right_target_volume), 0.0),
            right_port,
            reservoir_geometry["right_port_width"],
            right_color,
        ),
    ):
        _, voxel_region, voxel_liquid_depth = solve_port_connected_liquid_pool(
            animation_surface["voxel_x_grid"],
            animation_surface["voxel_width_grid"],
            animation_surface["voxel_bottom_grid"],
            animation_surface["voxel_top_grid"],
            voxel_valid,
            voxel_cell_area,
            port,
            port_width,
            target_volume,
            slope_length,
            slope_width,
            top_contact_wetting=top_contact_wetting,
        )
        filled, fill_fraction = compute_voxel_fill_from_liquid_depth(
            animation_surface,
            voxel_liquid_depth,
            voxel_region,
        )
        if not np.any(filled):
            continue
        facecolors = np.empty(filled.shape + (4,), dtype=float)
        red_value, green_value, blue_value, _ = to_rgba(color, alpha)
        facecolors[..., 0] = red_value
        facecolors[..., 1] = green_value
        facecolors[..., 2] = blue_value
        facecolors[..., 3] = np.clip(fill_fraction * alpha, 0.0, alpha)
        axis.voxels(
            animation_surface["voxel_x_corners"],
            animation_surface["voxel_width_corners"],
            animation_surface["voxel_z_corners"],
            filled,
            facecolors=facecolors,
            edgecolor=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
        )
    return True


def integrate_profile_volume_left_of_x(
    reservoir_geometry,
    x_limit,
    slope_length,
    slope_width,
    intercept,
):
    profile_points = np.asarray(reservoir_geometry["profile_points"], dtype=float)
    min_x = float(np.min(profile_points[:, 0]))
    max_x = float(np.max(profile_points[:, 0]))
    if x_limit <= min_x:
        return 0.0
    if x_limit >= max_x:
        return integrate_profile_volume(
            profile_points,
            reservoir_geometry["width"],
            slope_length,
            slope_width,
            intercept,
        )

    clipped_profile = clip_polygon_against_vertical_line(
        profile_points,
        float(x_limit),
        keep_left=True,
    )
    if len(clipped_profile) < 3:
        return 0.0
    return integrate_profile_volume(
        clipped_profile,
        reservoir_geometry["width"],
        slope_length,
        slope_width,
        intercept,
    )


def solve_port_anchored_split_x(
    reservoir_geometry,
    target_left_volume,
    slope_length,
    slope_width,
    intercept,
):
    profile_points = np.asarray(reservoir_geometry["profile_points"], dtype=float)
    min_x = float(np.min(profile_points[:, 0]))
    max_x = float(np.max(profile_points[:, 0]))
    total_volume = integrate_profile_volume(
        profile_points,
        reservoir_geometry["width"],
        slope_length,
        slope_width,
        intercept,
    )
    if target_left_volume <= 0.0:
        return min_x
    if target_left_volume >= total_volume:
        return max_x

    lower_bound = min_x
    upper_bound = max_x
    for _ in range(50):
        midpoint = 0.5 * (lower_bound + upper_bound)
        left_volume = integrate_profile_volume_left_of_x(
            reservoir_geometry,
            midpoint,
            slope_length,
            slope_width,
            intercept,
        )
        if left_volume < target_left_volume:
            lower_bound = midpoint
        else:
            upper_bound = midpoint
    return 0.5 * (lower_bound + upper_bound)


def plot_animation_water_surface(
    axis,
    reservoir_geometry,
    animation_surface,
    target_volume,
    slope_length,
    slope_width,
    x_offset=0.0,
    width_offset=0.0,
    z_offset=0.0,
    color="#3a8dde",
    alpha=0.9,
):
    if target_volume <= 0:
        return

    fill_state = compute_animation_voxel_fill(
        reservoir_geometry,
        animation_surface,
        target_volume,
        slope_length,
        slope_width,
    )
    filled = fill_state["filled"]
    fill_fraction = fill_state["fill_fraction"]
    if not np.any(filled):
        return

    voxel_x_corners = animation_surface["voxel_x_corners"] + float(x_offset)
    voxel_width_corners = animation_surface["voxel_width_corners"] + float(width_offset)
    voxel_z_corners = animation_surface["voxel_z_corners"] + float(z_offset)

    facecolors = np.empty(filled.shape + (4,), dtype=float)
    red_value, green_value, blue_value, _ = to_rgba(color, alpha)
    facecolors[..., 0] = red_value
    facecolors[..., 1] = green_value
    facecolors[..., 2] = blue_value
    facecolors[..., 3] = np.clip(fill_fraction * alpha, 0.0, alpha)
    axis.voxels(
        voxel_x_corners,
        voxel_width_corners,
        voxel_z_corners,
        filled,
        facecolors=facecolors,
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        linewidth=0.0,
    )


def plot_port_anchored_split_surface(
    axis,
    reservoir_geometry,
    animation_surface,
    total_volume,
    left_target_volume,
    right_target_volume,
    slope_length,
    slope_width,
    left_color="#ff006e",
    right_color="#00b4d8",
    alpha=0.9,
):
    if total_volume <= 0:
        return

    fill_state = compute_animation_voxel_fill(
        reservoir_geometry,
        animation_surface,
        total_volume,
        slope_length,
        slope_width,
    )
    filled = fill_state["filled"]
    fill_fraction = fill_state["fill_fraction"]
    if not np.any(filled):
        return

    voxel_x_corners = animation_surface["voxel_x_corners"]
    voxel_width_corners = animation_surface["voxel_width_corners"]
    voxel_z_corners = animation_surface["voxel_z_corners"]
    voxel_x_grid = animation_surface["voxel_x_grid"]
    voxel_width_grid = animation_surface["voxel_width_grid"]

    cell_dx = float(voxel_x_corners[1, 0, 0] - voxel_x_corners[0, 0, 0])
    cell_dy = float(voxel_width_corners[0, 1, 0] - voxel_width_corners[0, 0, 0])
    cell_dz = float(voxel_z_corners[0, 0, 1] - voxel_z_corners[0, 0, 0])
    cell_volume = cell_dx * cell_dy * cell_dz
    cell_liquid_volume = fill_fraction * cell_volume
    total_display_volume = float(np.sum(cell_liquid_volume))
    if total_display_volume <= 0.0:
        return

    local_total = float(max(left_target_volume, 0.0) + max(right_target_volume, 0.0))
    if local_total <= 1e-12:
        normalized_left_volume = 0.5 * total_display_volume
    else:
        normalized_left_volume = float(total_display_volume) * max(left_target_volume, 0.0) / local_total
    normalized_left_volume = float(np.clip(normalized_left_volume, 0.0, total_display_volume))

    voxel_valid_mask = (
        animation_surface["voxel_point_mask"]
        & np.isfinite(animation_surface["voxel_bottom_grid"])
        & np.isfinite(animation_surface["voxel_top_grid"])
    )
    partition_key = (
        float(reservoir_geometry["left_port"]["x"]),
        float(reservoir_geometry["left_port_width"]),
        float(reservoir_geometry["right_port"]["x"]),
        float(reservoir_geometry["right_port_width"]),
    )
    if animation_surface.get("_voxel_port_partition_key") != partition_key:
        (
            left_distance,
            right_distance,
            _left_region,
            _right_region,
        ) = get_port_connected_footprint_partition(
            reservoir_geometry,
            voxel_x_grid,
            voxel_width_grid,
            voxel_valid_mask,
        )
        animation_surface["_voxel_port_partition_key"] = partition_key
        animation_surface["_voxel_left_distance"] = left_distance
        animation_surface["_voxel_right_distance"] = right_distance
    port_priority = np.zeros_like(animation_surface["_voxel_left_distance"])
    np.subtract(
        animation_surface["_voxel_right_distance"],
        animation_surface["_voxel_left_distance"],
        out=port_priority,
        where=voxel_valid_mask,
    )
    left_priority = np.broadcast_to(
        port_priority[:, :, np.newaxis],
        filled.shape,
    )[filled]
    filled_cell_volumes = cell_liquid_volume[filled]
    if len(filled_cell_volumes) == 0:
        return

    order = np.argsort(-left_priority)
    ordered_volumes = filled_cell_volumes[order]
    cumulative_volumes = np.cumsum(ordered_volumes)
    left_alloc_sorted = np.zeros_like(ordered_volumes)
    fully_left_mask = cumulative_volumes <= normalized_left_volume
    left_alloc_sorted[fully_left_mask] = ordered_volumes[fully_left_mask]
    boundary_index = int(np.searchsorted(cumulative_volumes, normalized_left_volume, side="left"))
    if boundary_index < len(ordered_volumes):
        previous_volume = cumulative_volumes[boundary_index - 1] if boundary_index > 0 else 0.0
        boundary_fill = max(0.0, normalized_left_volume - previous_volume)
        left_alloc_sorted[boundary_index] = min(ordered_volumes[boundary_index], boundary_fill)

    left_alloc_flat = np.zeros_like(filled_cell_volumes)
    left_alloc_flat[order] = left_alloc_sorted
    left_fraction_flat = np.divide(
        left_alloc_flat,
        np.maximum(filled_cell_volumes, 1e-12),
    )
    right_fraction_flat = 1.0 - left_fraction_flat

    left_red, left_green, left_blue, _ = to_rgba(left_color, alpha)
    right_red, right_green, right_blue, _ = to_rgba(right_color, alpha)
    facecolors = np.zeros(filled.shape + (4,), dtype=float)
    facecolors[..., 0][filled] = (
        left_fraction_flat * left_red + right_fraction_flat * right_red
    )
    facecolors[..., 1][filled] = (
        left_fraction_flat * left_green + right_fraction_flat * right_green
    )
    facecolors[..., 2][filled] = (
        left_fraction_flat * left_blue + right_fraction_flat * right_blue
    )
    facecolors[..., 3][filled] = np.clip(fill_fraction[filled] * alpha, 0.0, alpha)

    axis.voxels(
        voxel_x_corners,
        voxel_width_corners,
        voxel_z_corners,
        filled,
        facecolors=facecolors,
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        linewidth=0.0,
    )


def draw_cylindrical_reservoir_wireframe(axis, centre_x, centre_y, radius, height):
    """Draw an upright cylindrical reservoir in the simple-tilting animation."""
    theta = np.linspace(0.0, 2.0 * math.pi, 96)
    circle_x = centre_x + radius * np.cos(theta)
    circle_y = centre_y + radius * np.sin(theta)
    axis.plot(circle_x, circle_y, np.zeros_like(theta), color="0.35", linewidth=0.9)
    axis.plot(circle_x, circle_y, np.full_like(theta, height), color="0.35", linewidth=0.9)
    for angle in np.linspace(0.0, 2.0 * math.pi, 10, endpoint=False):
        x_value = centre_x + radius * math.cos(angle)
        y_value = centre_y + radius * math.sin(angle)
        axis.plot([x_value, x_value], [y_value, y_value], [0.0, height], color="0.55", linewidth=0.55)


def plot_cylindrical_liquid_volume(
    axis,
    centre_x,
    centre_y,
    reservoir_geometry,
    target_volume,
    tilt_y,
    tilt_z,
    color="#3a8dde",
    alpha=0.86,
):
    """Render one tilted liquid volume inside a cylindrical reservoir."""
    if target_volume <= 0.0:
        return

    radius = float(reservoir_geometry["radius"])
    height = float(reservoir_geometry["height"])
    intercept = get_cylindrical_free_surface_intercept(
        target_volume,
        tilt_z,
        tilt_y,
        reservoir_geometry,
    )
    x_edges = np.linspace(-radius, radius, 25)
    y_edges = np.linspace(-radius, radius, 25)
    z_edges = np.linspace(0.0, height, 19)
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    local_x_grid, local_y_grid = np.meshgrid(x_centres, y_centres, indexing="ij")
    within_cylinder = local_x_grid**2 + local_y_grid**2 <= radius**2
    liquid_plane = (
        intercept
        - math.tan(tilt_z) * local_x_grid
        - math.tan(tilt_y) * local_y_grid
    )
    filled = within_cylinder[:, :, np.newaxis] & (
        z_centres[np.newaxis, np.newaxis, :] <= liquid_plane[:, :, np.newaxis]
    )
    if not np.any(filled):
        return

    voxel_x, voxel_y, voxel_z = np.meshgrid(
        x_edges + centre_x,
        y_edges + centre_y,
        z_edges,
        indexing="ij",
    )
    axis.voxels(
        voxel_x,
        voxel_y,
        voxel_z,
        filled,
        facecolors=to_rgba(color, alpha),
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        linewidth=0.0,
    )


def save_simple_tilting_cylinder_animation(
    reservoir_geometry,
    plot_data,
    output_path,
    fps=24,
    frame_count=None,
):
    """Save the two-cylinder animation without using the STL/box renderer."""
    radius = float(reservoir_geometry["radius"])
    height = float(reservoir_geometry["height"])
    channel_length = float(plot_data["selected_case"]["geometry1"]["length"])
    centre_offset = radius + 0.5 * channel_length
    cylinder_centres = (-centre_offset, centre_offset)
    time_values = get_realtime_animation_time_values(
        plot_data, fps, frame_count=frame_count
    )
    tilt_y_values = interpolate_series(plot_data["time_points"], plot_data["tilt_y"], time_values)
    tilt_z_values = interpolate_series(plot_data["time_points"], plot_data["tilt_z"], time_values)
    volume_1_values = interpolate_series(
        plot_data["time_points"], plot_data["reservoir_1_volume_ul"], time_values
    )
    volume_2_values = interpolate_series(
        plot_data["time_points"], plot_data["reservoir_2_volume_ul"], time_values
    )
    figure = plt.figure(figsize=(9.2, 6.8))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_box_aspect((2.0 * (centre_offset + radius), 2.0 * radius, height))

    def update(frame_index):
        axis.clear()
        tilt_y = math.radians(float(tilt_y_values[frame_index]))
        tilt_z = math.radians(float(tilt_z_values[frame_index]))
        for centre_x, volume in zip(cylinder_centres, (volume_1_values[frame_index], volume_2_values[frame_index])):
            draw_cylindrical_reservoir_wireframe(axis, centre_x, 0.0, radius, height)
            plot_cylindrical_liquid_volume(
                axis,
                centre_x,
                0.0,
                reservoir_geometry,
                float(volume),
                tilt_y,
                tilt_z,
            )

        axis.plot(
            [cylinder_centres[0] + radius, cylinder_centres[1] - radius],
            [0.0, 0.0],
            [0.0, 0.0],
            color="0.25",
            linewidth=2.0,
        )
        axis.scatter(cylinder_centres, [0.0, 0.0], [0.0, 0.0], color="#d62828", s=28, depthshade=False)
        axis.set_xlim(-centre_offset - radius, centre_offset + radius)
        axis.set_ylim(-radius, radius)
        axis.set_zlim(0.0, height)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_zlabel("z [mm]")
        axis.view_init(elev=24, azim=-58)
        axis.set_title(
            "Simple Tilting: Cylindrical Reservoirs  "
            f"t={time_values[frame_index]:.1f} s\n"
            f"V1={volume_1_values[frame_index]:.1f} uL, "
            f"V2={volume_2_values[frame_index]:.1f} uL"
        )

    animation = FuncAnimation(figure, update, frames=len(time_values), interval=1000 / fps)
    output_path = Path(output_path).resolve()
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=110)
    finally:
        plt.close(figure)
    return output_path


def save_nested_four_reservoir_loop_animation(
    inner_reservoir_geometry,
    outer_reservoir_geometry,
    plot_data,
    output_path,
    fps=24,
    frame_count=None,
):
    """Animate two independent mirrored two-reservoir loops about the origin."""
    loop_specs = []
    for geometry, channel_geometry, volume_indices, color, name in (
        (
            inner_reservoir_geometry,
            plot_data["selected_case"]["geometry1"],
            (1, 2),
            "#457b9d",
            "Inner loop",
        ),
        (
            outer_reservoir_geometry,
            plot_data["selected_case"]["geometry3"],
            (3, 4),
            "#e76f51",
            "Outer loop",
        ),
    ):
        original_geometry = build_animation_reservoir_geometry(geometry)
        mirrored_geometry = mirror_animation_reservoir_across_width(original_geometry)
        original_surface = build_animation_surface_context(original_geometry)
        mirrored_surface = build_animation_surface_context(mirrored_geometry)
        port_offsets = []
        for port_name in ("left", "right"):
            mirrored_port = get_animation_port_coordinates(mirrored_geometry, port_name)
            original_port = get_animation_port_coordinates(original_geometry, port_name)
            port_offsets.append(
                float(channel_geometry["length"])
                + mirrored_port[1]
                - original_port[1]
            )
        relative_offset = float(np.mean(port_offsets))
        # Center the two facing reservoir bodies around y=0 while preserving
        # the selected channel length between corresponding ports.
        mirrored_offset = -0.5 * relative_offset
        original_offset = 0.5 * relative_offset
        loop_specs.append(
            {
                "name": name,
                "color": color,
                "channel_length": float(channel_geometry["length"]),
                "mirrored_geometry": mirrored_geometry,
                "original_geometry": original_geometry,
                "mirrored_surface": mirrored_surface,
                "original_surface": original_surface,
                "x_offset": -0.5 * float(original_surface["plot_length"]),
                "mirrored_offset": mirrored_offset,
                "original_offset": original_offset,
                "volume_indices": volume_indices,
            }
        )

    time_values = get_realtime_animation_time_values(
        plot_data, fps, frame_count=frame_count
    )
    tilt_y_values = interpolate_series(
        plot_data["time_points"], plot_data["tilt_y"], time_values
    )
    tilt_z_values = interpolate_series(
        plot_data["time_points"], plot_data["tilt_z"], time_values
    )
    volume_values = {
        index: interpolate_series(
            plot_data["time_points"],
            plot_data[f"reservoir_{index}_volume_ul"],
            time_values,
        )
        for index in range(1, 5)
    }

    all_surfaces = [
        spec[key]
        for spec in loop_specs
        for key in ("mirrored_surface", "original_surface")
    ]
    min_x = min(float(spec["x_offset"]) for spec in loop_specs)
    max_x = max(
        float(spec["x_offset"]) + float(spec["original_surface"]["plot_length"])
        for spec in loop_specs
    )
    min_y = min(
        float(spec["mirrored_offset"])
        for spec in loop_specs
    )
    max_y = max(
        float(spec["original_offset"]) + float(spec["original_surface"]["footprint_width"])
        for spec in loop_specs
    )
    max_height = max(float(surface["plot_height"]) for surface in all_surfaces)
    padding = max(1.0, 0.06 * max(max_x - min_x, max_y - min_y, max_height))

    figure = plt.figure(figsize=(10.4, 7.4))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_box_aspect((max_x - min_x + 2 * padding, max_y - min_y + 2 * padding, max_height))

    def update(frame_index):
        axis.clear()
        tilt_y_degrees = float(tilt_y_values[frame_index])
        tilt_z_degrees = float(tilt_z_values[frame_index])
        slope_length, profile_slope_width, _ = get_stl_gravity_plane(
            math.radians(tilt_z_degrees), math.radians(tilt_y_degrees)
        )
        for loop_index, spec in enumerate(loop_specs):
            for geometry, surface, width_offset, volume_index, width_sign in (
                (
                    spec["mirrored_geometry"],
                    spec["mirrored_surface"],
                    spec["mirrored_offset"],
                    spec["volume_indices"][1],
                    -1.0,
                ),
                (
                    spec["original_geometry"],
                    spec["original_surface"],
                    spec["original_offset"],
                    spec["volume_indices"][0],
                    1.0,
                ),
            ):
                draw_reservoir_wireframe(
                    axis,
                    geometry,
                    x_offset=spec["x_offset"],
                    width_offset=width_offset,
                )
                plot_animation_water_surface(
                    axis,
                    geometry,
                    surface,
                    float(volume_values[volume_index][frame_index]),
                    slope_length,
                    width_sign * profile_slope_width
                    if geometry["type"] == "profile"
                    else 0.0,
                    x_offset=spec["x_offset"],
                    width_offset=width_offset,
                )

            for port_index, port_name in enumerate(("left", "right")):
                near_port = list(get_animation_port_coordinates(
                    spec["mirrored_geometry"], port_name, spec["mirrored_offset"]
                ))
                far_port = list(get_animation_port_coordinates(
                    spec["original_geometry"], port_name, spec["original_offset"]
                ))
                near_port[0] += float(spec["x_offset"])
                far_port[0] += float(spec["x_offset"])
                axis.plot(
                    [near_port[0], far_port[0]],
                    [near_port[1], far_port[1]],
                    [near_port[2], far_port[2]],
                    color=spec["color"],
                    linewidth=2.4,
                    label=f"{spec['name']} channels" if port_index == 0 else None,
                )
                axis.scatter(
                    [near_port[0], far_port[0]],
                    [near_port[1], far_port[1]],
                    [near_port[2], far_port[2]],
                    color=spec["color"],
                    s=24,
                    depthshade=False,
                )

        axis.set_xlim(min_x - padding, max_x + padding)
        axis.set_ylim(min_y - padding, max_y + padding)
        axis.set_zlim(0.0, max_height)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_zlabel("z [mm]")
        axis.view_init(elev=25, azim=-43)
        axis.legend(loc="upper left")
        axis.set_title(
            "Nested Four-Reservoir Recirculating Loops  "
            f"t={time_values[frame_index]:.1f} s\n"
            + ", ".join(
                f"V{index}={volume_values[index][frame_index]:.1f} uL"
                for index in range(1, 5)
            )
        )

    animation = FuncAnimation(figure, update, frames=len(time_values), interval=1000 / fps)
    output_path = Path(output_path).resolve()
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=110)
    finally:
        plt.close(figure)
    return output_path


def save_four_reservoir_ring_animation(
    reservoir_geometry,
    plot_data,
    output_path,
    fps=24,
    frame_count=None,
):
    """Animate four cylindrical reservoirs and their equal-length ring links."""
    radius = float(reservoir_geometry["radius"])
    height = float(reservoir_geometry["height"])
    channel_length = float(plot_data["selected_case"]["geometry1"]["length"])
    centre_offset = radius + 0.5 * channel_length
    cylinder_centres = (
        (-centre_offset, -centre_offset),
        (centre_offset, -centre_offset),
        (centre_offset, centre_offset),
        (-centre_offset, centre_offset),
    )
    time_values = get_realtime_animation_time_values(
        plot_data, fps, frame_count=frame_count
    )
    tilt_y_values = interpolate_series(
        plot_data["time_points"], plot_data["tilt_y"], time_values
    )
    tilt_z_values = interpolate_series(
        plot_data["time_points"], plot_data["tilt_z"], time_values
    )
    volume_values = tuple(
        interpolate_series(
            plot_data["time_points"],
            plot_data[f"reservoir_{reservoir_index}_volume_ul"],
            time_values,
        )
        for reservoir_index in range(1, 5)
    )
    figure = plt.figure(figsize=(9.2, 7.6))
    axis = figure.add_subplot(111, projection="3d")
    plot_extent = centre_offset + radius
    axis.set_box_aspect((2.0 * plot_extent, 2.0 * plot_extent, height))

    def update(frame_index):
        axis.clear()
        tilt_y = math.radians(float(tilt_y_values[frame_index]))
        tilt_z = math.radians(float(tilt_z_values[frame_index]))
        frame_volumes = [float(values[frame_index]) for values in volume_values]
        for (centre_x, centre_y), volume in zip(cylinder_centres, frame_volumes):
            draw_cylindrical_reservoir_wireframe(
                axis, centre_x, centre_y, radius, height
            )
            plot_cylindrical_liquid_volume(
                axis,
                centre_x,
                centre_y,
                reservoir_geometry,
                volume,
                tilt_y,
                tilt_z,
            )

        ring_segments = (
            ((-0.5 * channel_length, -centre_offset), (0.5 * channel_length, -centre_offset)),
            ((centre_offset, -0.5 * channel_length), (centre_offset, 0.5 * channel_length)),
            ((0.5 * channel_length, centre_offset), (-0.5 * channel_length, centre_offset)),
            ((-centre_offset, 0.5 * channel_length), (-centre_offset, -0.5 * channel_length)),
        )
        for channel_index, (start, end) in enumerate(ring_segments, start=1):
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [0.0, 0.0],
                color="0.25",
                linewidth=2.0,
                label="Connecting channels" if channel_index == 1 else None,
            )
            axis.scatter(
                [start[0], end[0]],
                [start[1], end[1]],
                [0.0, 0.0],
                color="#d62828",
                s=24,
                depthshade=False,
            )

        axis.set_xlim(-plot_extent, plot_extent)
        axis.set_ylim(-plot_extent, plot_extent)
        axis.set_zlim(0.0, height)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_zlabel("z [mm]")
        axis.view_init(elev=27, azim=-48)
        axis.set_title(
            "Four-Reservoir Ring: Cylindrical Reservoirs  "
            f"t={time_values[frame_index]:.1f} s\n"
            + ", ".join(
                f"V{index + 1}={volume:.1f} uL"
                for index, volume in enumerate(frame_volumes)
            )
        )

    animation = FuncAnimation(figure, update, frames=len(time_values), interval=1000 / fps)
    output_path = Path(output_path).resolve()
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=110)
    finally:
        plt.close(figure)
    return output_path


def get_animation_port_coordinates(reservoir_geometry, port_name, width_offset=0.0):
    """Return one animation port in the shared x, y, z coordinate system."""
    port = reservoir_geometry[f"{port_name}_port"]
    return (
        float(port["x"]),
        float(reservoir_geometry[f"{port_name}_port_width"]) + float(width_offset),
        float(port["y"]),
    )


def build_two_reservoir_mirrored_layout(
    reservoir_geometry,
    channel_1_geometry,
    channel_2_geometry,
):
    """Build the common plan-view layout used by the two-reservoir model."""
    original_geometry = build_animation_reservoir_geometry(reservoir_geometry)
    mirrored_geometry = mirror_reservoir_geometry_across_width(original_geometry)
    channel_lengths = {
        "right": float(channel_1_geometry["length"]),
        "left": float(channel_2_geometry["length"]),
    }
    port_offsets = []
    for port_name in ("left", "right"):
        mirrored_port = get_animation_port_coordinates(mirrored_geometry, port_name)
        original_port = get_animation_port_coordinates(original_geometry, port_name)
        port_offsets.append(
            channel_lengths[port_name] + mirrored_port[1] - original_port[1]
        )
    separation = 0.5 * float(np.mean(port_offsets))
    x_offset = -0.5 * get_animation_plot_extents(original_geometry)[0]

    def placed_port(geometry, port_name, position):
        port = np.asarray(get_animation_port_coordinates(geometry, port_name), dtype=float)
        return port + np.asarray(position, dtype=float)

    original_position = (x_offset, separation, 0.0)
    mirrored_position = (x_offset, -separation, 0.0)

    def placed_shell_points(geometry, position):
        shell_points = np.asarray(geometry["shell_triangles"], dtype=float).reshape(-1, 3)
        return shell_points + np.asarray(position, dtype=float)

    # Reservoir meshes are bottom/edge referenced after STL import. Rebase
    # the completed pair so the common geometry centre is the rotation origin.
    layout_points = np.vstack(
        (
            placed_shell_points(original_geometry, original_position),
            placed_shell_points(mirrored_geometry, mirrored_position),
        )
    )
    layout_centre = 0.5 * (
        np.min(layout_points, axis=0) + np.max(layout_points, axis=0)
    )
    original_position = tuple(np.asarray(original_position) - layout_centre)
    mirrored_position = tuple(np.asarray(mirrored_position) - layout_centre)

    return {
        "original_geometry": original_geometry,
        "mirrored_geometry": mirrored_geometry,
        "original_position": original_position,
        "mirrored_position": mirrored_position,
        "layout_centre_before_rebase_mm": tuple(float(value) for value in layout_centre),
        "channels": {
            "channel_1": (
                placed_port(original_geometry, "right", original_position),
                placed_port(mirrored_geometry, "right", mirrored_position),
            ),
            "channel_2": (
                placed_port(original_geometry, "left", original_position),
                placed_port(mirrored_geometry, "left", mirrored_position),
            ),
        },
    }


def get_two_reservoir_channel_port_positions(
    reservoir_geometry,
    channel_1_geometry,
    channel_2_geometry,
):
    """Return the placed endpoints of both channels in a mirrored reservoir pair."""
    return build_two_reservoir_mirrored_layout(
        reservoir_geometry,
        channel_1_geometry,
        channel_2_geometry,
    )["channels"]


def show_two_reservoir_layout_debug(
    reservoir_geometry,
    channel_1_geometry,
    channel_2_geometry,
):
    """Show a scaled plan view of the geometry used for port driving heads."""
    layout = build_two_reservoir_mirrored_layout(
        reservoir_geometry,
        channel_1_geometry,
        channel_2_geometry,
    )
    figure, axis = plt.subplots(figsize=(8.2, 6.5))

    def draw_reservoir(geometry, position, color, label):
        footprint = np.asarray(geometry["top_view_points"], dtype=float)
        footprint = footprint + np.asarray(position[:2], dtype=float)
        closed_footprint = np.vstack((footprint, footprint[0]))
        axis.fill(
            closed_footprint[:, 0],
            closed_footprint[:, 1],
            color=color,
            alpha=0.24,
            label=label,
        )
        axis.plot(closed_footprint[:, 0], closed_footprint[:, 1], color=color, linewidth=1.6)
        for port_name in ("left", "right"):
            port = np.asarray(
                get_animation_port_coordinates(geometry, port_name), dtype=float
            ) + np.asarray(position, dtype=float)
            axis.scatter(port[0], port[1], color=color, edgecolor="black", s=48, zorder=4)
            axis.annotate(
                f"{label} {port_name}",
                (port[0], port[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    draw_reservoir(
        layout["original_geometry"],
        layout["original_position"],
        "#457b9d",
        "Reservoir 1",
    )
    draw_reservoir(
        layout["mirrored_geometry"],
        layout["mirrored_position"],
        "#e76f51",
        "Reservoir 2 (mirrored)",
    )

    def channel_width(channel_geometry):
        if channel_geometry.get("shape") == "circular":
            return 2.0 * float(channel_geometry["radius"])
        return float(channel_geometry["width"])

    for channel_name, channel_geometry, color in (
        ("channel_1", channel_1_geometry, "#264653"),
        ("channel_2", channel_2_geometry, "#e9c46a"),
    ):
        source, destination = layout["channels"][channel_name]
        start = source[:2]
        end = destination[:2]
        direction = end - start
        distance = float(np.linalg.norm(direction))
        if distance <= 1e-9:
            continue
        normal = np.array((-direction[1], direction[0]), dtype=float) / distance
        half_width = 0.5 * channel_width(channel_geometry)
        channel_polygon = np.asarray(
            (
                start + normal * half_width,
                end + normal * half_width,
                end - normal * half_width,
                start - normal * half_width,
            )
        )
        axis.fill(channel_polygon[:, 0], channel_polygon[:, 1], color=color, alpha=0.36)
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=1.2,
            label=f"{channel_name.replace('_', ' ').title()} ({channel_geometry['length']:.2f} mm)",
        )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    add_geometry_dimension_panel(
        figure,
        (
            ("Reservoir", reservoir_geometry),
            ("Channel 1", channel_1_geometry),
            ("Channel 2", channel_2_geometry),
        ),
    )
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.88))
    show_geometry_layout_figure(figure)
    return figure


def show_geometry_layout_figure(figure):
    """Present a retained Matplotlib layout figure from the Qt application."""
    if "agg" in matplotlib.get_backend().lower():
        return
    figure.canvas.draw_idle()
    manager = figure.canvas.manager
    window = getattr(manager, "window", None)
    if window is not None:
        window.show()
        window.raise_()
        window.activateWindow()
    plt.show(block=False)


def format_reservoir_dimensions(label, reservoir_geometry):
    """Return a compact, consistently unit-labelled reservoir description."""
    if reservoir_geometry.get("type") == "cylindrical":
        diameter = 2.0 * float(reservoir_geometry["radius"])
        return (
            f"{label}: cylindrical, diameter {diameter:.2f} mm, "
            f"height {float(reservoir_geometry['height']):.2f} mm"
        )
    return (
        f"{label}: length {float(reservoir_geometry['length']):.2f} mm, "
        f"width {float(reservoir_geometry['width']):.2f} mm, "
        f"height {float(reservoir_geometry['height']):.2f} mm"
    )


def format_channel_dimensions(label, channel_geometry):
    """Return a compact, consistently unit-labelled channel description."""
    length = float(channel_geometry["length"])
    if channel_geometry.get("shape", "rectangular") == "circular":
        return (
            f"{label}: circular, length {length:.2f} mm, "
            f"radius {float(channel_geometry['radius']):.2f} mm"
        )
    return (
        f"{label}: rectangular, length {length:.2f} mm, "
        f"width {float(channel_geometry['width']):.2f} mm, "
        f"height {float(channel_geometry['height']):.2f} mm"
    )


def add_geometry_dimension_panel(figure, dimension_items):
    """Place model dimensions below a schematic without obscuring its layout."""
    lines = ["Dimensions"]
    for label, geometry in dimension_items:
        if geometry is None:
            continue
        if "shape" in geometry:
            lines.append(format_channel_dimensions(label, geometry))
        else:
            lines.append(format_reservoir_dimensions(label, geometry))
    figure.text(
        0.02,
        0.015,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.95},
    )


def show_model_geometry_layout(
    mode,
    reservoir_geometry,
    primary_geometry,
    secondary_geometry=None,
    outer_reservoir_geometry=None,
    outer_channel_geometry=None,
):
    """Show the scaled plan-view geometry corresponding to a solved model."""
    if mode in ("symmetric", "assymmetric"):
        return show_two_reservoir_layout_debug(
            reservoir_geometry,
            primary_geometry,
            secondary_geometry or primary_geometry,
        )

    figure, axis = plt.subplots(figsize=(8.6, 7.0))

    def channel_width(channel_geometry):
        return (
            2.0 * float(channel_geometry["radius"])
            if channel_geometry.get("shape") == "circular"
            else float(channel_geometry["width"])
        )

    def draw_channel(start, end, channel_geometry, color, label):
        start = np.asarray(start[:2], dtype=float)
        end = np.asarray(end[:2], dtype=float)
        direction = end - start
        distance = float(np.linalg.norm(direction))
        if distance <= 1e-9:
            return
        normal = np.array((-direction[1], direction[0]), dtype=float) / distance
        half_width = 0.5 * channel_width(channel_geometry)
        polygon = np.asarray(
            (start + normal * half_width, end + normal * half_width,
             end - normal * half_width, start - normal * half_width)
        )
        axis.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.34)
        axis.plot([start[0], end[0]], [start[1], end[1]], color=color, label=label)

    def draw_profile_reservoir(geometry, position, color, label):
        visual_geometry = build_animation_reservoir_geometry(geometry)
        footprint = np.asarray(visual_geometry["top_view_points"], dtype=float)
        footprint = footprint + np.asarray(position[:2], dtype=float)
        closed = np.vstack((footprint, footprint[0]))
        axis.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.22, label=label)
        axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.5)
        ports = {}
        for port_name in ("left", "right"):
            port = np.asarray(get_animation_port_coordinates(visual_geometry, port_name))
            port = port + np.asarray(position, dtype=float)
            ports[port_name] = port
            axis.scatter(port[0], port[1], color=color, edgecolor="black", s=42, zorder=4)
        return ports

    dimension_items = []
    if mode == "nested_four_reservoir_loops":
        dimension_items = (
            ("Inner reservoir", reservoir_geometry),
            ("Inner channels", primary_geometry),
            ("Outer reservoir", outer_reservoir_geometry),
            ("Outer channels", outer_channel_geometry),
        )
        for prefix, geometry, channel_geometry, colors in (
            ("Inner", reservoir_geometry, primary_geometry, ("#457b9d", "#264653")),
            ("Outer", outer_reservoir_geometry, outer_channel_geometry, ("#e76f51", "#9d4edd")),
        ):
            layout = build_two_reservoir_mirrored_layout(
                geometry, channel_geometry, channel_geometry
            )
            first_ports = draw_profile_reservoir(
                layout["original_geometry"], layout["original_position"], colors[0], f"{prefix} reservoir 1"
            )
            second_ports = draw_profile_reservoir(
                layout["mirrored_geometry"], layout["mirrored_position"], colors[0], f"{prefix} reservoir 2"
            )
            draw_channel(first_ports["right"], second_ports["right"], channel_geometry, colors[1], f"{prefix} channel 1")
            draw_channel(first_ports["left"], second_ports["left"], channel_geometry, colors[1], f"{prefix} channel 2")
    elif mode in ("single_reservoir", "single_reservoir_assymmetric"):
        dimension_items = (
            ("Reservoir", reservoir_geometry),
            ("Recirculation channel", primary_geometry),
        )
        visual_geometry = build_animation_reservoir_geometry(reservoir_geometry)
        points = np.asarray(visual_geometry["top_view_points"], dtype=float)
        centre = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
        ports = draw_profile_reservoir(
            visual_geometry, (-centre[0], -centre[1], 0.0), "#457b9d", "Reservoir"
        )
        draw_channel(ports["left"], ports["right"], primary_geometry, "#264653", "Recirculation channel")
    else:
        dimension_items = (
            ("Cylindrical reservoirs", reservoir_geometry),
            ("Channels", primary_geometry),
        )
        radius = float(reservoir_geometry["radius"])
        channel_length = float(primary_geometry["length"])
        offset = radius + 0.5 * channel_length
        if mode == "simple_tilting":
            centres = [np.array((0.0, -offset)), np.array((0.0, offset))]
            links = ((0, 1),)
        else:
            centres = [
                np.array((-offset, -offset)), np.array((offset, -offset)),
                np.array((offset, offset)), np.array((-offset, offset)),
            ]
            links = ((0, 1), (1, 2), (2, 3), (3, 0))
        for index, centre in enumerate(centres, start=1):
            axis.add_patch(plt.Circle(centre, radius, color="#457b9d", alpha=0.25))
            axis.plot(centre[0], centre[1], "o", color="#457b9d")
            axis.annotate(f"Reservoir {index}", centre, xytext=(4, 4), textcoords="offset points", fontsize=8)
        ring_colors = ("#264653", "#e76f51", "#2a9d8f", "#9d4edd")
        for index, (source_index, destination_index) in enumerate(links, start=1):
            source = centres[source_index] + (centres[destination_index] - centres[source_index]) / np.linalg.norm(centres[destination_index] - centres[source_index]) * radius
            destination = centres[destination_index] + (centres[source_index] - centres[destination_index]) / np.linalg.norm(centres[source_index] - centres[destination_index]) * radius
            draw_channel(
                source, destination, primary_geometry, ring_colors[index - 1], f"Channel {index}"
            )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    add_geometry_dimension_panel(figure, dimension_items)
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.88))
    show_geometry_layout_figure(figure)
    return figure


def save_two_reservoir_animation(
    reservoir_geometry,
    plot_data,
    output_path,
    fps=24,
    frame_count=None,
):
    """Animate both reservoirs and their two straight inlet/outlet links."""
    animation_geometry = build_animation_reservoir_geometry(reservoir_geometry)
    animation_surface = build_animation_surface_context(animation_geometry)
    mirrored_animation_geometry = mirror_animation_reservoir_across_width(
        animation_geometry
    )
    mirrored_animation_surface = build_animation_surface_context(
        mirrored_animation_geometry
    )
    plot_length = animation_surface["plot_length"]
    footprint_width = animation_surface["footprint_width"]
    plot_height = animation_surface["plot_height"]
    channel_length = float(plot_data["selected_case"]["geometry1"]["length"])
    # The reflected and original port positions are not generally at the same
    # width coordinate.  Translate the second reservoir so their mean
    # port-to-port separation is the selected channel length.  It is exact for
    # the normal case where the inlet and outlet share the same top-view width.
    port_offsets = []
    for port_name in ("left", "right"):
        near_port = get_animation_port_coordinates(
            mirrored_animation_geometry,
            port_name,
        )
        far_port = get_animation_port_coordinates(animation_geometry, port_name)
        port_offsets.append(channel_length + near_port[1] - far_port[1])
    second_reservoir_relative_offset = float(np.mean(port_offsets))
    first_reservoir_width_offset = max(0.0, -second_reservoir_relative_offset)
    second_reservoir_width_offset = (
        first_reservoir_width_offset + second_reservoir_relative_offset
    )
    total_width = max(
        first_reservoir_width_offset + footprint_width,
        second_reservoir_width_offset + footprint_width,
    )
    time_values = get_realtime_animation_time_values(
        plot_data, fps, frame_count=frame_count
    )
    tilt_y_values = interpolate_series(plot_data["time_points"], plot_data["tilt_y"], time_values)
    tilt_z_values = interpolate_series(plot_data["time_points"], plot_data["tilt_z"], time_values)
    volume_1_values = interpolate_series(
        plot_data["time_points"], plot_data["reservoir_1_volume_ul"], time_values
    )
    volume_2_values = interpolate_series(
        plot_data["time_points"], plot_data["reservoir_2_volume_ul"], time_values
    )
    figure = plt.figure(figsize=(10.2, 6.8))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_box_aspect((plot_length, total_width, plot_height))

    def update(frame_index):
        axis.clear()
        tilt_y_degrees = float(tilt_y_values[frame_index])
        tilt_z_degrees = float(tilt_z_values[frame_index])
        slope_length, profile_slope_width, _ = get_stl_gravity_plane(
            math.radians(tilt_z_degrees),
            math.radians(tilt_y_degrees),
        )
        slope_width = profile_slope_width if reservoir_geometry["type"] == "profile" else 0.0
        reservoir_volumes = (volume_1_values[frame_index], volume_2_values[frame_index])
        for (
            geometry,
            surface,
            width_offset,
            reservoir_volume,
            local_slope_width,
        ) in (
            (
                # Place the reflected reservoir on the near side and the
                # original one on the far side so both bodies face outward.
                mirrored_animation_geometry,
                mirrored_animation_surface,
                first_reservoir_width_offset,
                reservoir_volumes[1],
                -slope_width,
            ),
            (
                animation_geometry,
                animation_surface,
                second_reservoir_width_offset,
                reservoir_volumes[0],
                slope_width,
            ),
        ):
            draw_reservoir_wireframe(
                axis,
                geometry,
                width_offset=width_offset,
            )
            plot_animation_water_surface(
                axis,
                geometry,
                surface,
                float(reservoir_volume),
                slope_length,
                local_slope_width,
                width_offset=width_offset,
            )

        for port_name, color, label in (
            ("left", "#d62828", "Inlet connection"),
            ("right", "#2a9d8f", "Outlet connection"),
        ):
            first_port = get_animation_port_coordinates(
                mirrored_animation_geometry,
                port_name,
                first_reservoir_width_offset,
            )
            second_port = get_animation_port_coordinates(
                animation_geometry,
                port_name,
                second_reservoir_width_offset,
            )
            axis.plot(
                [first_port[0], second_port[0]],
                [first_port[1], second_port[1]],
                [first_port[2], second_port[2]],
                color=color,
                linewidth=2.2,
                label=label,
            )
            axis.scatter(
                [first_port[0], second_port[0]],
                [first_port[1], second_port[1]],
                [first_port[2], second_port[2]],
                color=color,
                s=30,
                depthshade=False,
            )

        axis.set_xlim(0.0, plot_length)
        axis.set_ylim(0.0, total_width)
        axis.set_zlim(0.0, plot_height)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_zlabel("z [mm]")
        axis.view_init(elev=25, azim=-43)
        axis.legend(loc="upper left")
        axis.set_title(
            "Two-Reservoir Liquid Levels  "
            f"t={time_values[frame_index]:.1f} s\n"
            f"V1={reservoir_volumes[0]:.1f} uL, V2={reservoir_volumes[1]:.1f} uL; "
            f"tilt_y={tilt_y_degrees:.1f} deg, tilt_z={tilt_z_degrees:.1f} deg"
        )

    animation = FuncAnimation(figure, update, frames=len(time_values), interval=1000 / fps)
    output_path = Path(output_path).resolve()
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=110)
    finally:
        plt.close(figure)
    return output_path


def save_reservoir_animation(
    reservoir_geometry,
    plot_data,
    output_path,
    fps=24,
    frame_count=None,
):
    if plot_data["selected_case"]["mode"] == "simple_tilting":
        return save_simple_tilting_cylinder_animation(
            reservoir_geometry,
            plot_data,
            output_path,
            fps=fps,
            frame_count=frame_count,
        )

    if plot_data["selected_case"]["mode"] == "four_reservoir_tilting":
        return save_four_reservoir_ring_animation(
            reservoir_geometry,
            plot_data,
            output_path,
            fps=fps,
            frame_count=frame_count,
        )

    if plot_data["selected_case"]["mode"] == "nested_four_reservoir_loops":
        return save_nested_four_reservoir_loop_animation(
            plot_data["selected_case"]["inner_reservoir_geometry"],
            plot_data["selected_case"]["outer_reservoir_geometry"],
            plot_data,
            output_path,
            fps=fps,
            frame_count=frame_count,
        )

    if plot_data["selected_case"]["mode"] in ("symmetric", "assymmetric"):
        return save_two_reservoir_animation(
            reservoir_geometry,
            plot_data,
            output_path,
            fps=fps,
            frame_count=frame_count,
        )

    animation_geometry = build_animation_reservoir_geometry(reservoir_geometry)
    animation_surface = build_animation_surface_context(animation_geometry)
    plot_length = animation_surface["plot_length"]
    footprint_width = animation_surface["footprint_width"]
    plot_height = animation_surface["plot_height"]
    mode = plot_data["selected_case"]["mode"]
    time_values = get_realtime_animation_time_values(
        plot_data, fps, frame_count=frame_count
    )
    animation_tilt_y = interpolate_series(plot_data["time_points"], plot_data["tilt_y"], time_values)
    animation_tilt_z = interpolate_series(plot_data["time_points"], plot_data["tilt_z"], time_values)
    animation_reservoir_1_volume = interpolate_series(
        plot_data["time_points"],
        plot_data["reservoir_1_volume_ul"],
        time_values,
    )
    animation_left_local_volume = interpolate_series(
        plot_data["time_points"],
        np.nan_to_num(plot_data["left_local_volume_ul"], nan=0.0),
        time_values,
    )
    animation_right_local_volume = interpolate_series(
        plot_data["time_points"],
        np.nan_to_num(plot_data["right_local_volume_ul"], nan=0.0),
        time_values,
    )
    render_top_contact_port_pools = (
        mode in ("single_reservoir", "single_reservoir_assymmetric")
        and bool(animation_geometry.get("top_contact_wetting", False))
    )
    figure = plt.figure(figsize=(8.8, 6.8))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_box_aspect(
        (
            plot_length,
            footprint_width,
            plot_height,
        )
    )

    def update(frame_index):
        axis.clear()
        time_value = float(time_values[frame_index])
        tilt_y_degrees = float(animation_tilt_y[frame_index])
        tilt_z_degrees = float(animation_tilt_z[frame_index])
        tilt_y = math.radians(tilt_y_degrees)
        tilt_z = math.radians(tilt_z_degrees)
        slope_length, profile_slope_width, _ = get_stl_gravity_plane(
            tilt_z,
            tilt_y,
        )
        slope_width = (
            profile_slope_width
            if mode in ("single_reservoir", "single_reservoir_assymmetric")
            else 0.0
        )

        draw_reservoir_wireframe(axis, animation_geometry)
        rendered_port_pools = False
        if render_top_contact_port_pools and (
            animation_left_local_volume[frame_index] > 1e-9
            or animation_right_local_volume[frame_index] > 1e-9
        ):
            rendered_port_pools = plot_stl_port_pools(
                axis,
                animation_geometry,
                animation_surface,
                float(animation_left_local_volume[frame_index]),
                float(animation_right_local_volume[frame_index]),
                slope_length,
                slope_width,
                left_color="#3a8dde",
                right_color="#3a8dde",
            )
        if not rendered_port_pools:
            plot_animation_water_surface(
                axis,
                animation_geometry,
                animation_surface,
                float(animation_reservoir_1_volume[frame_index]),
                slope_length,
                slope_width,
            )
        axis.scatter(
            [
                animation_geometry["left_port"]["x"],
                animation_geometry["right_port"]["x"],
            ],
            [
                animation_geometry["left_port_width"],
                animation_geometry["right_port_width"],
            ],
            [
                animation_geometry["left_port"]["y"],
                animation_geometry["right_port"]["y"],
            ],
            color=["#d62828", "#2a9d8f"],
            s=30,
            depthshade=False,
        )

        axis.set_xlim(0.0, plot_length)
        axis.set_ylim(0.0, footprint_width)
        axis.set_zlim(0.0, plot_height)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_zlabel("z [mm]")
        axis.view_init(elev=24, azim=-58)
        axis.set_title(
            f"Reservoir Liquid Surface  t={time_value:.1f} s\n"
            f"tilt_y={tilt_y_degrees:.1f} deg, tilt_z={tilt_z_degrees:.1f} deg"
        )

    animation = FuncAnimation(figure, update, frames=len(time_values), interval=1000 / fps)
    output_path = Path(output_path).resolve()
    try:
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=110)
    finally:
        plt.close(figure)
    return output_path


def launch_gui():
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    app = QApplication.instance() or QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("Microfluidic Flow Plotter")

    layout = QVBoxLayout()
    window.setLayout(layout)

    advanced_mode_checkbox = QCheckBox("Advanced mode")
    advanced_mode_checkbox.setToolTip(
        "Show solver, capillary, wet/dry gating, and Fourier-series controls."
    )
    layout.addWidget(advanced_mode_checkbox)

    fourier_info_label = QLabel(
        "Generate a plot, then select a flow curve and number of terms."
    )
    fourier_info_label.setWordWrap(True)

    fourier_group = QGroupBox("Fourier series settings")
    fourier_group_layout = QVBoxLayout()
    fourier_group.setLayout(fourier_group_layout)
    fourier_fit_layout = QHBoxLayout()
    fourier_fit_layout.addWidget(QLabel("Fourier curve"))
    fourier_curve_input = QComboBox()
    fourier_curve_input.addItem("Flow channel 1 / Q", "q_1")
    fourier_curve_input.addItem("Flow channel 2 / Q2", "q_2")
    fourier_curve_input.addItem("Flow channel 3 / Q3", "q_3")
    fourier_curve_input.addItem("Flow channel 4 / Q4", "q_4")
    fourier_fit_layout.addWidget(fourier_curve_input)
    fourier_fit_layout.addWidget(QLabel("Terms [1-50]"))
    fourier_terms_input = QLineEdit("8")
    fourier_terms_input.setMaximumWidth(55)
    fourier_fit_layout.addWidget(fourier_terms_input)
    fourier_group_layout.addLayout(fourier_fit_layout)

    content_layout = QHBoxLayout()
    left_column_layout = QVBoxLayout()
    middle_column_layout = QVBoxLayout()
    right_column_layout = QVBoxLayout()
    content_layout.addLayout(left_column_layout, 1)
    content_layout.addLayout(middle_column_layout, 1)
    content_layout.addLayout(right_column_layout, 1)
    layout.addLayout(content_layout)

    model_group = QGroupBox("Model setting")
    model_group_layout = QVBoxLayout()
    model_group.setLayout(model_group_layout)

    two_reservoir_radio = QRadioButton("2-reservoir")
    two_reservoir_radio.setChecked(True)
    model_group_layout.addWidget(two_reservoir_radio)

    single_reservoir_radio = QRadioButton("1-reservoir")
    model_group_layout.addWidget(single_reservoir_radio)

    simple_tilting_radio = QRadioButton("Simple tilting")
    model_group_layout.addWidget(simple_tilting_radio)

    nested_four_reservoir_radio = QRadioButton("4-reservoir nested loops")
    nested_four_reservoir_radio.setToolTip(
        "Two independent two-reservoir recirculating loops, arranged concentrically."
    )
    model_group_layout.addWidget(nested_four_reservoir_radio)

    four_reservoir_checkbox = QCheckBox("Use 4-reservoir ring")
    four_reservoir_checkbox.setToolTip(
        "Build four cylindrical reservoirs in a ring, connected by four equal channels."
    )
    model_group_layout.addWidget(four_reservoir_checkbox)

    stop_time_layout = QHBoxLayout()
    stop_time_layout.addWidget(QLabel("Computation time [60-600 s]"))
    stop_time_input = QLineEdit(str(default_stop_time))
    stop_time_input.setMaximumWidth(100)
    stop_time_layout.addWidget(stop_time_input)
    model_group_layout.addLayout(stop_time_layout)

    time_step_layout = QHBoxLayout()
    time_step_layout.addWidget(QLabel("Time step [0.0001-1 s]"))
    time_step_input = QLineEdit(str(default_time_step))
    time_step_input.setMaximumWidth(100)
    time_step_layout.addWidget(time_step_input)
    model_group_layout.addLayout(time_step_layout)

    viscosity_widget = QWidget()
    viscosity_layout = QHBoxLayout()
    viscosity_layout.setContentsMargins(0, 0, 0, 0)
    viscosity_widget.setLayout(viscosity_layout)
    viscosity_layout.addWidget(QLabel("Viscosity [0.1-10 mPa*s]"))
    viscosity_slider = QSlider(Qt.Orientation.Horizontal)
    viscosity_slider.setRange(1, 100)
    viscosity_slider.setValue(int(round(default_viscosity_mpa_s * 10)))
    viscosity_slider.setToolTip(
        "Dynamic viscosity in 0.1 mPa*s increments. It is used for channel and "
        "reservoir resistance and WSS."
    )
    viscosity_layout.addWidget(viscosity_slider, 1)
    viscosity_value_label = QLabel(f"{default_viscosity_mpa_s:.1f} mPa*s")
    viscosity_value_label.setMinimumWidth(75)
    viscosity_layout.addWidget(viscosity_value_label)
    model_group_layout.addWidget(viscosity_widget)

    stl_wetting_cache_widget = QWidget()
    stl_wetting_cache_layout = QVBoxLayout()
    stl_wetting_cache_layout.setContentsMargins(0, 0, 0, 0)
    stl_wetting_cache_widget.setLayout(stl_wetting_cache_layout)

    stl_volume_bins_layout = QHBoxLayout()
    stl_volume_bins_label = QLabel("STL local-volume bins [10-2000]")
    stl_volume_bins_label.setToolTip(
        "More bins improve the local-pool wetting resolution but increase computation time."
    )
    stl_volume_bins_layout.addWidget(stl_volume_bins_label)
    stl_volume_bins_input = QLineEdit(str(default_stl_wetting_volume_bins))
    stl_volume_bins_input.setMaximumWidth(100)
    stl_volume_bins_layout.addWidget(stl_volume_bins_input)
    stl_wetting_cache_layout.addLayout(stl_volume_bins_layout)

    stl_tilt_step_layout = QHBoxLayout()
    stl_tilt_step_label = QLabel("STL tilt step [0.01-5 deg]")
    stl_tilt_step_label.setToolTip(
        "A smaller angle step improves wetting-state resolution but increases computation time."
    )
    stl_tilt_step_layout.addWidget(stl_tilt_step_label)
    stl_tilt_step_input = QLineEdit(str(default_stl_wetting_tilt_step_deg))
    stl_tilt_step_input.setMaximumWidth(100)
    stl_tilt_step_layout.addWidget(stl_tilt_step_input)
    stl_wetting_cache_layout.addLayout(stl_tilt_step_layout)

    stl_wetting_solver_widget = QWidget()
    stl_wetting_solver_layout = QHBoxLayout()
    stl_wetting_solver_layout.setContentsMargins(0, 0, 0, 0)
    stl_wetting_solver_widget.setLayout(stl_wetting_solver_layout)
    stl_wetting_solver_layout.addWidget(QLabel("STL wetting solver"))
    stl_wetting_solver_input = QComboBox()
    stl_wetting_solver_input.addItem("Cached (fast)", "cached")
    stl_wetting_solver_input.addItem(
        "Predictor-corrector (accurate)", "predictor_corrector"
    )
    stl_wetting_solver_input.setCurrentIndex(1)
    stl_wetting_solver_input.setToolTip(
        "Cached uses the selected volume and tilt bins. Predictor-corrector uses three exact STL surface evaluations per time step."
    )
    stl_wetting_solver_layout.addWidget(stl_wetting_solver_input)
    stl_wetting_cache_layout.addWidget(stl_wetting_solver_widget)
    model_group_layout.addWidget(stl_wetting_cache_widget)

    top_contact_wetting_checkbox = QCheckBox("Use top-contact wetting (90 deg)")
    top_contact_wetting_checkbox.setToolTip(
        "For an STL reservoir, change a port-connected pool to a plug after "
        "its free surface reaches the reservoir top. In two-reservoir mode, "
        "the rule is applied to both port pools within each reservoir."
    )
    model_group_layout.addWidget(top_contact_wetting_checkbox)

    two_reservoir_stl_solver_widget = QWidget()
    two_reservoir_stl_solver_layout = QHBoxLayout()
    two_reservoir_stl_solver_layout.setContentsMargins(0, 0, 0, 0)
    two_reservoir_stl_solver_widget.setLayout(two_reservoir_stl_solver_layout)
    two_reservoir_stl_solver_layout.addWidget(QLabel("2-reservoir STL solver"))
    two_reservoir_stl_solver_input = QComboBox()
    two_reservoir_stl_solver_input.addItem("Cached water levels (fast)", "cached")
    two_reservoir_stl_solver_input.addItem("Implicit water levels (stable)", "implicit")
    two_reservoir_stl_solver_input.setToolTip(
        "Cached uses one explicit update from quantized STL water levels. "
        "Implicit solves the coupled next volumes using the same cached STL water levels."
    )
    two_reservoir_stl_solver_layout.addWidget(two_reservoir_stl_solver_input)
    model_group_layout.addWidget(two_reservoir_stl_solver_widget)

    legacy_channel_driving_heads_checkbox = QCheckBox(
        "Use legacy channel-length driving heads"
    )
    legacy_channel_driving_heads_checkbox.setChecked(False)
    legacy_channel_driving_heads_checkbox.setToolTip(
        "Uses the previous channel-length times sin(tilt) approximation. "
        "Unchecked calculates each channel head from its actual mirrored port endpoints."
    )
    model_group_layout.addWidget(legacy_channel_driving_heads_checkbox)

    left_column_layout.addWidget(model_group)

    rotator_group = QGroupBox("Rotator setting")
    rotator_group_layout = QVBoxLayout()
    rotator_group.setLayout(rotator_group_layout)

    tilt_layout = QHBoxLayout()
    tilt_layout.addWidget(QLabel("Tilt [1-45 deg]"))
    tilt_input = QLineEdit(str(default_tilt))
    tilt_input.setMaximumWidth(100)
    tilt_layout.addWidget(tilt_input)
    rotator_group_layout.addLayout(tilt_layout)

    speed1_layout = QHBoxLayout()
    speed1_layout.addWidget(QLabel("Speed 1 [0.1-10 rpm]"))
    speed1_input = QLineEdit(str(default_mode_settings["symmetric"]["speed1"]))
    speed1_input.setMaximumWidth(100)
    speed1_layout.addWidget(speed1_input)
    rotator_group_layout.addLayout(speed1_layout)

    assymmetric_tilting_checkbox = QCheckBox("Use assymmetric tilting")
    assymmetric_tilting_checkbox.setChecked(False)
    rotator_group_layout.addWidget(assymmetric_tilting_checkbox)

    negative_phase_diff_checkbox = QCheckBox("Counterclockwise rotation")
    negative_phase_diff_checkbox.setChecked(False)
    negative_phase_diff_checkbox.setToolTip(
        "Unchecked uses +90 deg; checked reverses the roll phase offset to -90 deg."
    )
    rotator_group_layout.addWidget(negative_phase_diff_checkbox)

    left_column_layout.addWidget(rotator_group)

    reservoir_group = QGroupBox("Reservoir geometry")
    reservoir_group_layout = QVBoxLayout()
    reservoir_group.setLayout(reservoir_group_layout)

    use_stl_reservoir_checkbox = QCheckBox("Use STL profile reservoir")
    reservoir_group_layout.addWidget(use_stl_reservoir_checkbox)

    cylinder_radius_layout = QHBoxLayout()
    cylinder_radius_label = QLabel("Cylinder radius [1-10 mm]")
    cylinder_radius_layout.addWidget(cylinder_radius_label)
    cylinder_radius_input = QLineEdit("3.0")
    cylinder_radius_input.setMaximumWidth(100)
    cylinder_radius_layout.addWidget(cylinder_radius_input)
    reservoir_group_layout.addLayout(cylinder_radius_layout)

    cylinder_height_layout = QHBoxLayout()
    cylinder_height_label = QLabel("Cylinder height [1-10 mm]")
    cylinder_height_layout.addWidget(cylinder_height_label)
    cylinder_height_input = QLineEdit("6.0")
    cylinder_height_input.setMaximumWidth(100)
    cylinder_height_layout.addWidget(cylinder_height_input)
    reservoir_group_layout.addLayout(cylinder_height_layout)

    reservoir_width_layout = QHBoxLayout()
    reservoir_width_label = QLabel("Width [1-9 mm]")
    reservoir_width_layout.addWidget(reservoir_width_label)
    reservoir_width_input = QLineEdit(str(default_reservoir_settings["width"]))
    reservoir_width_input.setMaximumWidth(100)
    reservoir_width_layout.addWidget(reservoir_width_input)
    reservoir_group_layout.addLayout(reservoir_width_layout)

    reservoir_height_layout = QHBoxLayout()
    reservoir_height_label = QLabel("Height [1-10 mm]")
    reservoir_height_layout.addWidget(reservoir_height_label)
    reservoir_height_input = QLineEdit(str(default_reservoir_settings["height"]))
    reservoir_height_input.setMaximumWidth(100)
    reservoir_height_layout.addWidget(reservoir_height_input)
    reservoir_group_layout.addLayout(reservoir_height_layout)

    reservoir_length_layout = QHBoxLayout()
    reservoir_length_label = QLabel("Length [1-100 mm]")
    reservoir_length_layout.addWidget(reservoir_length_label)
    reservoir_length_input = QLineEdit(str(default_reservoir_settings["length"]))
    reservoir_length_input.setMaximumWidth(100)
    reservoir_length_layout.addWidget(reservoir_length_input)
    reservoir_group_layout.addLayout(reservoir_length_layout)

    reservoir_fill_layout = QHBoxLayout()
    reservoir_fill_layout.addWidget(QLabel("V_fill [0.1-10 mL]"))
    reservoir_fill_input = QLineEdit(str(default_reservoir_settings["v_fill_ml"]))
    reservoir_fill_input.setMaximumWidth(100)
    reservoir_fill_layout.addWidget(reservoir_fill_input)
    reservoir_group_layout.addLayout(reservoir_fill_layout)

    stl_xy_rotation_widget = QWidget()
    stl_xy_rotation_layout = QHBoxLayout()
    stl_xy_rotation_layout.setContentsMargins(0, 0, 0, 0)
    stl_xy_rotation_widget.setLayout(stl_xy_rotation_layout)
    stl_xy_rotation_layout.addWidget(QLabel("STL X/Y rotation [-180 to 180 deg]"))
    stl_xy_rotation_input = QLineEdit("0")
    stl_xy_rotation_input.setMaximumWidth(100)
    stl_xy_rotation_input.setToolTip(
        "Rotate the imported STL about the center of its X/Y bounding box before "
        "the port-selection previews and simulation. Positive angles are counterclockwise."
    )
    stl_xy_rotation_layout.addWidget(stl_xy_rotation_input)
    reservoir_group_layout.addWidget(stl_xy_rotation_widget)

    reservoir_profile_button = QPushButton("Load STL and Pick Ports")
    reservoir_group_layout.addWidget(reservoir_profile_button)

    reservoir_profile_info = QLabel("Using rectangular reservoir input.")
    reservoir_profile_info.setWordWrap(True)
    reservoir_group_layout.addWidget(reservoir_profile_info)

    middle_column_layout.addWidget(reservoir_group)

    outer_reservoir_group = QGroupBox("Outer-loop reservoir geometry")
    outer_reservoir_layout = QVBoxLayout()
    outer_reservoir_group.setLayout(outer_reservoir_layout)

    use_outer_stl_reservoir_checkbox = QCheckBox("Use STL profile for outer reservoir")
    outer_reservoir_layout.addWidget(use_outer_stl_reservoir_checkbox)

    outer_reservoir_width_layout = QHBoxLayout()
    outer_reservoir_width_layout.addWidget(QLabel("Outer width [1-20 mm]"))
    outer_reservoir_width_input = QLineEdit("4.0")
    outer_reservoir_width_input.setMaximumWidth(100)
    outer_reservoir_width_layout.addWidget(outer_reservoir_width_input)
    outer_reservoir_layout.addLayout(outer_reservoir_width_layout)

    outer_reservoir_height_layout = QHBoxLayout()
    outer_reservoir_height_layout.addWidget(QLabel("Outer height [1-20 mm]"))
    outer_reservoir_height_input = QLineEdit("6.0")
    outer_reservoir_height_input.setMaximumWidth(100)
    outer_reservoir_height_layout.addWidget(outer_reservoir_height_input)
    outer_reservoir_layout.addLayout(outer_reservoir_height_layout)

    outer_reservoir_length_layout = QHBoxLayout()
    outer_reservoir_length_layout.addWidget(QLabel("Outer length [1-100 mm]"))
    outer_reservoir_length_input = QLineEdit("25.0")
    outer_reservoir_length_input.setMaximumWidth(100)
    outer_reservoir_length_layout.addWidget(outer_reservoir_length_input)
    outer_reservoir_layout.addLayout(outer_reservoir_length_layout)

    outer_reservoir_fill_layout = QHBoxLayout()
    outer_reservoir_fill_layout.addWidget(QLabel("Outer-loop V_fill [0.1-10 mL]"))
    outer_reservoir_fill_input = QLineEdit(str(default_reservoir_settings["v_fill_ml"]))
    outer_reservoir_fill_input.setMaximumWidth(100)
    outer_reservoir_fill_layout.addWidget(outer_reservoir_fill_input)
    outer_reservoir_layout.addLayout(outer_reservoir_fill_layout)

    outer_stl_xy_rotation_layout = QHBoxLayout()
    outer_stl_xy_rotation_layout.addWidget(QLabel("Outer STL X/Y rotation [-180 to 180 deg]"))
    outer_stl_xy_rotation_input = QLineEdit("0")
    outer_stl_xy_rotation_input.setMaximumWidth(100)
    outer_stl_xy_rotation_layout.addWidget(outer_stl_xy_rotation_input)
    outer_reservoir_layout.addLayout(outer_stl_xy_rotation_layout)

    outer_reservoir_profile_button = QPushButton("Load Outer STL and Pick Ports")
    outer_reservoir_layout.addWidget(outer_reservoir_profile_button)
    outer_reservoir_profile_info = QLabel("Using rectangular outer reservoir input.")
    outer_reservoir_profile_info.setWordWrap(True)
    outer_reservoir_layout.addWidget(outer_reservoir_profile_info)
    middle_column_layout.addWidget(outer_reservoir_group)

    channel_group = QGroupBox("Channel geometries")
    channel_group_layout = QVBoxLayout()
    channel_group.setLayout(channel_group_layout)

    cap_force_widget = QWidget()
    cap_force_layout = QHBoxLayout()
    cap_force_layout.setContentsMargins(0, 0, 0, 0)
    cap_force_widget.setLayout(cap_force_layout)
    cap_force_layout.addWidget(QLabel("Capillary pressure [0-10 Pa]"))
    cap_force_input = QLineEdit(str(default_cap_force))
    cap_force_input.setMaximumWidth(100)
    cap_force_layout.addWidget(cap_force_input)
    channel_group_layout.addWidget(cap_force_widget)

    wet_dry_gating_checkbox = QCheckBox("Enable wet/dry port gating")
    wet_dry_gating_checkbox.setChecked(True)
    wet_dry_gating_checkbox.setToolTip(
        "When enabled, a dry donor port blocks flow and a dry receiver uses the capillary entry threshold."
    )
    channel_group_layout.addWidget(wet_dry_gating_checkbox)

    primary_geometry_widget = QWidget()
    primary_geometry_layout = QVBoxLayout()
    primary_geometry_layout.setContentsMargins(0, 0, 0, 0)
    primary_geometry_widget.setLayout(primary_geometry_layout)

    primary_geometry_title = QLabel("Primary channel geometry")
    primary_geometry_layout.addWidget(primary_geometry_title)

    primary_shape_layout = QHBoxLayout()
    primary_shape_layout.addWidget(QLabel("Cross section"))
    primary_shape_input = QComboBox()
    primary_shape_input.addItems(["Rectangular channel", "Circular pipe"])
    primary_shape_layout.addWidget(primary_shape_input)
    primary_geometry_layout.addLayout(primary_shape_layout)

    primary_width_layout = QHBoxLayout()
    primary_width_label = QLabel("Width [0.1-5 mm]")
    primary_width_layout.addWidget(primary_width_label)
    primary_width_input = QLineEdit(str(default_geometry_settings["primary"]["width"]))
    primary_width_input.setMaximumWidth(100)
    primary_width_layout.addWidget(primary_width_input)
    primary_geometry_layout.addLayout(primary_width_layout)

    primary_height_layout = QHBoxLayout()
    primary_height_label = QLabel("Height [0.1-2 mm]")
    primary_height_layout.addWidget(primary_height_label)
    primary_height_input = QLineEdit(str(default_geometry_settings["primary"]["height"]))
    primary_height_input.setMaximumWidth(100)
    primary_height_layout.addWidget(primary_height_input)
    primary_geometry_layout.addLayout(primary_height_layout)

    primary_length_layout = QHBoxLayout()
    primary_length_layout.addWidget(QLabel("Length [1-100 mm]"))
    primary_length_input = QLineEdit(str(default_geometry_settings["primary"]["length"]))
    primary_length_input.setMaximumWidth(100)
    primary_length_layout.addWidget(primary_length_input)
    primary_geometry_layout.addLayout(primary_length_layout)

    primary_radius_layout = QHBoxLayout()
    primary_radius_label = QLabel("Pipe radius [0.01-10 mm]")
    primary_radius_layout.addWidget(primary_radius_label)
    primary_radius_input = QLineEdit(str(default_geometry_settings["primary"]["radius"]))
    primary_radius_input.setMaximumWidth(100)
    primary_radius_layout.addWidget(primary_radius_input)
    primary_geometry_layout.addLayout(primary_radius_layout)

    channel_group_layout.addWidget(primary_geometry_widget)

    outer_channel_geometry_widget = QWidget()
    outer_channel_geometry_layout = QVBoxLayout()
    outer_channel_geometry_layout.setContentsMargins(0, 0, 0, 0)
    outer_channel_geometry_widget.setLayout(outer_channel_geometry_layout)
    outer_channel_geometry_layout.addWidget(QLabel("Outer-loop channel geometry (used for both outer channels)"))

    outer_channel_shape_layout = QHBoxLayout()
    outer_channel_shape_layout.addWidget(QLabel("Outer cross section"))
    outer_channel_shape_input = QComboBox()
    outer_channel_shape_input.addItems(["Rectangular channel", "Circular pipe"])
    outer_channel_shape_layout.addWidget(outer_channel_shape_input)
    outer_channel_geometry_layout.addLayout(outer_channel_shape_layout)

    outer_channel_width_layout = QHBoxLayout()
    outer_channel_width_label = QLabel("Outer width [0.1-5 mm]")
    outer_channel_width_layout.addWidget(outer_channel_width_label)
    outer_channel_width_input = QLineEdit(str(default_geometry_settings["primary"]["width"]))
    outer_channel_width_input.setMaximumWidth(100)
    outer_channel_width_layout.addWidget(outer_channel_width_input)
    outer_channel_geometry_layout.addLayout(outer_channel_width_layout)

    outer_channel_height_layout = QHBoxLayout()
    outer_channel_height_label = QLabel("Outer height [0.1-2 mm]")
    outer_channel_height_layout.addWidget(outer_channel_height_label)
    outer_channel_height_input = QLineEdit(str(default_geometry_settings["primary"]["height"]))
    outer_channel_height_input.setMaximumWidth(100)
    outer_channel_height_layout.addWidget(outer_channel_height_input)
    outer_channel_geometry_layout.addLayout(outer_channel_height_layout)

    outer_channel_length_layout = QHBoxLayout()
    outer_channel_length_layout.addWidget(QLabel("Outer length [1-100 mm]"))
    outer_channel_length_input = QLineEdit("25.0")
    outer_channel_length_input.setMaximumWidth(100)
    outer_channel_length_layout.addWidget(outer_channel_length_input)
    outer_channel_geometry_layout.addLayout(outer_channel_length_layout)

    outer_channel_radius_layout = QHBoxLayout()
    outer_channel_radius_label = QLabel("Outer pipe radius [0.01-10 mm]")
    outer_channel_radius_layout.addWidget(outer_channel_radius_label)
    outer_channel_radius_input = QLineEdit(str(default_geometry_settings["primary"]["radius"]))
    outer_channel_radius_input.setMaximumWidth(100)
    outer_channel_radius_layout.addWidget(outer_channel_radius_input)
    outer_channel_geometry_layout.addLayout(outer_channel_radius_layout)
    channel_group_layout.addWidget(outer_channel_geometry_widget)

    assymmetric_rotator_widget = QWidget()
    assymmetric_rotator_layout = QVBoxLayout()
    assymmetric_rotator_layout.setContentsMargins(0, 0, 0, 0)
    assymmetric_rotator_widget.setLayout(assymmetric_rotator_layout)

    assymmetric_rotator_layout.addWidget(QLabel("Assymmetric rotator settings"))

    speed2_layout = QHBoxLayout()
    speed2_layout.addWidget(QLabel("Speed 2 [0.1-10 rpm]"))
    speed2_input = QLineEdit(str(default_mode_settings["assymmetric"]["speed2"]))
    speed2_input.setMaximumWidth(100)
    speed2_layout.addWidget(speed2_input)
    assymmetric_rotator_layout.addLayout(speed2_layout)

    slow_start_layout = QHBoxLayout()
    slow_start_layout.addWidget(QLabel("Slow start [0.0-0.9]"))
    slow_start_input = QLineEdit(str(default_mode_settings["assymmetric"]["slow_start"]))
    slow_start_input.setMaximumWidth(100)
    slow_start_layout.addWidget(slow_start_input)
    assymmetric_rotator_layout.addLayout(slow_start_layout)

    slow_duration_layout = QHBoxLayout()
    slow_duration_layout.addWidget(QLabel("Slow duration [0.05-0.95]"))
    slow_duration_input = QLineEdit(
        str(default_mode_settings["assymmetric"]["slow_duration"])
    )
    slow_duration_input.setMaximumWidth(100)
    slow_duration_layout.addWidget(slow_duration_input)
    assymmetric_rotator_layout.addLayout(slow_duration_layout)

    rotator_group_layout.addWidget(assymmetric_rotator_widget)

    secondary_geometry_widget = QWidget()
    secondary_geometry_layout = QVBoxLayout()
    secondary_geometry_layout.setContentsMargins(0, 0, 0, 0)
    secondary_geometry_widget.setLayout(secondary_geometry_layout)

    secondary_geometry_title = QLabel("Secondary channel geometry")
    secondary_geometry_layout.addWidget(secondary_geometry_title)

    secondary_shape_layout = QHBoxLayout()
    secondary_shape_layout.addWidget(QLabel("Cross section 2"))
    secondary_shape_input = QComboBox()
    secondary_shape_input.addItems(["Rectangular channel", "Circular pipe"])
    secondary_shape_layout.addWidget(secondary_shape_input)
    secondary_geometry_layout.addLayout(secondary_shape_layout)

    secondary_width_layout = QHBoxLayout()
    secondary_width_label = QLabel("Width 2 [0.1-5 mm]")
    secondary_width_layout.addWidget(secondary_width_label)
    secondary_width_input = QLineEdit(str(default_geometry_settings["secondary"]["width"]))
    secondary_width_input.setMaximumWidth(100)
    secondary_width_layout.addWidget(secondary_width_input)
    secondary_geometry_layout.addLayout(secondary_width_layout)

    secondary_height_layout = QHBoxLayout()
    secondary_height_label = QLabel("Height 2 [0.1-2 mm]")
    secondary_height_layout.addWidget(secondary_height_label)
    secondary_height_input = QLineEdit(str(default_geometry_settings["secondary"]["height"]))
    secondary_height_input.setMaximumWidth(100)
    secondary_height_layout.addWidget(secondary_height_input)
    secondary_geometry_layout.addLayout(secondary_height_layout)

    secondary_length_layout = QHBoxLayout()
    secondary_length_layout.addWidget(QLabel("Length 2 [1-100 mm]"))
    secondary_length_input = QLineEdit(str(default_geometry_settings["secondary"]["length"]))
    secondary_length_input.setMaximumWidth(100)
    secondary_length_layout.addWidget(secondary_length_input)
    secondary_geometry_layout.addLayout(secondary_length_layout)

    secondary_radius_layout = QHBoxLayout()
    secondary_radius_label = QLabel("Pipe radius 2 [0.01-10 mm]")
    secondary_radius_layout.addWidget(secondary_radius_label)
    secondary_radius_input = QLineEdit(str(default_geometry_settings["secondary"]["radius"]))
    secondary_radius_input.setMaximumWidth(100)
    secondary_radius_layout.addWidget(secondary_radius_input)
    secondary_geometry_layout.addLayout(secondary_radius_layout)

    channel_group_layout.addWidget(secondary_geometry_widget)

    right_column_layout.addWidget(channel_group)

    plot_group = QGroupBox("Plot settings")
    plot_group_layout = QVBoxLayout()
    plot_group.setLayout(plot_group_layout)

    plot_tilts_checkbox = QCheckBox("Show pitch and roll")
    plot_tilts_checkbox.setChecked(True)
    plot_group_layout.addWidget(plot_tilts_checkbox)

    plot_start_time_layout = QHBoxLayout()
    plot_start_time_layout.addWidget(QLabel("Plot start time [s]"))
    plot_start_time_input = QLineEdit("0")
    plot_start_time_input.setMaximumWidth(100)
    plot_start_time_layout.addWidget(plot_start_time_input)
    plot_group_layout.addLayout(plot_start_time_layout)

    flow_channel_selection_layout = QHBoxLayout()
    plot_q1_checkbox = QCheckBox("Plot flow rate channel 1")
    plot_q1_checkbox.setChecked(True)
    plot_q2_checkbox = QCheckBox("Plot flow rate channel 2")
    plot_q2_checkbox.setChecked(False)
    flow_channel_selection_layout.addWidget(plot_q1_checkbox)
    flow_channel_selection_layout.addWidget(plot_q2_checkbox)
    plot_group_layout.addLayout(flow_channel_selection_layout)

    plot_q3_checkbox = QCheckBox("Plot flow rate channel 3")
    plot_q3_checkbox.setChecked(True)
    plot_q4_checkbox = QCheckBox("Plot flow rate channel 4")
    plot_q4_checkbox.setChecked(True)
    outer_flow_selection_layout = QHBoxLayout()
    outer_flow_selection_layout.addWidget(plot_q3_checkbox)
    outer_flow_selection_layout.addWidget(plot_q4_checkbox)
    plot_group_layout.addLayout(outer_flow_selection_layout)

    ring_global_direction_checkbox = QCheckBox(
        "Use global x/y direction for 4-ring flows"
    )
    ring_global_direction_checkbox.setChecked(False)
    ring_global_direction_checkbox.setToolTip(
        "Display Q1 and Q3 relative to +x, and Q2 and Q4 relative to +y. "
        "This plots -Q3 and -Q4 only; the ring solver and exported data retain "
        "their circulation-direction convention."
    )
    plot_group_layout.addWidget(ring_global_direction_checkbox)

    flow_low_pass_widget = QWidget()
    flow_low_pass_layout = QVBoxLayout()
    flow_low_pass_layout.setContentsMargins(0, 0, 0, 0)
    flow_low_pass_widget.setLayout(flow_low_pass_layout)
    flow_low_pass_checkbox = QCheckBox("Smooth flow plot (low-pass)")
    flow_low_pass_checkbox.setToolTip(
        "Applies a zero-phase low-pass filter to the displayed flow curves only."
    )
    flow_low_pass_layout.addWidget(flow_low_pass_checkbox)
    flow_low_pass_cutoff_layout = QHBoxLayout()
    flow_low_pass_cutoff_layout.setContentsMargins(16, 0, 0, 0)
    flow_low_pass_cutoff_layout.addWidget(QLabel("Cutoff frequency [Hz]"))
    flow_low_pass_cutoff_input = QLineEdit("0.1")
    flow_low_pass_cutoff_input.setMaximumWidth(100)
    flow_low_pass_cutoff_layout.addWidget(flow_low_pass_cutoff_input)
    flow_low_pass_layout.addLayout(flow_low_pass_cutoff_layout)
    plot_group_layout.addWidget(flow_low_pass_widget)

    plot_local_volumes_checkbox = QCheckBox("Plot reservoir volumes")
    plot_local_volumes_checkbox.setChecked(True)
    plot_group_layout.addWidget(plot_local_volumes_checkbox)

    plot_midpoint_gravity_pressure_checkbox = QCheckBox(
        "Plot midpoint channel driving pressure"
    )
    plot_midpoint_gravity_pressure_checkbox.setChecked(False)
    plot_midpoint_gravity_pressure_checkbox.setToolTip(
        "Plots half of the signed, capillary-corrected driving pressure for each channel."
    )
    plot_group_layout.addWidget(plot_midpoint_gravity_pressure_checkbox)

    plot_port_heads_checkbox = QCheckBox("Plot inlet/outlet water-table heads")
    plot_port_heads_checkbox.setChecked(False)
    plot_group_layout.addWidget(plot_port_heads_checkbox)

    plot_inner_outer_inlet_hydrostatic_checkbox = QCheckBox(
        "Plot channel 1 midpoint pressure difference"
    )
    plot_inner_outer_outlet_hydrostatic_checkbox = QCheckBox(
        "Plot channel 2 midpoint pressure difference"
    )
    plot_inner_outer_raw_pressure_checkbox = QCheckBox(
        "Plot water-table pressure relative to circuit centre"
    )
    pressure_difference_selection_layout = QHBoxLayout()
    for checkbox in (
        plot_inner_outer_inlet_hydrostatic_checkbox,
        plot_inner_outer_outlet_hydrostatic_checkbox,
    ):
        checkbox.setChecked(False)
        pressure_difference_selection_layout.addWidget(checkbox)
    plot_group_layout.addLayout(pressure_difference_selection_layout)
    plot_inner_outer_raw_pressure_checkbox.setChecked(False)
    plot_group_layout.addWidget(plot_inner_outer_raw_pressure_checkbox)
    plot_inner_outer_inlet_hydrostatic_checkbox.setToolTip(
        "Plots outer minus inner pressure potential at the channel 1 midpoint. Dry channels are omitted."
    )
    plot_inner_outer_outlet_hydrostatic_checkbox.setToolTip(
        "Plots outer minus inner pressure potential at the channel 2 midpoint. Dry channels are omitted."
    )
    plot_inner_outer_raw_pressure_checkbox.setToolTip(
        "Plots signed water-table pressure potentials at all four channel inlets, "
        "relative to the nested-circuit centre. Dry ports are omitted."
    )

    plot_wss_checkbox = QCheckBox("Plot WSS at channel bottom")
    plot_wss_checkbox.setChecked(False)
    plot_group_layout.addWidget(plot_wss_checkbox)

    plot_wss_reference_widget = QWidget()
    plot_wss_reference_row = QVBoxLayout()
    plot_wss_reference_row.setContentsMargins(0, 0, 0, 0)
    plot_wss_reference_widget.setLayout(plot_wss_reference_row)
    plot_wss_reference_checkbox = QCheckBox("Plot WSS reference")
    plot_wss_reference_checkbox.setChecked(False)
    plot_wss_reference_row.addWidget(plot_wss_reference_checkbox)

    reference_wss_values_widget = QWidget()
    reference_wss_values_layout = QVBoxLayout()
    reference_wss_values_layout.setContentsMargins(16, 0, 0, 0)
    reference_wss_values_widget.setLayout(reference_wss_values_layout)

    reference_wss_layout = QHBoxLayout()
    reference_wss_layout.addWidget(QLabel("WSS reference [0-10 Pa]"))
    reference_wss_input = QLineEdit(str(default_reference_wss_pa))
    reference_wss_input.setMaximumWidth(100)
    reference_wss_layout.addWidget(reference_wss_input)
    reference_wss_values_layout.addLayout(reference_wss_layout)

    reference_flow_layout = QHBoxLayout()
    reference_flow_layout.addWidget(QLabel("Reference flow rate [0-10 uL/s]"))
    reference_flow_input = QLineEdit(str(default_reference_flow_uL_s))
    reference_flow_input.setMaximumWidth(100)
    reference_flow_layout.addWidget(reference_flow_input)
    reference_wss_values_layout.addLayout(reference_flow_layout)

    plot_wss_reference_row.addWidget(reference_wss_values_widget)
    plot_group_layout.addWidget(plot_wss_reference_widget)

    right_column_layout.addWidget(plot_group)

    fourier_group_layout.insertWidget(0, fourier_info_label)
    right_column_layout.addWidget(fourier_group)

    channel_wss_group = QGroupBox("Channel WSS")
    channel_wss_group_layout = QVBoxLayout()
    channel_wss_group.setLayout(channel_wss_group_layout)

    channel_wss_wall_widget = QWidget()
    channel_wss_wall_layout = QHBoxLayout()
    channel_wss_wall_layout.setContentsMargins(0, 0, 0, 0)
    channel_wss_wall_layout.addWidget(QLabel("Wall"))
    channel_wss_wall_input = QComboBox()
    channel_wss_wall_input.addItems(["Bottom wall", "Left side wall"])
    channel_wss_wall_layout.addWidget(channel_wss_wall_input)
    channel_wss_wall_widget.setLayout(channel_wss_wall_layout)
    channel_wss_group_layout.addWidget(channel_wss_wall_widget)

    channel_wss_flow_mode_layout = QHBoxLayout()
    channel_wss_flow_mode_layout.addWidget(QLabel("Representative flow"))
    channel_wss_flow_mode_input = QComboBox()
    channel_wss_flow_mode_input.addItems(["Average |Q1| over one period", "Maximum |Q1| over one period"])
    channel_wss_flow_mode_layout.addWidget(channel_wss_flow_mode_input)
    channel_wss_group_layout.addLayout(channel_wss_flow_mode_layout)

    channel_wss_range_start_widget = QWidget()
    channel_wss_start_layout = QHBoxLayout()
    channel_wss_start_layout.setContentsMargins(0, 0, 0, 0)
    channel_wss_start_layout.addWidget(QLabel("Range start [mm]"))
    channel_wss_start_input = QLineEdit("0.0")
    channel_wss_start_input.setMaximumWidth(120)
    channel_wss_start_layout.addWidget(channel_wss_start_input)
    channel_wss_range_start_widget.setLayout(channel_wss_start_layout)
    channel_wss_group_layout.addWidget(channel_wss_range_start_widget)

    channel_wss_range_end_widget = QWidget()
    channel_wss_end_layout = QHBoxLayout()
    channel_wss_end_layout.setContentsMargins(0, 0, 0, 0)
    channel_wss_end_layout.addWidget(QLabel("Range end [mm]"))
    channel_wss_end_input = QLineEdit(str(default_geometry_settings["primary"]["width"]))
    channel_wss_end_input.setMaximumWidth(120)
    channel_wss_end_layout.addWidget(channel_wss_end_input)
    channel_wss_range_end_widget.setLayout(channel_wss_end_layout)
    channel_wss_group_layout.addWidget(channel_wss_range_end_widget)

    channel_wss_hint_label = QLabel("")
    channel_wss_hint_label.setWordWrap(True)
    channel_wss_group_layout.addWidget(channel_wss_hint_label)

    channel_wss_button_row = QHBoxLayout()
    channel_wss_button = QPushButton("Plot Channel WSS")
    channel_wss_value_label = QLabel("")
    channel_wss_value_label.setWordWrap(True)
    save_channel_wss_image_button = QPushButton("Save Channel WSS Image")
    export_channel_wss_data_button = QPushButton("Export Channel WSS CSV")
    channel_wss_button.setEnabled(False)
    save_channel_wss_image_button.setEnabled(False)
    export_channel_wss_data_button.setEnabled(False)
    channel_wss_button_row.addWidget(channel_wss_button)
    channel_wss_button_row.addWidget(channel_wss_value_label, 1)
    channel_wss_button_row.addWidget(save_channel_wss_image_button)
    channel_wss_button_row.addWidget(export_channel_wss_data_button)
    channel_wss_group_layout.addLayout(channel_wss_button_row)

    right_column_layout.addWidget(channel_wss_group)

    control_group = QGroupBox("Simulation controls")
    control_group_layout = QVBoxLayout()
    control_group.setLayout(control_group_layout)

    status_label = QLabel(
        "Pick the traces, then generate a plot and choose where to save it."
    )
    status_label.setWordWrap(True)
    control_group_layout.addWidget(status_label)

    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)
    progress_bar.setValue(0)
    control_group_layout.addWidget(progress_bar)

    progress_label = QLabel("Idle")
    control_group_layout.addWidget(progress_label)

    circuit_centre_label = QLabel(
        "Nested-circuit centre [mm]: generate a nested-loop plot to display it."
    )
    circuit_centre_label.setWordWrap(True)
    control_group_layout.addWidget(circuit_centre_label)

    plot_button = QPushButton("Generate Plot")
    geometry_layout_button = QPushButton("Show Geometry Layout")
    animation_button = QPushButton("Save 3D Animation")
    plot_animation_button = QPushButton("Save Plot Animation")
    animation_frame_count_label = QLabel("Animation frames [24-10000]")
    animation_frame_count_input = QLineEdit("240")
    animation_frame_count_input.setMaximumWidth(65)
    animation_frame_count_input.setToolTip(
        "More frames improve temporal detail but take longer to render. "
        "Playback remains at 24 fps."
    )
    save_image_button = QPushButton("Save Plot Image")
    export_data_button = QPushButton("Save Plot Data")
    animation_button.setEnabled(False)
    geometry_layout_button.setEnabled(False)
    plot_animation_button.setEnabled(False)
    save_image_button.setEnabled(False)
    export_data_button.setEnabled(False)
    control_group_layout.addWidget(plot_button)
    control_group_layout.addWidget(geometry_layout_button)

    animation_controls_layout = QHBoxLayout()
    animation_controls_layout.addWidget(animation_button)
    animation_controls_layout.addWidget(animation_frame_count_label)
    animation_controls_layout.addWidget(animation_frame_count_input)
    control_group_layout.addLayout(animation_controls_layout)
    control_group_layout.addWidget(plot_animation_button)
    control_group_layout.addWidget(save_image_button)
    control_group_layout.addWidget(export_data_button)
    left_column_layout.addWidget(control_group)

    fourier_action_layout = QHBoxLayout()
    fourier_fit_button = QPushButton("Plot Fourier Series")
    export_fourier_button = QPushButton("Export Fourier Series")
    fourier_fit_button.setEnabled(False)
    export_fourier_button.setEnabled(False)
    fourier_action_layout.addWidget(fourier_fit_button)
    fourier_action_layout.addWidget(export_fourier_button)
    fourier_group_layout.addLayout(fourier_action_layout)

    plot_state = {
        "data": None,
        "figure": None,
        "fourier_fit": None,
        "layout_inputs": None,
        "layout_figure": None,
        "is_updating": False,
    }
    channel_wss_state = {
        "figure": None,
        "result": None,
        "selection": None,
        "selected_average_wss_pa": None,
    }
    reservoir_profile_state = {
        "template": None,
    }
    outer_reservoir_profile_state = {
        "template": None,
    }

    def clear_channel_wss_state():
        if channel_wss_state["figure"] is not None:
            plt.close(channel_wss_state["figure"])
            channel_wss_state["figure"] = None
        channel_wss_state["result"] = None
        channel_wss_state["selection"] = None
        channel_wss_state["selected_average_wss_pa"] = None
        channel_wss_value_label.setText("")
        save_channel_wss_image_button.setEnabled(False)
        export_channel_wss_data_button.setEnabled(False)

    def clear_cached_plot(reason_text=None):
        plot_state["data"] = None
        plot_state["fourier_fit"] = None
        plot_state["layout_inputs"] = None
        if plot_state["figure"] is not None:
            plt.close(plot_state["figure"])
            plot_state["figure"] = None
        clear_channel_wss_state()
        animation_button.setEnabled(True)
        fourier_fit_button.setEnabled(False)
        export_fourier_button.setEnabled(False)
        fourier_info_label.setText(
            "Generate a plot, then select a flow curve and number of terms."
        )
        channel_wss_button.setEnabled(False)
        geometry_layout_button.setEnabled(False)
        plot_animation_button.setEnabled(False)
        save_image_button.setEnabled(False)
        export_data_button.setEnabled(False)
        if reason_text:
            status_label.setText(reason_text)

    def update_phase_difference():
        global phaseDiff
        phaseDiff = -math.pi / 2 if negative_phase_diff_checkbox.isChecked() else math.pi / 2
        phase_degrees = "-90" if negative_phase_diff_checkbox.isChecked() else "+90"
        clear_cached_plot(
            f"Pitch/roll phase difference set to {phase_degrees} deg. Generate a fresh plot to apply it."
        )

    def build_primary_geometry_from_inputs():
        length_mm = validate_float_range(
            primary_length_input.text(), "Primary channel length", 1, 100
        )
        if primary_shape_input.currentText() == "Circular pipe":
            return build_geometry(
                length=length_mm,
                shape="circular",
                radius=validate_float_range(
                    primary_radius_input.text(), "Primary pipe radius", 0.01, 10
                ),
            )
        return build_geometry(
            validate_float_range(primary_width_input.text(), "Primary channel width", 0.1, 5),
            validate_float_range(primary_height_input.text(), "Primary channel height", 0.1, 2),
            length_mm,
        )

    def build_secondary_geometry_from_inputs():
        length_mm = validate_float_range(
            secondary_length_input.text(), "Secondary channel length", 1, 100
        )
        if secondary_shape_input.currentText() == "Circular pipe":
            return build_geometry(
                length=length_mm,
                shape="circular",
                radius=validate_float_range(
                    secondary_radius_input.text(), "Secondary pipe radius", 0.01, 10
                ),
            )
        return build_geometry(
            validate_float_range(secondary_width_input.text(), "Secondary channel width", 0.1, 5),
            validate_float_range(secondary_height_input.text(), "Secondary channel height", 0.1, 2),
            length_mm,
        )

    def build_outer_channel_geometry_from_inputs():
        length_mm = validate_float_range(
            outer_channel_length_input.text(), "Outer channel length", 1, 100
        )
        if outer_channel_shape_input.currentText() == "Circular pipe":
            return build_geometry(
                length=length_mm,
                shape="circular",
                radius=validate_float_range(
                    outer_channel_radius_input.text(), "Outer pipe radius", 0.01, 10
                ),
            )
        return build_geometry(
            validate_float_range(outer_channel_width_input.text(), "Outer channel width", 0.1, 5),
            validate_float_range(outer_channel_height_input.text(), "Outer channel height", 0.1, 2),
            length_mm,
        )

    def update_channel_shape_visibility():
        primary_is_pipe = primary_shape_input.currentText() == "Circular pipe"
        secondary_is_pipe = secondary_shape_input.currentText() == "Circular pipe"
        outer_is_pipe = outer_channel_shape_input.currentText() == "Circular pipe"
        for widget in (
            primary_width_label,
            primary_width_input,
            primary_height_label,
            primary_height_input,
        ):
            widget.setVisible(not primary_is_pipe)
        for widget in (primary_radius_label, primary_radius_input):
            widget.setVisible(primary_is_pipe)
        for widget in (
            secondary_width_label,
            secondary_width_input,
            secondary_height_label,
            secondary_height_input,
        ):
            widget.setVisible(not secondary_is_pipe)
        for widget in (secondary_radius_label, secondary_radius_input):
            widget.setVisible(secondary_is_pipe)
        for widget in (
            outer_channel_width_label,
            outer_channel_width_input,
            outer_channel_height_label,
            outer_channel_height_input,
        ):
            widget.setVisible(not outer_is_pipe)
        for widget in (outer_channel_radius_label, outer_channel_radius_input):
            widget.setVisible(outer_is_pipe)

        plot_wss_checkbox.setText(
            "Plot WSS" if primary_is_pipe else "Plot WSS at channel bottom"
        )
        plot_wss_reference_widget.setVisible(not primary_is_pipe)
        if primary_is_pipe:
            plot_wss_reference_checkbox.setChecked(False)

        # A circular pipe has uniform wall WSS, so no wall or range selection applies.
        channel_wss_group.setVisible(True)
        for widget in (
            channel_wss_wall_widget,
            channel_wss_range_start_widget,
            channel_wss_range_end_widget,
            channel_wss_hint_label,
        ):
            widget.setVisible(not primary_is_pipe)
        channel_wss_button.setText(
            "Calculate Channel WSS" if primary_is_pipe else "Plot Channel WSS"
        )
        channel_wss_value_label.setVisible(primary_is_pipe)
        save_channel_wss_image_button.setVisible(not primary_is_pipe)
        export_channel_wss_data_button.setVisible(not primary_is_pipe)
        if primary_is_pipe:
            clear_channel_wss_state()

    def on_channel_shape_changed():
        update_channel_shape_visibility()
        clear_cached_plot("Channel shape changed. Generate a fresh plot to apply it.")

    def update_mode_visibility():
        is_single_reservoir_mode = single_reservoir_radio.isChecked()
        is_simple_tilting_mode = simple_tilting_radio.isChecked()
        is_nested_four_reservoir_mode = nested_four_reservoir_radio.isChecked()
        is_four_reservoir_mode = (
            is_simple_tilting_mode and four_reservoir_checkbox.isChecked()
        )
        is_two_reservoir_mode = two_reservoir_radio.isChecked()
        is_advanced_mode = advanced_mode_checkbox.isChecked()
        has_nested_stl = is_nested_four_reservoir_mode and (
            use_stl_reservoir_checkbox.isChecked()
            or use_outer_stl_reservoir_checkbox.isChecked()
        )
        use_assymmetric_tilting = (
            assymmetric_tilting_checkbox.isChecked()
            and not is_nested_four_reservoir_mode
        )
        assymmetric_rotator_widget.setVisible(
            use_assymmetric_tilting
        )
        assymmetric_tilting_checkbox.setVisible(
            not is_simple_tilting_mode and not is_nested_four_reservoir_mode
        )
        assymmetric_tilting_checkbox.setEnabled(
            not is_simple_tilting_mode and not is_nested_four_reservoir_mode
        )
        four_reservoir_checkbox.setVisible(is_simple_tilting_mode)
        four_reservoir_checkbox.setEnabled(is_simple_tilting_mode)
        outer_reservoir_group.setVisible(is_nested_four_reservoir_mode)
        outer_channel_geometry_widget.setVisible(is_nested_four_reservoir_mode)
        secondary_geometry_widget.setVisible(is_two_reservoir_mode and use_assymmetric_tilting)
        plot_q2_checkbox.setVisible(not is_single_reservoir_mode and (not is_simple_tilting_mode or is_four_reservoir_mode))
        plot_q2_checkbox.setEnabled(not is_single_reservoir_mode and (not is_simple_tilting_mode or is_four_reservoir_mode))
        plot_q3_checkbox.setVisible(is_four_reservoir_mode or is_nested_four_reservoir_mode)
        plot_q3_checkbox.setEnabled(is_four_reservoir_mode or is_nested_four_reservoir_mode)
        plot_q4_checkbox.setVisible(is_four_reservoir_mode or is_nested_four_reservoir_mode)
        plot_q4_checkbox.setEnabled(is_four_reservoir_mode or is_nested_four_reservoir_mode)
        ring_global_direction_checkbox.setVisible(is_four_reservoir_mode)
        ring_global_direction_checkbox.setEnabled(is_four_reservoir_mode)
        plot_local_volumes_checkbox.setVisible(True)
        plot_local_volumes_checkbox.setEnabled(True)
        plot_midpoint_gravity_pressure_checkbox.setVisible(is_two_reservoir_mode)
        plot_midpoint_gravity_pressure_checkbox.setEnabled(is_two_reservoir_mode)
        plot_port_heads_checkbox.setVisible(
            is_single_reservoir_mode or (is_simple_tilting_mode and not is_four_reservoir_mode)
        )
        plot_port_heads_checkbox.setEnabled(plot_port_heads_checkbox.isVisible())
        for checkbox in (
            plot_inner_outer_inlet_hydrostatic_checkbox,
            plot_inner_outer_outlet_hydrostatic_checkbox,
            plot_inner_outer_raw_pressure_checkbox,
        ):
            checkbox.setVisible(is_nested_four_reservoir_mode)
            checkbox.setEnabled(is_nested_four_reservoir_mode)
            if not is_nested_four_reservoir_mode:
                checkbox.setChecked(False)
        wet_dry_gating_checkbox.setVisible(
            is_advanced_mode and not is_simple_tilting_mode
        )
        wet_dry_gating_checkbox.setEnabled(
            is_advanced_mode and not is_simple_tilting_mode
        )
        stl_wetting_cache_widget.setVisible(
            is_advanced_mode
            and (is_single_reservoir_mode or is_two_reservoir_mode or is_nested_four_reservoir_mode)
            and (use_stl_reservoir_checkbox.isChecked() or has_nested_stl)
        )
        stl_wetting_solver_widget.setVisible(
            is_advanced_mode
            and is_single_reservoir_mode
            and use_stl_reservoir_checkbox.isChecked()
        )
        top_contact_wetting_checkbox.setVisible(
            is_advanced_mode
            and (is_single_reservoir_mode or is_two_reservoir_mode or is_nested_four_reservoir_mode)
            and (
                reservoir_profile_state["template"] is not None
                or outer_reservoir_profile_state["template"] is not None
            )
        )
        top_contact_wetting_checkbox.setEnabled(
            top_contact_wetting_checkbox.isVisible()
        )
        two_reservoir_stl_solver_widget.setVisible(
            is_advanced_mode
            and (is_two_reservoir_mode or is_nested_four_reservoir_mode)
            and (
                reservoir_profile_state["template"] is not None
                or outer_reservoir_profile_state["template"] is not None
            )
        )
        legacy_channel_driving_heads_checkbox.setVisible(
            is_advanced_mode
            and (is_two_reservoir_mode or is_nested_four_reservoir_mode)
        )
        legacy_channel_driving_heads_checkbox.setEnabled(
            legacy_channel_driving_heads_checkbox.isVisible()
        )
        if (
            (is_single_reservoir_mode or (is_simple_tilting_mode and not is_four_reservoir_mode))
            and fourier_curve_input.currentData() in ("q_2", "q_3", "q_4")
        ):
            fourier_curve_input.setCurrentIndex(0)
        # All modes, including simple tilting, have a dedicated animation path.
        animation_button.setEnabled(True)
        if not is_nested_four_reservoir_mode:
            reservoir_group.setTitle("Reservoir geometry")
        if is_single_reservoir_mode:
            plot_q2_checkbox.setChecked(False)
            plot_midpoint_gravity_pressure_checkbox.setChecked(False)
            plot_q1_checkbox.setText("Plot recirculation flow")
            plot_local_volumes_checkbox.setText("Plot V_left_local / V_right_local")
            plot_port_heads_checkbox.setText("Plot inlet/outlet water-table heads")
            plot_wss_checkbox.setText("Plot channel WSS")
            primary_geometry_title.setText("Channel geometry")
            if use_assymmetric_tilting:
                status_label.setText(
                    "Single-reservoir mode uses one reservoir, one recirculating channel, and assymmetric tilting."
                )
            else:
                status_label.setText(
                    "Single-reservoir mode uses one reservoir and one recirculating channel."
                )
        elif is_nested_four_reservoir_mode:
            reservoir_group.setTitle("Inner-loop reservoir geometry")
            plot_q1_checkbox.setChecked(True)
            plot_q2_checkbox.setChecked(False)
            plot_q3_checkbox.setChecked(False)
            plot_q4_checkbox.setChecked(False)
            plot_midpoint_gravity_pressure_checkbox.setChecked(False)
            plot_q1_checkbox.setText("Plot inner channel 1 flow")
            plot_q2_checkbox.setText("Plot inner channel 2 flow")
            plot_q3_checkbox.setText("Plot outer channel 1 flow")
            plot_q4_checkbox.setText("Plot outer channel 2 flow")
            plot_local_volumes_checkbox.setText("Plot inner and outer reservoir volumes")
            plot_port_heads_checkbox.setChecked(False)
            plot_wss_checkbox.setText("Plot inner-loop channel WSS")
            primary_geometry_title.setText("Inner-loop channel geometry")
            status_label.setText(
                "Nested-loop mode uses two independent recirculating reservoir pairs. "
                "The outer reservoir and channel must be longer than the inner pair."
            )
        elif is_simple_tilting_mode:
            reservoir_group.setTitle("Reservoir geometry")
            if not is_four_reservoir_mode:
                plot_q2_checkbox.setChecked(False)
                plot_q3_checkbox.setChecked(False)
                plot_q4_checkbox.setChecked(False)
            else:
                plot_q2_checkbox.setChecked(False)
                plot_q3_checkbox.setChecked(False)
                plot_q4_checkbox.setChecked(False)
            plot_midpoint_gravity_pressure_checkbox.setChecked(False)
            plot_q1_checkbox.setText(
                "Plot flow rate channel 1" if is_four_reservoir_mode else "Plot cylinder-to-cylinder flow"
            )
            plot_q2_checkbox.setText("Plot flow rate channel 2")
            plot_q3_checkbox.setText("Plot flow rate channel 3")
            plot_q4_checkbox.setText("Plot flow rate channel 4")
            plot_local_volumes_checkbox.setText(
                "Plot four cylinder volumes" if is_four_reservoir_mode else "Plot cylinder 1 / cylinder 2 volumes"
            )
            plot_port_heads_checkbox.setText("Plot cylinder port heads")
            plot_wss_checkbox.setText("Plot channel WSS")
            primary_geometry_title.setText("Connecting channel geometry")
            status_label.setText(
                "Four-reservoir ring uses four identical cylindrical reservoirs and four equal connecting channels."
                if is_four_reservoir_mode
                else "Simple tilting uses two identical cylindrical reservoirs and one connecting channel."
            )
        elif use_assymmetric_tilting:
            reservoir_group.setTitle("Reservoir geometry")
            plot_q1_checkbox.setText("Plot flow rate channel 1")
            plot_local_volumes_checkbox.setText("Plot reservoir 1 / reservoir 2 volumes")
            plot_wss_checkbox.setText("Plot WSS channel bottom")
            primary_geometry_title.setText("Primary channel geometry")
            secondary_geometry_title.setText("Secondary channel geometry")
            plot_port_heads_checkbox.setChecked(False)
            status_label.setText(
                "Assymmetric mode uses two reservoirs connected by two unequal channels."
            )
        else:
            reservoir_group.setTitle("Reservoir geometry")
            plot_q1_checkbox.setText("Plot flow rate channel 1")
            plot_local_volumes_checkbox.setText("Plot reservoir 1 / reservoir 2 volumes")
            plot_wss_checkbox.setText("Plot WSS channel bottom")
            primary_geometry_title.setText("Primary channel geometry")
            secondary_geometry_title.setText("Secondary channel geometry")
            plot_port_heads_checkbox.setChecked(False)
            status_label.setText(
                "Symmetric mode uses two mirrored reservoirs connected by two equal channels."
            )
        if not (is_single_reservoir_mode or is_simple_tilting_mode or is_nested_four_reservoir_mode):
            plot_q2_checkbox.setText("Plot flow rate channel 2")
            plot_q3_checkbox.setChecked(False)
            plot_q4_checkbox.setChecked(False)
        update_channel_shape_visibility()
        clear_cached_plot("Mode changed. Generate a fresh plot for the selected topology.")

    def update_reference_wss_visibility():
        reference_wss_values_widget.setVisible(
            plot_wss_reference_checkbox.isChecked()
            and primary_shape_input.currentText() != "Circular pipe"
        )

    def update_reservoir_mode_visibility():
        is_simple_tilting_mode = simple_tilting_radio.isChecked()
        is_nested_four_reservoir_mode = nested_four_reservoir_radio.isChecked()
        is_profile_mode = use_stl_reservoir_checkbox.isChecked()
        is_outer_profile_mode = use_outer_stl_reservoir_checkbox.isChecked()
        is_advanced_mode = advanced_mode_checkbox.isChecked()
        for widget in (
            reservoir_width_label,
            reservoir_width_input,
            reservoir_height_label,
            reservoir_height_input,
            reservoir_length_label,
            reservoir_length_input,
        ):
            widget.setVisible(not is_simple_tilting_mode)
            widget.setEnabled(not is_profile_mode and not is_simple_tilting_mode)
        for widget in (cylinder_radius_label, cylinder_radius_input, cylinder_height_label, cylinder_height_input):
            widget.setVisible(is_simple_tilting_mode)
            widget.setEnabled(is_simple_tilting_mode)
        use_stl_reservoir_checkbox.setVisible(not is_simple_tilting_mode)
        reservoir_profile_button.setVisible(not is_simple_tilting_mode)
        reservoir_profile_info.setVisible(not is_simple_tilting_mode)
        stl_xy_rotation_widget.setVisible(not is_simple_tilting_mode)
        stl_xy_rotation_widget.setEnabled(not is_simple_tilting_mode)
        reservoir_profile_button.setEnabled(is_profile_mode and not is_simple_tilting_mode)
        stl_wetting_cache_widget.setVisible(
            is_advanced_mode
            and (single_reservoir_radio.isChecked() or two_reservoir_radio.isChecked())
            and is_profile_mode
            and not is_simple_tilting_mode
        )
        stl_wetting_solver_widget.setVisible(
            is_advanced_mode
            and single_reservoir_radio.isChecked()
            and is_profile_mode
            and not is_simple_tilting_mode
        )
        top_contact_wetting_checkbox.setVisible(
            is_advanced_mode
            and (single_reservoir_radio.isChecked() or two_reservoir_radio.isChecked())
            and is_profile_mode
            and not is_simple_tilting_mode
            and reservoir_profile_state["template"] is not None
        )
        top_contact_wetting_checkbox.setEnabled(
            top_contact_wetting_checkbox.isVisible()
        )
        two_reservoir_stl_solver_widget.setVisible(
            is_advanced_mode
            and two_reservoir_radio.isChecked()
            and is_profile_mode
            and not is_simple_tilting_mode
            and reservoir_profile_state["template"] is not None
        )
        if is_profile_mode:
            if reservoir_profile_state["template"] is None:
                reservoir_profile_info.setText(
                    "Load an extruded STL reservoir and click the left and right ports."
                )
        else:
            reservoir_profile_info.setText("Using rectangular reservoir input.")
        for widget in (
            outer_reservoir_width_input,
            outer_reservoir_height_input,
            outer_reservoir_length_input,
        ):
            widget.setEnabled(is_nested_four_reservoir_mode and not is_outer_profile_mode)
        outer_reservoir_profile_button.setEnabled(
            is_nested_four_reservoir_mode and is_outer_profile_mode
        )
        outer_reservoir_profile_info.setVisible(is_nested_four_reservoir_mode)
        if not is_outer_profile_mode:
            outer_reservoir_profile_info.setText("Using rectangular outer reservoir input.")
        elif outer_reservoir_profile_state["template"] is None:
            outer_reservoir_profile_info.setText(
                "Load the outer STL after the inner reservoir geometry is ready."
            )

    def update_progress(completed, total, message):
        percent = 0 if total <= 0 else min(100, max(0, completed / total * 100))
        progress_bar.setValue(int(percent))
        progress_label.setText(f"{message} ({percent:.1f}%)")
        app.processEvents()

    def get_reference_wss_values():
        if not plot_wss_reference_checkbox.isChecked():
            return default_reference_wss_pa, default_reference_flow_uL_s
        reference_wss_pa = validate_float_range(
            reference_wss_input.text(), "WSS reference", 0, 10
        )
        reference_flow_uL_s = validate_float_range(
            reference_flow_input.text(), "Reference flow rate", 0, 10
        )
        if reference_flow_uL_s <= 0.0:
            raise ValueError("Reference flow rate must be greater than zero.")
        return reference_wss_pa, reference_flow_uL_s

    def update_flow_low_pass_visibility():
        flow_low_pass_cutoff_input.setEnabled(flow_low_pass_checkbox.isChecked())

    def get_flow_low_pass_settings():
        if not flow_low_pass_checkbox.isChecked():
            return False, 0.1
        return (
            True,
            validate_float_range(
                flow_low_pass_cutoff_input.text(),
                "Flow low-pass cutoff frequency",
                0.0001,
                1000,
            ),
        )

    def get_plot_start_time(stop_time):
        start_time = validate_float_range(
            plot_start_time_input.text(), "Plot start time", 0, stop_time
        )
        if start_time >= stop_time:
            raise ValueError("Plot start time must be below the computation time.")
        return start_time

    def get_stl_wetting_cache_inputs():
        if not advanced_mode_checkbox.isChecked():
            return (
                default_stl_wetting_volume_bins,
                default_stl_wetting_tilt_step_deg,
                "cached",
            )
        return (
            validate_integer_range(
                stl_volume_bins_input.text(), "STL local-volume bins", 10, 2000
            ),
            validate_float_range(
                stl_tilt_step_input.text(), "STL tilt step", 0.01, 5
            ),
            stl_wetting_solver_input.currentData(),
        )

    def get_two_reservoir_stl_solver_input():
        return (
            two_reservoir_stl_solver_input.currentData()
            if advanced_mode_checkbox.isChecked()
            else "cached"
        )

    def get_geometry_port_driving_heads_input():
        return not (
            advanced_mode_checkbox.isChecked()
            and legacy_channel_driving_heads_checkbox.isChecked()
        )

    def get_capillary_force_input():
        if not advanced_mode_checkbox.isChecked():
            return default_cap_force
        return validate_float_range(
            cap_force_input.text(), "Capillary pressure", 0, 10
        )

    def get_wet_dry_gating_input():
        return (
            wet_dry_gating_checkbox.isChecked()
            if advanced_mode_checkbox.isChecked()
            else True
        )

    def get_top_contact_wetting_input():
        return (
            advanced_mode_checkbox.isChecked()
            and (
                single_reservoir_radio.isChecked()
                or two_reservoir_radio.isChecked()
                or nested_four_reservoir_radio.isChecked()
            )
            and (
                use_stl_reservoir_checkbox.isChecked()
                or use_outer_stl_reservoir_checkbox.isChecked()
            )
            and top_contact_wetting_checkbox.isChecked()
        )

    def get_selected_viscosity_pa_s():
        if not advanced_mode_checkbox.isChecked():
            return default_viscosity_mpa_s * 1e-3
        return (viscosity_slider.value() / 10.0) * 1e-3

    def apply_selected_viscosity():
        global viscosity
        viscosity = get_selected_viscosity_pa_s()
        viscosity_value_label.setText(f"{viscosity * 1e3:.1f} mPa*s")
        return viscosity

    def on_viscosity_slider_changed(_value):
        apply_selected_viscosity()
        if advanced_mode_checkbox.isChecked():
            clear_cached_plot(
                "Viscosity changed. Generate a fresh plot to apply it."
            )

    def update_advanced_mode_visibility():
        is_advanced_mode = advanced_mode_checkbox.isChecked()
        if is_advanced_mode:
            # Basic mode always uses cached water levels; keep that solver as
            # the explicit starting point when exposing advanced controls.
            stl_wetting_solver_input.setCurrentIndex(
                stl_wetting_solver_input.findData("cached")
            )
            two_reservoir_stl_solver_input.setCurrentIndex(
                two_reservoir_stl_solver_input.findData("cached")
            )
        cap_force_widget.setVisible(is_advanced_mode)
        viscosity_widget.setVisible(is_advanced_mode)
        apply_selected_viscosity()
        fourier_group.setVisible(is_advanced_mode)
        update_mode_visibility()
        update_reservoir_mode_visibility()

    def on_advanced_mode_changed(_checked):
        update_advanced_mode_visibility()
        clear_cached_plot(
            "Advanced-mode settings changed. Generate a fresh plot to apply them."
        )

    def get_channel_wss_geometry_from_inputs():
        return build_primary_geometry_from_inputs()

    def update_channel_wss_selection_hint(reset_to_full=False):
        channel_geometry = get_channel_wss_geometry_from_inputs()
        if channel_geometry.get("shape", "rectangular") == "circular":
            return
        wall_name = channel_wss_wall_input.currentText()
        wall_length = get_channel_wss_wall_length(channel_geometry, wall_name)
        channel_wss_hint_label.setText(
            f"Selected range is measured along the {wall_name.lower()} from 0 to {wall_length:.3f} mm."
        )
        if reset_to_full:
            channel_wss_start_input.setText("0.0")
            channel_wss_end_input.setText(f"{wall_length:.3f}")
            return

        try:
            start_value = float(channel_wss_start_input.text())
        except ValueError:
            start_value = 0.0
        try:
            end_value = float(channel_wss_end_input.text())
        except ValueError:
            end_value = wall_length

        start_value = max(0.0, min(start_value, wall_length))
        end_value = max(start_value, min(end_value, wall_length))
        channel_wss_start_input.setText(f"{start_value:.3f}")
        channel_wss_end_input.setText(f"{end_value:.3f}")

    def get_representative_channel_flow(plot_data):
        flow_mode = channel_wss_flow_mode_input.currentText()
        if flow_mode == "Maximum |Q1| over one period":
            representative_flow_uL_s, period_seconds, start_time, end_time = (
                calculate_last_period_max_absolute(
                    plot_data["time_points"],
                    plot_data["q_1"],
                    plot_data["selected_case"],
                )
            )
            return (
                representative_flow_uL_s,
                (
                    f"Last-period max |Q1| "
                    f"({start_time:.1f}-{end_time:.1f} s, T={period_seconds:.2f} s)"
                ),
            )

        representative_flow_uL_s, period_seconds, start_time, end_time = (
            calculate_last_period_mean_absolute(
                plot_data["time_points"],
                plot_data["q_1"],
                plot_data["selected_case"],
            )
        )
        return (
            representative_flow_uL_s,
            (
                f"Last-period mean |Q1| "
                f"({start_time:.1f}-{end_time:.1f} s, T={period_seconds:.2f} s)"
            ),
        )

    def redraw_cached_plot():
        if plot_state["is_updating"] or plot_state["data"] is None:
            return

        plot_state["is_updating"] = True
        try:
            reference_wss_pa, reference_flow_uL_s = get_reference_wss_values()
            flow_low_pass_enabled, flow_low_pass_cutoff_hz = get_flow_low_pass_settings()
            plot_start_time = get_plot_start_time(plot_state["data"]["stop_time"])
            if plot_state["figure"] is not None:
                plt.close(plot_state["figure"])

            figure, _ = create_figure_from_data(
                plot_state["data"],
                plot_tilts=plot_tilts_checkbox.isChecked(),
                plot_q1=plot_q1_checkbox.isChecked(),
                plot_q2=plot_q2_checkbox.isChecked(),
                plot_q3=plot_q3_checkbox.isChecked(),
                plot_q4=plot_q4_checkbox.isChecked(),
                plot_local_volumes=plot_local_volumes_checkbox.isChecked(),
                plot_midpoint_gravity_pressure=plot_midpoint_gravity_pressure_checkbox.isChecked(),
                plot_port_heads=plot_port_heads_checkbox.isChecked(),
                plot_inner_outer_inlet_hydrostatic_difference=plot_inner_outer_outlet_hydrostatic_checkbox.isChecked(),
                plot_inner_outer_outlet_hydrostatic_difference=plot_inner_outer_inlet_hydrostatic_checkbox.isChecked(),
                plot_inner_outer_raw_hydrostatic_pressures=plot_inner_outer_raw_pressure_checkbox.isChecked(),
                plot_wss=plot_wss_checkbox.isChecked(),
                plot_wss_reference=plot_wss_reference_checkbox.isChecked(),
                reference_wss_pa=reference_wss_pa,
                reference_flow_uL_s=reference_flow_uL_s,
                flow_low_pass_enabled=flow_low_pass_enabled,
                flow_low_pass_cutoff_hz=flow_low_pass_cutoff_hz,
                ring_global_direction_convention=(
                    ring_global_direction_checkbox.isChecked()
                ),
                plot_start_time=plot_start_time,
            )
            plot_state["figure"] = figure
            plot_state["fourier_fit"] = None
            figure.show()
            plt.pause(0.1)
            fourier_fit_button.setEnabled(True)
            export_fourier_button.setEnabled(False)
            fourier_info_label.setText(
                "The plot was updated. Fit a new Fourier series before exporting it."
            )
            status_label.setText("Updated displayed plot from cached simulation.")
        except ValueError as exc:
            status_label.setText(str(exc))
        finally:
            plot_state["is_updating"] = False

    def select_profile_ports(profile_template):
        profile_polygon = profile_template["profile_points"]
        top_view_polygon = profile_template["top_view_points"]

        cross_figure, cross_axis = plt.subplots()
        top_figure, top_axis = plt.subplots()

        closed_profile = np.vstack([profile_polygon, profile_polygon[0]])
        cross_axis.fill(profile_polygon[:, 0], profile_polygon[:, 1], alpha=0.25, color="tab:blue")
        cross_axis.plot(closed_profile[:, 0], closed_profile[:, 1], color="tab:blue")
        cross_axis.set_aspect("equal", adjustable="box")
        cross_axis.set_xlabel(f"{profile_template['profile_axes'][0]} [mm]")
        cross_axis.set_ylabel(f"{profile_template['profile_axes'][1]} height [mm]")
        cross_axis.set_title("Cross section: click the shared inlet/outlet height")
        cross_axis.grid(True, alpha=0.3)

        closed_top = np.vstack([top_view_polygon, top_view_polygon[0]])
        top_axis.fill(top_view_polygon[:, 0], top_view_polygon[:, 1], alpha=0.25, color="tab:green")
        top_axis.plot(closed_top[:, 0], closed_top[:, 1], color="tab:green")
        top_axis.set_aspect("equal", adjustable="box")
        top_axis.set_xlabel(f"{profile_template['top_view_axes'][0]} [mm]")
        top_axis.set_ylabel(f"{profile_template['top_view_axes'][1]} [mm]")
        top_axis.set_title("Top view: click inlet first, then outlet")
        top_axis.grid(True, alpha=0.3)

        cross_figure.tight_layout()
        top_figure.tight_layout()
        cross_figure.show()
        top_figure.show()
        set_figure_window_geometry(cross_figure, 80, 80, 700, 520)
        set_figure_window_geometry(top_figure, 820, 80, 700, 520)
        plt.pause(0.1)

        cross_click = cross_figure.ginput(1, timeout=-1)
        if len(cross_click) != 1:
            plt.close(cross_figure)
            plt.close(top_figure)
            raise ValueError("Height selection requires one click in the cross-section window.")

        shared_height = float(np.clip(cross_click[0][1], 0.0, np.max(profile_polygon[:, 1])))
        cross_axis.axhline(shared_height, color="tab:red", linestyle="--")
        cross_axis.set_title(
            f"Cross section: selected shared height = {shared_height:.3f} mm"
        )
        cross_figure.canvas.draw_idle()
        plt.pause(0.1)

        top_clicks = top_figure.ginput(2, timeout=-1)
        plt.close(cross_figure)
        plt.close(top_figure)
        if len(top_clicks) != 2:
            raise ValueError("Top view selection requires two clicks: inlet, then outlet.")

        top_points = [
            np.asarray(
                [
                    float(np.clip(point[0], 0.0, np.max(top_view_polygon[:, 0]))),
                    float(np.clip(point[1], 0.0, np.max(top_view_polygon[:, 1]))),
                ],
                dtype=float,
            )
            for point in top_clicks
        ]

        inlet_point, outlet_point = top_points

        inlet_world = build_world_port_coordinates(profile_template, inlet_point, shared_height)
        outlet_world = build_world_port_coordinates(profile_template, outlet_point, shared_height)

        return {
            **profile_template,
            "left_port": {"x": float(inlet_point[0]), "y": shared_height},
            "right_port": {"x": float(outlet_point[0]), "y": shared_height},
            "inlet_top_view": {"x": float(inlet_point[0]), "z": float(inlet_point[1])},
            "outlet_top_view": {"x": float(outlet_point[0]), "z": float(outlet_point[1])},
            "left_port_world": inlet_world,
            "right_port_world": outlet_world,
            "shared_port_height": shared_height,
        }

    def format_profile_info(profile_template):
        shell_points = np.asarray(profile_template["shell_triangles"], dtype=float).reshape(-1, 3)
        bounds = {
            "length": float(np.max(shell_points[:, 0]) - np.min(shell_points[:, 0])),
            "width": float(np.max(shell_points[:, 1]) - np.min(shell_points[:, 1])),
            "height": float(np.max(shell_points[:, 2]) - np.min(shell_points[:, 2])),
        }
        left_port = profile_template["left_port"]
        right_port = profile_template["right_port"]
        xy_rotation_degrees = float(profile_template.get("xy_rotation_degrees", 0.0))
        return (
            f"Loaded {Path(profile_template['source_path']).name} "
            f"(extrusion axis: {profile_template['extrusion_axis_name']}, "
            f"profile axes: {profile_template['profile_axes'][0]}/{profile_template['profile_axes'][1]}).\n"
            f"STL X/Y rotation: {xy_rotation_degrees:.1f} deg.\n"
            f"Width: {bounds['width']:.3f} mm, "
            f"Length: {bounds['length']:.3f} mm, Height used: {bounds['height']:.3f} mm.\n"
            f"Shared port height: {profile_template['shared_port_height']:.3f} mm.\n"
            f"Inlet (x,z,y): {format_xyz_coordinates(profile_template['left_port_world'])}; "
            f"Outlet (x,z,y): {format_xyz_coordinates(profile_template['right_port_world'])}.\n"
            f"Solver ports (length,height): "
            f"inlet ({left_port['x']:.3f}, {left_port['y']:.3f}) mm, "
            f"outlet ({right_port['x']:.3f}, {right_port['y']:.3f}) mm."
        )

    def on_load_stl_profile():
        stl_path, _ = QFileDialog.getOpenFileName(
            window,
            "Open reservoir STL",
            str(Path(__file__).resolve().parent),
            "STL mesh (*.stl)",
        )
        if not stl_path:
            status_label.setText("STL load was cancelled.")
            return

        try:
            profile_template = load_profile_from_stl(stl_path)
            xy_rotation_degrees = validate_float_range(
                stl_xy_rotation_input.text(), "STL X/Y rotation", -180, 180
            )
            profile_template = rotate_stl_profile_template_in_xy_plane(
                profile_template,
                xy_rotation_degrees,
            )
            profile_template = select_profile_ports(profile_template)
        except Exception as exc:
            QMessageBox.critical(window, "Reservoir import failed", str(exc))
            return

        reservoir_profile_state["template"] = profile_template
        reservoir_profile_info.setText(format_profile_info(profile_template))
        try:
            current_fill = validate_float_range(reservoir_fill_input.text(), "V_fill", 0.1, 10)
        except ValueError:
            current_fill = default_reservoir_settings["v_fill_ml"]
        imported_geometry = build_profile_reservoir_geometry(
            profile_template["profile_points"],
            profile_template["width"],
            current_fill,
            [
                profile_template["left_port"]["x"],
                profile_template["left_port"]["y"],
            ],
            [
                profile_template["right_port"]["x"],
                profile_template["right_port"]["y"],
            ],
            source_path=profile_template["source_path"],
            extrusion_axis=profile_template["extrusion_axis_name"],
            profile_axes=profile_template["profile_axes"],
            top_view_points=profile_template["top_view_points"],
            top_view_triangles=profile_template["top_view_triangles"],
            shell_triangles=profile_template["shell_triangles"],
            left_port_world=profile_template["left_port_world"],
            right_port_world=profile_template["right_port_world"],
            axis_mins=profile_template["axis_mins"],
            axis_maxs=profile_template["axis_maxs"],
            axis_names=stl_axis_names,
        )
        status_label.setText(
            "STL reservoir loaded and ports selected. "
            f"Inlet (x,z,y): {format_xyz_coordinates(profile_template['left_port_world'])}. "
            f"Outlet (x,z,y): {format_xyz_coordinates(profile_template['right_port_world'])}. "
            f"{describe_reservoir_geometry(imported_geometry)}"
        )
        # Importing changes the availability of the STL-only solver controls.
        # Refresh them immediately instead of waiting for a model-mode change.
        update_reservoir_mode_visibility()
        update_mode_visibility()

    def on_load_outer_stl_profile():
        if use_stl_reservoir_checkbox.isChecked() and reservoir_profile_state["template"] is None:
            QMessageBox.information(
                window,
                "Load inner reservoir first",
                "Load and configure the inner STL reservoir before importing the outer STL.",
            )
            return
        stl_path, _ = QFileDialog.getOpenFileName(
            window,
            "Open outer reservoir STL",
            str(Path(__file__).resolve().parent),
            "STL mesh (*.stl)",
        )
        if not stl_path:
            status_label.setText("Outer STL load was cancelled.")
            return
        try:
            profile_template = load_profile_from_stl(stl_path)
            xy_rotation_degrees = validate_float_range(
                outer_stl_xy_rotation_input.text(), "Outer STL X/Y rotation", -180, 180
            )
            profile_template = rotate_stl_profile_template_in_xy_plane(
                profile_template, xy_rotation_degrees
            )
            profile_template = select_profile_ports(profile_template)
        except Exception as exc:
            QMessageBox.critical(window, "Outer reservoir import failed", str(exc))
            return

        outer_reservoir_profile_state["template"] = profile_template
        outer_reservoir_profile_info.setText(format_profile_info(profile_template))
        status_label.setText(
            "Outer STL reservoir loaded and ports selected. "
            f"Inlet (x,z,y): {format_xyz_coordinates(profile_template['left_port_world'])}. "
            f"Outlet (x,z,y): {format_xyz_coordinates(profile_template['right_port_world'])}."
        )
        update_reservoir_mode_visibility()
        clear_cached_plot("Outer reservoir geometry changed. Generate a fresh plot to apply it.")

    def on_save_image():
        if plot_state["figure"] is None:
            QMessageBox.information(window, "No plot", "Generate a plot before saving an image.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            window,
            "Save plot as",
            str(Path(__file__).with_name("assync-tilt.png")),
            "PNG image (*.png)",
        )
        if not save_path:
            status_label.setText("Image save was cancelled.")
            return

        plot_state["figure"].savefig(save_path, bbox_inches="tight")
        status_label.setText(f"Saved latest plot to {Path(save_path).resolve()}")

    def on_save_plot_animation():
        if plot_state["data"] is None:
            QMessageBox.information(
                window,
                "No plot data",
                "Generate a plot before saving an animated plot.",
            )
            return

        try:
            animation_frame_count = validate_integer_range(
                animation_frame_count_input.text(), "Animation frames", 24, 10000
            )
            reference_wss_pa, reference_flow_uL_s = get_reference_wss_values()
            flow_low_pass_enabled, flow_low_pass_cutoff_hz = get_flow_low_pass_settings()
        except ValueError as exc:
            QMessageBox.critical(window, "Invalid animation settings", str(exc))
            return

        output_path, _ = QFileDialog.getSaveFileName(
            window,
            "Save plot animation as",
            str(Path(__file__).with_name("flow-plot-animation.gif")),
            "GIF animation (*.gif)",
        )
        if not output_path:
            status_label.setText("Plot-animation export was cancelled.")
            return

        plot_animation_button.setEnabled(False)
        progress_bar.setRange(0, 0)
        progress_label.setText("Generating plot animation...")
        status_label.setText(
            f"Generating a {animation_frame_count}-frame 24 fps plot animation "
            "for the final full actuation period."
        )
        app.processEvents()
        try:
            saved_animation_path = save_plot_animation(
                plot_state["data"],
                output_path,
                plot_tilts=plot_tilts_checkbox.isChecked(),
                plot_q1=plot_q1_checkbox.isChecked(),
                plot_q2=plot_q2_checkbox.isChecked(),
                plot_q3=plot_q3_checkbox.isChecked(),
                plot_q4=plot_q4_checkbox.isChecked(),
                plot_local_volumes=plot_local_volumes_checkbox.isChecked(),
                plot_midpoint_gravity_pressure=(
                    plot_midpoint_gravity_pressure_checkbox.isChecked()
                ),
                plot_port_heads=plot_port_heads_checkbox.isChecked(),
                plot_inner_outer_inlet_hydrostatic_difference=plot_inner_outer_outlet_hydrostatic_checkbox.isChecked(),
                plot_inner_outer_outlet_hydrostatic_difference=plot_inner_outer_inlet_hydrostatic_checkbox.isChecked(),
                plot_inner_outer_raw_hydrostatic_pressures=plot_inner_outer_raw_pressure_checkbox.isChecked(),
                plot_wss=plot_wss_checkbox.isChecked(),
                plot_wss_reference=plot_wss_reference_checkbox.isChecked(),
                reference_wss_pa=reference_wss_pa,
                reference_flow_uL_s=reference_flow_uL_s,
                flow_low_pass_enabled=flow_low_pass_enabled,
                flow_low_pass_cutoff_hz=flow_low_pass_cutoff_hz,
                ring_global_direction_convention=(
                    ring_global_direction_checkbox.isChecked()
                ),
                frame_count=animation_frame_count,
            )
            status_label.setText(f"Saved plot animation to {saved_animation_path}")
            progress_bar.setRange(0, 100)
            progress_bar.setValue(100)
            progress_label.setText("Plot animation saved")
        except Exception as exc:
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_label.setText("Idle")
            QMessageBox.critical(window, "Plot animation failed", str(exc))
        finally:
            plot_animation_button.setEnabled(plot_state["data"] is not None)

    def build_current_reservoir_geometry(v_fill_ml):
        if simple_tilting_radio.isChecked():
            return build_cylindrical_reservoir_geometry(
                validate_float_range(cylinder_radius_input.text(), "Cylinder radius", 1, 10),
                validate_float_range(cylinder_height_input.text(), "Cylinder height", 1, 10),
                v_fill_ml,
            )
        if use_stl_reservoir_checkbox.isChecked():
            if reservoir_profile_state["template"] is None:
                raise ValueError("Load an STL reservoir and pick the ports before continuing.")
            profile_template = reservoir_profile_state["template"]
            return build_profile_reservoir_geometry(
                profile_template["profile_points"],
                profile_template["width"],
                v_fill_ml,
                [
                    profile_template["left_port"]["x"],
                    profile_template["left_port"]["y"],
                ],
                [
                    profile_template["right_port"]["x"],
                    profile_template["right_port"]["y"],
                ],
                source_path=profile_template["source_path"],
                extrusion_axis=profile_template["extrusion_axis_name"],
                profile_axes=profile_template["profile_axes"],
                top_view_points=profile_template["top_view_points"],
                top_view_triangles=profile_template["top_view_triangles"],
                shell_triangles=profile_template["shell_triangles"],
                left_port_world=profile_template["left_port_world"],
                right_port_world=profile_template["right_port_world"],
                axis_mins=profile_template["axis_mins"],
                axis_maxs=profile_template["axis_maxs"],
                axis_names=stl_axis_names,
            )
        return build_reservoir_geometry(
            validate_float_range(reservoir_width_input.text(), "Reservoir width", 1, 9),
            validate_float_range(reservoir_height_input.text(), "Reservoir height", 1, 10),
            validate_float_range(reservoir_length_input.text(), "Reservoir length", 1, 100),
            v_fill_ml,
        )

    def build_outer_reservoir_geometry(v_fill_ml):
        if use_outer_stl_reservoir_checkbox.isChecked():
            if outer_reservoir_profile_state["template"] is None:
                raise ValueError(
                    "Load an outer STL reservoir and pick its ports before continuing."
                )
            profile_template = outer_reservoir_profile_state["template"]
            return build_profile_reservoir_geometry(
                profile_template["profile_points"],
                profile_template["width"],
                v_fill_ml,
                [profile_template["left_port"]["x"], profile_template["left_port"]["y"]],
                [profile_template["right_port"]["x"], profile_template["right_port"]["y"]],
                source_path=profile_template["source_path"],
                extrusion_axis=profile_template["extrusion_axis_name"],
                profile_axes=profile_template["profile_axes"],
                top_view_points=profile_template["top_view_points"],
                top_view_triangles=profile_template["top_view_triangles"],
                shell_triangles=profile_template["shell_triangles"],
                left_port_world=profile_template["left_port_world"],
                right_port_world=profile_template["right_port_world"],
                axis_mins=profile_template["axis_mins"],
                axis_maxs=profile_template["axis_maxs"],
                axis_names=stl_axis_names,
            )
        return build_reservoir_geometry(
            validate_float_range(outer_reservoir_width_input.text(), "Outer reservoir width", 1, 20),
            validate_float_range(outer_reservoir_height_input.text(), "Outer reservoir height", 1, 20),
            validate_float_range(outer_reservoir_length_input.text(), "Outer reservoir length", 1, 100),
            v_fill_ml,
        )

    def on_save_animation():
        try:
            apply_selected_viscosity()
            stop_time = validate_stop_time(stop_time_input.text())
            time_step = validate_float_range(time_step_input.text(), "Time step", 0.0001, 1)
            tilt = validate_float_range(tilt_input.text(), "Tilt", 1, 45)
            speed1 = validate_float_range(speed1_input.text(), "Speed 1", 0.1, 10)
            animation_frame_count = validate_integer_range(
                animation_frame_count_input.text(), "Animation frames", 24, 10000
            )
            cap_force = get_capillary_force_input()
            v_fill_ml = validate_float_range(reservoir_fill_input.text(), "V_fill", 0.1, 10)
            stl_wetting_volume_bins, stl_wetting_tilt_step_deg, stl_wetting_solver = (
                get_stl_wetting_cache_inputs()
            )
            two_reservoir_stl_solver = get_two_reservoir_stl_solver_input()
            reservoir_geometry = build_current_reservoir_geometry(v_fill_ml)
            if nested_four_reservoir_radio.isChecked():
                mode = "nested_four_reservoir_loops"
            elif simple_tilting_radio.isChecked():
                mode = (
                    "four_reservoir_tilting"
                    if four_reservoir_checkbox.isChecked()
                    else "simple_tilting"
                )
            elif single_reservoir_radio.isChecked():
                mode = (
                    "single_reservoir_assymmetric"
                    if assymmetric_tilting_checkbox.isChecked()
                    else "single_reservoir"
                )
            else:
                mode = "assymmetric" if assymmetric_tilting_checkbox.isChecked() else "symmetric"
            speed2 = None
            slow_start = None
            slow_duration = None
            secondary_geometry = None
            outer_reservoir_geometry = None
            outer_channel_geometry = None
            if mode == "nested_four_reservoir_loops":
                outer_fill_ml = validate_float_range(
                    outer_reservoir_fill_input.text(), "Outer-loop V_fill", 0.1, 10
                )
                outer_reservoir_geometry = build_outer_reservoir_geometry(outer_fill_ml)
                outer_channel_geometry = build_outer_channel_geometry_from_inputs()
            if mode in ("assymmetric", "single_reservoir_assymmetric"):
                speed2 = validate_float_range(speed2_input.text(), "Speed 2", 0.1, 10)
                slow_start = validate_float_range(
                    slow_start_input.text(), "Slow traversal start", 0, 0.9
                )
                slow_duration = validate_float_range(
                    slow_duration_input.text(), "Slow traversal duration", 0.05, 0.95
                )
                if slow_start + slow_duration > 1:
                    raise ValueError(
                        "Slow traversal start plus duration must not exceed 1."
                    )
            if mode == "assymmetric":
                secondary_geometry = build_secondary_geometry_from_inputs()
            plot_data = compute_plot_data(
                stop_time,
                mode=mode,
                tilt=tilt,
                speed1=speed1,
                time_step=time_step,
                reservoir_geometry=reservoir_geometry,
                primary_geometry=build_primary_geometry_from_inputs(),
                cap_force=cap_force,
                speed2=speed2,
                slow_start=slow_start,
                slow_duration=slow_duration,
                secondary_geometry=secondary_geometry,
                outer_reservoir_geometry=outer_reservoir_geometry,
                outer_channel_geometry=outer_channel_geometry,
                use_wet_dry_gating=get_wet_dry_gating_input(),
                stl_wetting_solver=stl_wetting_solver,
                two_reservoir_stl_solver=two_reservoir_stl_solver,
                stl_wetting_volume_bins=stl_wetting_volume_bins,
                stl_wetting_tilt_step_deg=stl_wetting_tilt_step_deg,
                top_contact_wetting=get_top_contact_wetting_input(),
            )
        except ValueError as exc:
            QMessageBox.critical(window, "Invalid input", str(exc))
            return

        output_path, _ = QFileDialog.getSaveFileName(
            window,
            "Save 3D animation as",
            str(Path(__file__).with_name("reservoir-liquid-animation.gif")),
            "GIF animation (*.gif)",
        )
        if not output_path:
            status_label.setText("Animation export was cancelled.")
            return

        animation_button.setEnabled(False)
        progress_bar.setRange(0, 0)
        progress_label.setText("Generating 3D animation...")
        animation_scope = (
            "both reservoirs and their connecting channels"
            if mode in ("symmetric", "assymmetric", "simple_tilting")
            else "four cylindrical reservoirs and their connecting channels"
            if mode == "four_reservoir_tilting"
            else "inner and outer recirculating loops"
            if mode == "nested_four_reservoir_loops"
            else "the reservoir"
        )
        status_label.setText(
            f"Generating a {animation_frame_count}-frame 24 fps 3D animation "
            f"for the final full actuation period ({animation_scope})."
        )
        app.processEvents()
        try:
            saved_animation_path = save_reservoir_animation(
                reservoir_geometry,
                plot_data,
                output_path,
                frame_count=animation_frame_count,
            )
            status_label.setText(f"Saved 3D animation to {saved_animation_path}")
            progress_bar.setRange(0, 100)
            progress_bar.setValue(100)
            progress_label.setText("Animation saved")
        except Exception as exc:
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_label.setText("Idle")
            QMessageBox.critical(window, "Animation failed", str(exc))
        finally:
            animation_button.setEnabled(True)

    def on_export_data():
        if plot_state["data"] is None:
            QMessageBox.information(window, "No data", "Generate a plot before exporting data.")
            return

        export_path, _ = QFileDialog.getSaveFileName(
            window,
            "Export data as",
            str(Path(__file__).with_name("plot-data.csv")),
            "CSV file (*.csv)",
        )
        if not export_path:
            status_label.setText("Data export was cancelled.")
            return

        reference_wss_pa, reference_flow_uL_s = get_reference_wss_values()
        columns = build_export_columns(
            plot_state["data"],
            plot_q1=plot_q1_checkbox.isChecked(),
            plot_q2=plot_q2_checkbox.isChecked(),
            plot_q3=plot_q3_checkbox.isChecked(),
            plot_q4=plot_q4_checkbox.isChecked(),
            plot_local_volumes=plot_local_volumes_checkbox.isChecked(),
            plot_midpoint_gravity_pressure=plot_midpoint_gravity_pressure_checkbox.isChecked(),
            plot_port_heads=plot_port_heads_checkbox.isChecked(),
            plot_inner_outer_inlet_hydrostatic_difference=plot_inner_outer_outlet_hydrostatic_checkbox.isChecked(),
            plot_inner_outer_outlet_hydrostatic_difference=plot_inner_outer_inlet_hydrostatic_checkbox.isChecked(),
            plot_inner_outer_raw_hydrostatic_pressures=plot_inner_outer_raw_pressure_checkbox.isChecked(),
            plot_wss=plot_wss_checkbox.isChecked(),
            plot_wss_reference=plot_wss_reference_checkbox.isChecked(),
            reference_wss_pa=reference_wss_pa,
            reference_flow_uL_s=reference_flow_uL_s,
        )
        headers = [header for header, _ in columns]
        series = [values for _, values in columns]

        with open(export_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            for row in zip(*series):
                writer.writerow(row)

        status_label.setText(f"Exported plot data to {Path(export_path).resolve()}")

    def on_plot_channel_wss():
        if plot_state["data"] is None:
            QMessageBox.information(
                window,
                "No flow data",
                "Generate the flow-rate plot before opening the channel WSS view.",
            )
            return

        try:
            primary_geometry = plot_state["data"]["selected_case"]["geometry1"]
            representative_flow_uL_s, representative_flow_basis = get_representative_channel_flow(
                plot_state["data"]
            )
            if primary_geometry.get("shape", "rectangular") == "circular":
                selection = None
                result = calculate_circular_channel_wss(
                    primary_geometry,
                    representative_flow_uL_s,
                    representative_flow_basis,
                )
                clear_channel_wss_state()
                channel_wss_state["result"] = result
                channel_wss_state["selection"] = None
                channel_wss_state["selected_average_wss_pa"] = result.wall_shear_stress_pa
                channel_wss_value_label.setText(
                    f"WSS: {result.wall_shear_stress_pa:.5g} Pa"
                )
                status_label.setText(
                    "Circular-pipe WSS calculated from the selected representative |Q1|."
                )
                return
            else:
                selection = build_channel_wss_selection(
                    primary_geometry,
                    channel_wss_wall_input.currentText(),
                    channel_wss_start_input.text(),
                    channel_wss_end_input.text(),
                )
                result = calculate_channel_wss_profile(
                    primary_geometry,
                    representative_flow_uL_s,
                    representative_flow_basis,
                )
                figure, selected_average_wss_pa = create_channel_wss_figure(
                    result,
                    selection,
                )
            clear_channel_wss_state()
            channel_wss_state["figure"] = figure
            channel_wss_state["result"] = result
            channel_wss_state["selection"] = selection
            channel_wss_state["selected_average_wss_pa"] = selected_average_wss_pa
            figure.show()
            plt.pause(0.1)
            save_channel_wss_image_button.setEnabled(True)
            export_channel_wss_data_button.setEnabled(True)
            status_label.setText(
                "Rectangular-channel WSS figure displayed using the primary channel geometry."
            )
        except ValueError as exc:
            QMessageBox.critical(window, "Invalid channel WSS input", str(exc))
        except Exception as exc:
            QMessageBox.critical(window, "Channel WSS plotting failed", str(exc))

    def on_save_channel_wss_image():
        if channel_wss_state["figure"] is None:
            QMessageBox.information(
                window,
                "No channel WSS plot",
                "Plot the channel WSS before saving an image.",
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            window,
            "Save channel WSS plot as",
            str(Path(__file__).with_name("channel-wss.png")),
            "PNG image (*.png)",
        )
        if not save_path:
            status_label.setText("Channel WSS image save was cancelled.")
            return

        channel_wss_state["figure"].savefig(save_path, bbox_inches="tight", dpi=300)
        status_label.setText(
            f"Saved channel WSS image to {Path(save_path).resolve()}"
        )

    def on_export_channel_wss_data():
        if channel_wss_state["result"] is None:
            QMessageBox.information(
                window,
                "No channel WSS data",
                "Plot the channel WSS before exporting CSV data.",
            )
            return

        export_path, _ = QFileDialog.getSaveFileName(
            window,
            "Export channel WSS data as",
            str(Path(__file__).with_name("channel-wss.csv")),
            "CSV file (*.csv)",
        )
        if not export_path:
            status_label.setText("Channel WSS data export was cancelled.")
            return

        csv_path = export_channel_wss_csv(
            export_path,
            channel_wss_state["result"],
            channel_wss_state["selection"],
            channel_wss_state["selected_average_wss_pa"],
        )
        status_label.setText(
            f"Exported channel WSS data to {Path(csv_path).resolve()}"
        )

    def on_fit_fourier_series():
        if plot_state["data"] is None or plot_state["figure"] is None:
            QMessageBox.information(
                window,
                "No flow data",
                "Generate a flow-rate plot before fitting a Fourier series.",
            )
            return

        try:
            term_count = int(fourier_terms_input.text())
            if not 1 <= term_count <= 50:
                raise ValueError("Fourier terms must be an integer between 1 and 50.")
            series_key = fourier_curve_input.currentData()
            series_names = {
                "q_1": "Q1",
                "q_2": "Q2",
                "q_3": "Q3",
                "q_4": "Q4",
            }
            use_ring_global_direction = (
                plot_state["data"]["selected_case"]["mode"]
                == "four_reservoir_tilting"
                and ring_global_direction_checkbox.isChecked()
            )
            if use_ring_global_direction and series_key in ("q_3", "q_4"):
                direction = "+x" if series_key == "q_3" else "+y"
                series_names[series_key] = f"-{series_names[series_key]} ({direction})"
            target_axis = next(
                (
                    axis
                    for axis in plot_state["figure"].axes
                    if (
                        getattr(axis, "_simulation_series_key", None) == series_key
                        or series_key in getattr(axis, "_simulation_series_keys", set())
                    )
                ),
                None,
            )
            if target_axis is None:
                raise ValueError(
                    f"Enable plotting for {series_names[series_key]} before fitting it."
                )

            fit_data = build_fourier_period_fit(
                plot_state["data"]["time_points"],
                (
                    -plot_state["data"][series_key]
                    if use_ring_global_direction and series_key in ("q_3", "q_4")
                    else plot_state["data"][series_key]
                ),
                plot_state["data"]["selected_case"],
                term_count,
            )
            for line in list(target_axis.lines):
                if line.get_gid() == "fourier-series-fit":
                    line.remove()
            target_axis.plot(
                fit_data["time_points"],
                fit_data["fitted_values"],
                color="#ff7f0e",
                linewidth=2.0,
                label=(
                    f"Fourier fit ({term_count} terms, "
                    f"T={fit_data['period_seconds']:.2f} s)"
                ),
                gid="fourier-series-fit",
            )
            # Keep the fit label outside the data area, consistent with the main plots.
            handles, labels = target_axis.get_legend_handles_labels()
            target_axis.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2,
                borderaxespad=0.0,
            )
            plot_state["fourier_fit"] = {
                **fit_data,
                "series_key": series_key,
                "series_name": series_names[series_key],
                "term_count": term_count,
            }
            export_fourier_button.setEnabled(True)
            plot_state["figure"].canvas.draw_idle()
            fourier_info_label.setText(
                f"Fitted {series_names[series_key]} with {term_count} terms over "
                f"T={fit_data['period_seconds']:.2f} s."
            )
            status_label.setText(
                f"Added {term_count}-term Fourier fit for {series_names[series_key]} over "
                f"{fit_data['start_time']:.2f}-{fit_data['end_time']:.2f} s "
                f"(T={fit_data['period_seconds']:.2f} s)."
            )
        except (TypeError, ValueError) as exc:
            QMessageBox.critical(window, "Fourier fit failed", str(exc))

    def on_export_fourier_series():
        fit_state = plot_state["fourier_fit"]
        if fit_state is None:
            QMessageBox.information(
                window,
                "No Fourier fit",
                "Fit a Fourier series before exporting it.",
            )
            return

        export_path, _ = QFileDialog.getSaveFileName(
            window,
            "Export Fourier series as",
            str(Path(__file__).with_name("fourier-series.txt")),
            "Text file (*.txt)",
        )
        if not export_path:
            status_label.setText("Fourier-series export was cancelled.")
            return

        text_path = Path(export_path).with_suffix(".txt")
        text_contents = "\n".join(
            (
                "Fourier series expansion",
                f"Curve: {fit_state['series_name']}",
                f"Terms: {fit_state['term_count']}",
                f"Fitting window: {fit_state['start_time']:.4f}-{fit_state['end_time']:.4f} s",
                f"Period P: {fit_state['period_seconds']:.4f} s",
                "Time variable: t is measured from the start of the fitting window.",
                "",
                "f(t) =",
                format_fourier_series_text(
                    fit_state["coefficients"],
                    fit_state["period_seconds"],
                ),
                "",
            )
        )
        text_path.write_text(text_contents, encoding="utf-8")
        fourier_info_label.setText(f"Exported Fourier series to {text_path.resolve()}")
        status_label.setText(f"Exported Fourier series to {text_path.resolve()}")

    def on_plot():
        plot_button.setEnabled(False)
        clear_channel_wss_state()
        channel_wss_button.setEnabled(False)
        try:
            apply_selected_viscosity()
            stop_time = validate_stop_time(stop_time_input.text())
            time_step = validate_float_range(time_step_input.text(), "Time step", 0.0001, 1)
            tilt = validate_float_range(tilt_input.text(), "Tilt", 1, 45)
            speed1 = validate_float_range(speed1_input.text(), "Speed 1", 0.1, 10)
            cap_force = get_capillary_force_input()
            v_fill_ml = validate_float_range(reservoir_fill_input.text(), "V_fill", 0.1, 10)
            stl_wetting_volume_bins, stl_wetting_tilt_step_deg, stl_wetting_solver = (
                get_stl_wetting_cache_inputs()
            )
            two_reservoir_stl_solver = get_two_reservoir_stl_solver_input()
            reservoir_geometry = build_current_reservoir_geometry(v_fill_ml)
            if use_stl_reservoir_checkbox.isChecked():
                status_label.setText(
                    "Using STL reservoir. "
                    f"Inlet (x,z,y): {format_xyz_coordinates(reservoir_geometry['left_port_world'])}. "
                    f"Outlet (x,z,y): {format_xyz_coordinates(reservoir_geometry['right_port_world'])}. "
                    f"{describe_reservoir_geometry(reservoir_geometry)}"
                )
                app.processEvents()
            primary_geometry = build_primary_geometry_from_inputs()
            if nested_four_reservoir_radio.isChecked():
                mode = "nested_four_reservoir_loops"
            elif simple_tilting_radio.isChecked():
                mode = (
                    "four_reservoir_tilting"
                    if four_reservoir_checkbox.isChecked()
                    else "simple_tilting"
                )
            elif single_reservoir_radio.isChecked():
                mode = (
                    "single_reservoir_assymmetric"
                    if assymmetric_tilting_checkbox.isChecked()
                    else "single_reservoir"
                )
            else:
                mode = "assymmetric" if assymmetric_tilting_checkbox.isChecked() else "symmetric"
            speed2 = None
            slow_start = None
            slow_duration = None
            secondary_geometry = None
            outer_reservoir_geometry = None
            outer_channel_geometry = None
            if mode == "nested_four_reservoir_loops":
                outer_fill_ml = validate_float_range(
                    outer_reservoir_fill_input.text(), "Outer-loop V_fill", 0.1, 10
                )
                outer_reservoir_geometry = build_outer_reservoir_geometry(outer_fill_ml)
                outer_channel_geometry = build_outer_channel_geometry_from_inputs()
            if mode in ("assymmetric", "single_reservoir_assymmetric"):
                speed2 = validate_float_range(speed2_input.text(), "Speed 2", 0.1, 10)
                slow_start = validate_float_range(
                    slow_start_input.text(), "Slow traversal start", 0, 0.9
                )
                slow_duration = validate_float_range(
                    slow_duration_input.text(), "Slow traversal duration", 0.05, 0.95
                )
                if slow_start + slow_duration > 1:
                    raise ValueError(
                        "Slow traversal start plus duration must not exceed 1."
                    )
            if mode == "assymmetric":
                secondary_geometry = build_secondary_geometry_from_inputs()
            progress_bar.setValue(0)
            progress_label.setText("Starting...")
            plot_data = compute_plot_data(
                stop_time,
                mode=mode,
                tilt=tilt,
                speed1=speed1,
                time_step=time_step,
                reservoir_geometry=reservoir_geometry,
                primary_geometry=primary_geometry,
                cap_force=cap_force,
                speed2=speed2,
                slow_start=slow_start,
                slow_duration=slow_duration,
                secondary_geometry=secondary_geometry,
                outer_reservoir_geometry=outer_reservoir_geometry,
                outer_channel_geometry=outer_channel_geometry,
                use_wet_dry_gating=get_wet_dry_gating_input(),
                stl_wetting_solver=stl_wetting_solver,
                two_reservoir_stl_solver=two_reservoir_stl_solver,
                stl_wetting_volume_bins=stl_wetting_volume_bins,
                stl_wetting_tilt_step_deg=stl_wetting_tilt_step_deg,
                top_contact_wetting=get_top_contact_wetting_input(),
                use_geometry_port_driving_heads=get_geometry_port_driving_heads_input(),
                progress_callback=update_progress,
            )
        except ValueError as exc:
            progress_bar.setValue(0)
            progress_label.setText("Idle")
            plot_button.setEnabled(True)
            QMessageBox.critical(window, "Invalid input", str(exc))
            return
        except Exception as exc:
            progress_bar.setValue(0)
            progress_label.setText("Idle")
            plot_button.setEnabled(True)
            QMessageBox.critical(window, "Plotting failed", str(exc))
            return

        try:
            plot_state["data"] = plot_data
            plot_state["fourier_fit"] = None
            plot_state["layout_inputs"] = {
                "mode": mode,
                "reservoir_geometry": reservoir_geometry,
                "primary_geometry": primary_geometry,
                "secondary_geometry": secondary_geometry,
                "outer_reservoir_geometry": outer_reservoir_geometry,
                "outer_channel_geometry": outer_channel_geometry,
            }
            export_fourier_button.setEnabled(False)
            if plot_state["figure"] is not None:
                plt.close(plot_state["figure"])

            reference_wss_pa, reference_flow_uL_s = get_reference_wss_values()
            flow_low_pass_enabled, flow_low_pass_cutoff_hz = get_flow_low_pass_settings()
            plot_start_time = get_plot_start_time(plot_state["data"]["stop_time"])
            figure, _ = create_figure_from_data(
                plot_state["data"],
                plot_tilts=plot_tilts_checkbox.isChecked(),
                plot_q1=plot_q1_checkbox.isChecked(),
                plot_q2=plot_q2_checkbox.isChecked(),
                plot_q3=plot_q3_checkbox.isChecked(),
                plot_q4=plot_q4_checkbox.isChecked(),
                plot_local_volumes=plot_local_volumes_checkbox.isChecked(),
                plot_midpoint_gravity_pressure=plot_midpoint_gravity_pressure_checkbox.isChecked(),
                plot_port_heads=plot_port_heads_checkbox.isChecked(),
                plot_inner_outer_inlet_hydrostatic_difference=plot_inner_outer_outlet_hydrostatic_checkbox.isChecked(),
                plot_inner_outer_outlet_hydrostatic_difference=plot_inner_outer_inlet_hydrostatic_checkbox.isChecked(),
                plot_inner_outer_raw_hydrostatic_pressures=plot_inner_outer_raw_pressure_checkbox.isChecked(),
                plot_wss=plot_wss_checkbox.isChecked(),
                plot_wss_reference=plot_wss_reference_checkbox.isChecked(),
                reference_wss_pa=reference_wss_pa,
                reference_flow_uL_s=reference_flow_uL_s,
                flow_low_pass_enabled=flow_low_pass_enabled,
                flow_low_pass_cutoff_hz=flow_low_pass_cutoff_hz,
                ring_global_direction_convention=(
                    ring_global_direction_checkbox.isChecked()
                ),
                plot_start_time=plot_start_time,
            )
            plot_state["figure"] = figure
            figure.show()
            plt.pause(0.1)
            channel_wss_button.setEnabled(True)
            fourier_fit_button.setEnabled(True)
            fourier_info_label.setText(
                "Flow data is ready. Select a curve and number of terms, then plot its Fourier series."
            )
            plot_animation_button.setEnabled(True)
            geometry_layout_button.setEnabled(True)
            save_image_button.setEnabled(True)
            export_data_button.setEnabled(True)
            circuit_reference = plot_data["selected_case"].get("circuit_reference_mm")
            if circuit_reference is None:
                circuit_centre_label.setText("Nested-circuit centre [mm]: not applicable to this model.")
                status_label.setText("Plot displayed. Use Save Image or Export Data if needed.")
            else:
                centre_x, centre_y, centre_z = circuit_reference
                circuit_centre_label.setText(
                    "Nested-circuit centre [mm] (x, y, z): "
                    f"({centre_x:.3f}, {centre_y:.3f}, {centre_z:.3f})"
                )
                status_label.setText(
                    "Plot displayed. The nested-circuit centre is shown above."
                )
            progress_bar.setValue(100)
            progress_label.setText("Done")
        except Exception as exc:
            progress_bar.setValue(0)
            progress_label.setText("Idle")
            QMessageBox.critical(window, "Saving failed", str(exc))
        finally:
            plot_button.setEnabled(True)

    def on_show_geometry_layout():
        layout_inputs = plot_state["layout_inputs"]
        if layout_inputs is None:
            QMessageBox.information(
                window,
                "No model geometry",
                "Generate a plot before opening the geometry layout.",
            )
            return
        try:
            plot_state["layout_figure"] = show_model_geometry_layout(**layout_inputs)
            status_label.setText("Displayed the scaled model geometry layout.")
        except Exception as exc:
            QMessageBox.critical(window, "Geometry layout failed", str(exc))

    plot_button.clicked.connect(on_plot)
    geometry_layout_button.clicked.connect(on_show_geometry_layout)
    fourier_fit_button.clicked.connect(on_fit_fourier_series)
    export_fourier_button.clicked.connect(on_export_fourier_series)
    animation_button.clicked.connect(on_save_animation)
    plot_animation_button.clicked.connect(on_save_plot_animation)
    reservoir_profile_button.clicked.connect(on_load_stl_profile)
    outer_reservoir_profile_button.clicked.connect(on_load_outer_stl_profile)
    channel_wss_button.clicked.connect(on_plot_channel_wss)
    save_image_button.clicked.connect(on_save_image)
    export_data_button.clicked.connect(on_export_data)
    save_channel_wss_image_button.clicked.connect(on_save_channel_wss_image)
    export_channel_wss_data_button.clicked.connect(on_export_channel_wss_data)
    advanced_mode_checkbox.toggled.connect(on_advanced_mode_changed)
    legacy_channel_driving_heads_checkbox.toggled.connect(
        lambda _: clear_cached_plot(
            "Driving-head model changed. Generate a fresh plot to apply it."
        )
    )
    viscosity_slider.valueChanged.connect(on_viscosity_slider_changed)
    two_reservoir_radio.toggled.connect(update_mode_visibility)
    single_reservoir_radio.toggled.connect(update_mode_visibility)
    simple_tilting_radio.toggled.connect(update_mode_visibility)
    nested_four_reservoir_radio.toggled.connect(update_mode_visibility)
    four_reservoir_checkbox.toggled.connect(update_mode_visibility)
    two_reservoir_radio.toggled.connect(update_reservoir_mode_visibility)
    single_reservoir_radio.toggled.connect(update_reservoir_mode_visibility)
    simple_tilting_radio.toggled.connect(update_reservoir_mode_visibility)
    nested_four_reservoir_radio.toggled.connect(update_reservoir_mode_visibility)
    use_stl_reservoir_checkbox.toggled.connect(update_reservoir_mode_visibility)
    use_stl_reservoir_checkbox.toggled.connect(update_mode_visibility)
    use_outer_stl_reservoir_checkbox.toggled.connect(update_reservoir_mode_visibility)
    use_outer_stl_reservoir_checkbox.toggled.connect(update_mode_visibility)
    top_contact_wetting_checkbox.stateChanged.connect(
        lambda _: clear_cached_plot(
            "Top-contact wetting changed. Generate a fresh plot to apply it."
        )
    )
    plot_q1_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    plot_q2_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    plot_q3_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    plot_q4_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    ring_global_direction_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    plot_tilts_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    plot_start_time_input.editingFinished.connect(redraw_cached_plot)
    flow_low_pass_checkbox.stateChanged.connect(
        lambda _: update_flow_low_pass_visibility()
    )
    flow_low_pass_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    flow_low_pass_cutoff_input.editingFinished.connect(redraw_cached_plot)
    plot_local_volumes_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    plot_midpoint_gravity_pressure_checkbox.stateChanged.connect(
        lambda _: redraw_cached_plot()
    )
    plot_port_heads_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    for checkbox in (
        plot_inner_outer_inlet_hydrostatic_checkbox,
        plot_inner_outer_outlet_hydrostatic_checkbox,
        plot_inner_outer_raw_pressure_checkbox,
    ):
        checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    plot_wss_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    plot_wss_reference_checkbox.stateChanged.connect(lambda _: update_reference_wss_visibility())
    plot_wss_reference_checkbox.stateChanged.connect(lambda _: redraw_cached_plot())
    assymmetric_tilting_checkbox.stateChanged.connect(
        lambda _: update_mode_visibility()
    )
    negative_phase_diff_checkbox.stateChanged.connect(
        lambda _: update_phase_difference()
    )
    primary_shape_input.currentTextChanged.connect(
        lambda _: on_channel_shape_changed()
    )
    secondary_shape_input.currentTextChanged.connect(
        lambda _: on_channel_shape_changed()
    )
    outer_channel_shape_input.currentTextChanged.connect(
        lambda _: on_channel_shape_changed()
    )
    reference_wss_input.editingFinished.connect(redraw_cached_plot)
    reference_flow_input.editingFinished.connect(redraw_cached_plot)
    channel_wss_wall_input.currentTextChanged.connect(
        lambda _: update_channel_wss_selection_hint(reset_to_full=True)
    )
    channel_wss_flow_mode_input.currentTextChanged.connect(
        lambda _: clear_channel_wss_state()
    )
    primary_width_input.editingFinished.connect(update_channel_wss_selection_hint)
    primary_height_input.editingFinished.connect(update_channel_wss_selection_hint)

    update_mode_visibility()
    update_channel_shape_visibility()
    update_reference_wss_visibility()
    update_flow_low_pass_visibility()
    update_reservoir_mode_visibility()
    update_advanced_mode_visibility()
    update_channel_wss_selection_hint(reset_to_full=True)

    left_column_layout.addStretch(1)
    middle_column_layout.addStretch(1)
    right_column_layout.addStretch(1)

    window.resize(1320, 760)
    window.show()
    app.exec()


def main():
    parser = argparse.ArgumentParser(description="Microfluidic flow plotting tool.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Generate the default plot without opening the GUI.",
    )
    parser.add_argument(
        "--stop-time",
        type=int,
        default=default_stop_time,
        help="Stop time in seconds for headless runs.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=default_time_step,
        help="Time step in seconds for headless runs.",
    )
    parser.add_argument(
        "--plot-q1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle plotting of Q_1 for headless runs.",
    )
    parser.add_argument(
        "--plot-q2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle plotting of Q_2 for headless runs.",
    )
    parser.add_argument(
        "--plot-wss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggle plotting of wall shear stress for headless runs.",
    )
    parser.add_argument(
        "--plot-wss-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggle plotting of reference wall shear stress for headless runs.",
    )
    parser.add_argument(
        "--reference-wss-pa",
        "--reference-wss-at-1uls",
        dest="reference_wss_pa",
        type=float,
        default=default_reference_wss_pa,
        help="Reference WSS in Pa at the configured reference flow.",
    )
    parser.add_argument(
        "--reference-flow",
        type=float,
        default=default_reference_flow_uL_s,
        help="Reference flow rate in uL/s for headless runs.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "symmetric",
            "assymmetric",
            "single_reservoir",
            "single_reservoir_assymmetric",
            "simple_tilting",
            "four_reservoir_tilting",
        ),
        default="symmetric",
        help="Simulation mode for headless runs.",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=default_tilt,
        help="Tilt in degrees for headless runs.",
    )
    parser.add_argument(
        "--speed1",
        type=float,
        default=default_mode_settings["symmetric"]["speed1"],
        help="Speed 1 in rpm for headless runs.",
    )
    parser.add_argument(
        "--speed2",
        type=float,
        default=default_mode_settings["assymmetric"]["speed2"],
        help="Speed 2 in rpm for assymmetric headless runs.",
    )
    parser.add_argument(
        "--slow-start",
        type=float,
        default=default_mode_settings["assymmetric"]["slow_start"],
        help="Normalized cycle start for the slow traversal in assymmetric headless runs.",
    )
    parser.add_argument(
        "--slow-duration",
        type=float,
        default=default_mode_settings["assymmetric"]["slow_duration"],
        help="Normalized duration of the slow traversal in assymmetric headless runs.",
    )
    parser.add_argument(
        "--cap-force",
        type=float,
        default=default_cap_force,
        help="Capillary pressure in Pa for headless runs (0-10).",
    )
    parser.add_argument(
        "--res-width",
        type=float,
        default=default_reservoir_settings["width"],
        help="Reservoir width in mm for headless runs.",
    )
    parser.add_argument(
        "--res-height",
        type=float,
        default=default_reservoir_settings["height"],
        help="Reservoir height in mm for headless runs.",
    )
    parser.add_argument(
        "--res-length",
        type=float,
        default=default_reservoir_settings["length"],
        help="Reservoir length in mm for headless runs.",
    )
    parser.add_argument(
        "--v-fill-ml",
        type=float,
        default=default_reservoir_settings["v_fill_ml"],
        help="Filled reservoir volume in mL for headless runs.",
    )
    parser.add_argument(
        "--reservoir-stl",
        type=str,
        default=None,
        help="Optional STL path for an extruded reservoir profile in headless runs.",
    )
    parser.add_argument(
        "--reservoir-left-port",
        type=str,
        default=None,
        help="Left port location on the extracted STL profile as x,y in millimeters.",
    )
    parser.add_argument(
        "--reservoir-right-port",
        type=str,
        default=None,
        help="Right port location on the extracted STL profile as x,y in millimeters.",
    )
    parser.add_argument(
        "--reservoir-extrusion-axis",
        choices=("auto", "x", "y", "z"),
        default="auto",
        help="Extrusion axis to use when slicing the STL profile.",
    )
    parser.add_argument(
        "--channel-width1",
        type=float,
        default=default_geometry_settings["primary"]["width"],
        help="Primary channel width in mm for headless runs.",
    )
    parser.add_argument(
        "--channel-height1",
        type=float,
        default=default_geometry_settings["primary"]["height"],
        help="Primary channel height in mm for headless runs.",
    )
    parser.add_argument(
        "--channel-length1",
        type=float,
        default=default_geometry_settings["primary"]["length"],
        help="Primary channel length in mm for headless runs.",
    )
    parser.add_argument(
        "--channel-width2",
        type=float,
        default=default_geometry_settings["secondary"]["width"],
        help="Secondary channel width in mm for assymmetric headless runs.",
    )
    parser.add_argument(
        "--channel-height2",
        type=float,
        default=default_geometry_settings["secondary"]["height"],
        help="Secondary channel height in mm for assymmetric headless runs.",
    )
    parser.add_argument(
        "--channel-length2",
        type=float,
        default=default_geometry_settings["secondary"]["length"],
        help="Secondary channel length in mm for assymmetric headless runs.",
    )
    args = parser.parse_args()

    if args.headless:
        stop_time = validate_stop_time(str(args.stop_time))
        time_step = validate_float_range(str(args.time_step), "Time step", 0.0001, 1)
        tilt = validate_float_range(str(args.tilt), "Tilt", 1, 45)
        speed1 = validate_float_range(str(args.speed1), "Speed 1", 0.1, 10)
        cap_force = validate_float_range(str(args.cap_force), "Capillary pressure", 0, 10)
        reference_wss_pa = validate_float_range(
            str(args.reference_wss_pa), "WSS reference", 0, 10
        )
        reference_flow_uL_s = validate_float_range(
            str(args.reference_flow), "Reference flow rate", 0, 10
        )
        if reference_flow_uL_s <= 0.0:
            raise ValueError("Reference flow rate must be greater than zero.")
        v_fill_ml = validate_float_range(str(args.v_fill_ml), "V_fill", 0.1, 10)
        if args.reservoir_stl:
            if args.reservoir_left_port is None or args.reservoir_right_port is None:
                raise ValueError(
                    "Headless STL runs require both --reservoir-left-port and --reservoir-right-port."
                )
            profile_template = load_profile_from_stl(
                args.reservoir_stl,
                extrusion_axis=parse_extrusion_axis(args.reservoir_extrusion_axis),
            )
            left_port = parse_port_coordinates(args.reservoir_left_port, "Left port")
            right_port = parse_port_coordinates(args.reservoir_right_port, "Right port")
            mid_width = profile_template["width"] / 2.0
            reservoir_geometry = build_profile_reservoir_geometry(
                profile_template["profile_points"],
                profile_template["width"],
                v_fill_ml,
                left_port,
                right_port,
                source_path=profile_template["source_path"],
                extrusion_axis=profile_template["extrusion_axis_name"],
                profile_axes=profile_template["profile_axes"],
                top_view_points=profile_template["top_view_points"],
                top_view_triangles=profile_template["top_view_triangles"],
                shell_triangles=profile_template["shell_triangles"],
                left_port_world=build_world_port_coordinates(
                    profile_template,
                    [left_port[0], mid_width],
                    left_port[1],
                ),
                right_port_world=build_world_port_coordinates(
                    profile_template,
                    [right_port[0], mid_width],
                    right_port[1],
                ),
                axis_mins=profile_template["axis_mins"],
                axis_maxs=profile_template["axis_maxs"],
                axis_names=stl_axis_names,
            )
            print(describe_reservoir_geometry(reservoir_geometry))
        else:
            reservoir_geometry = build_reservoir_geometry(
                validate_float_range(str(args.res_width), "Reservoir width", 1, 9),
                validate_float_range(str(args.res_height), "Reservoir height", 1, 10),
                validate_float_range(str(args.res_length), "Reservoir length", 1, 100),
                v_fill_ml,
            )
        primary_geometry = build_geometry(
            validate_float_range(str(args.channel_width1), "Primary channel width", 0.1, 5),
            validate_float_range(str(args.channel_height1), "Primary channel height", 0.1, 2),
            validate_float_range(str(args.channel_length1), "Primary channel length", 1, 100),
        )
        speed2 = None
        slow_start = None
        slow_duration = None
        secondary_geometry = None
        if args.mode in ("assymmetric", "single_reservoir_assymmetric"):
            speed2 = validate_float_range(str(args.speed2), "Speed 2", 0.1, 10)
            slow_start = validate_float_range(
                str(args.slow_start), "Slow traversal start", 0, 0.9
            )
            slow_duration = validate_float_range(
                str(args.slow_duration), "Slow traversal duration", 0.05, 0.95
            )
            if slow_start + slow_duration > 1:
                raise ValueError(
                    "Slow traversal start plus duration must not exceed 1."
                )
        if args.mode == "assymmetric":
            secondary_geometry = build_geometry(
                validate_float_range(str(args.channel_width2), "Secondary channel width", 0.1, 5),
                validate_float_range(str(args.channel_height2), "Secondary channel height", 0.1, 2),
                validate_float_range(str(args.channel_length2), "Secondary channel length", 1, 100),
            )
        plot_data = compute_plot_data(
            stop_time,
            mode=args.mode,
            tilt=tilt,
            speed1=speed1,
            time_step=time_step,
            reservoir_geometry=reservoir_geometry,
            primary_geometry=primary_geometry,
            cap_force=cap_force,
            speed2=speed2,
            slow_start=slow_start,
            slow_duration=slow_duration,
            secondary_geometry=secondary_geometry,
        )
        figure, output_path = create_figure_from_data(
            plot_data,
            plot_q1=args.plot_q1,
            plot_q2=args.plot_q2,
            plot_wss=args.plot_wss,
            plot_wss_reference=args.plot_wss_reference,
            reference_wss_pa=reference_wss_pa,
            reference_flow_uL_s=reference_flow_uL_s,
            save_path=Path(__file__).with_name("assync-tilt.png"),
        )
        plt.close(figure)
        print(f"Saved plot to {output_path.resolve()}")
        return

    launch_gui()


if __name__ == "__main__":
    main()
