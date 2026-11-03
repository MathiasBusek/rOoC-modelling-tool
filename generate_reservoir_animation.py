import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.argv.append("--headless")
import modellingScriptAssymetric as model

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def build_rectangular_profile_geometry(v_fill_ml):
    length = model.default_reservoir_settings["length"]
    height = model.default_reservoir_settings["height"]
    width = model.default_reservoir_settings["width"]
    profile_points = np.asarray(
        [
            [0.0, 0.0],
            [length, 0.0],
            [length, height],
            [0.0, height],
        ],
        dtype=float,
    )
    mid_width = width / 2
    return model.build_profile_reservoir_geometry(
        profile_points,
        width,
        v_fill_ml,
        [0.0, 0.0],
        [length, 0.0],
        source_path="rectangular-default",
        extrusion_axis="y",
        profile_axes=("x", "z"),
        left_port_world={"x": 0.0, "y": mid_width, "z": 0.0},
        right_port_world={"x": length, "y": mid_width, "z": 0.0},
        axis_mins=[0.0, 0.0, 0.0],
        axis_maxs=[length, width, height],
        axis_names=model.stl_axis_names,
    )


def build_profile_geometry_from_stl(stl_path, v_fill_ml):
    profile_template = model.load_profile_from_stl(stl_path)
    length = float(np.max(profile_template["profile_points"][:, 0]))
    height = float(np.max(profile_template["profile_points"][:, 1]))
    width = profile_template["width"]
    left_profile_port = [0.1 * length, 0.15 * height]
    right_profile_port = [0.9 * length, 0.15 * height]
    mid_width = width / 2
    left_world = model.build_world_port_coordinates(
        profile_template,
        [left_profile_port[0], mid_width],
        left_profile_port[1],
    )
    right_world = model.build_world_port_coordinates(
        profile_template,
        [right_profile_port[0], mid_width],
        right_profile_port[1],
    )
    return model.build_profile_reservoir_geometry(
        profile_template["profile_points"],
        width,
        v_fill_ml,
        left_profile_port,
        right_profile_port,
        source_path=profile_template["source_path"],
        extrusion_axis=profile_template["extrusion_axis_name"],
        profile_axes=profile_template["profile_axes"],
        left_port_world=left_world,
        right_port_world=right_world,
        axis_mins=profile_template["axis_mins"],
        axis_maxs=profile_template["axis_maxs"],
        axis_names=model.stl_axis_names,
    )


def get_default_output_path(cwd):
    return Path(cwd) / "reservoir-liquid-animation.gif"


def get_default_stl_path(cwd):
    candidate = Path(cwd) / "test-box-reservoir.stl"
    return candidate if candidate.exists() else None


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


def build_water_surface(geometry, intercept, slope_length, slope_width, x_count=70, width_count=30):
    length = geometry["length"]
    width = geometry["width"]
    x_values = np.linspace(0.0, length, x_count)
    width_values = np.linspace(0.0, width, width_count)
    x_grid, width_grid = np.meshgrid(x_values, width_values)
    z_grid = np.full_like(x_grid, np.nan, dtype=float)

    for x_index, x_value in enumerate(x_values):
        bounds = get_profile_vertical_bounds(geometry["profile_points"], x_value)
        if bounds is None:
            continue
        bottom_z, top_z = bounds
        plane_heights = intercept - slope_length * x_value - slope_width * width_values
        visible = plane_heights > bottom_z
        if not np.any(visible):
            continue
        z_column = np.where(visible, np.minimum(plane_heights, top_z), np.nan)
        z_grid[:, x_index] = z_column

    return x_grid, width_grid, z_grid


def draw_reservoir_wireframe(axis, geometry):
    profile = geometry["profile_points"]
    width = geometry["width"]
    closed_profile = np.vstack([profile, profile[0]])
    axis.plot(closed_profile[:, 0], np.zeros(len(closed_profile)), closed_profile[:, 1], color="0.35")
    axis.plot(closed_profile[:, 0], np.full(len(closed_profile), width), closed_profile[:, 1], color="0.35")

    connector_indices = np.linspace(0, len(profile) - 1, min(len(profile), 16), dtype=int)
    for index in connector_indices:
        point = profile[index]
        axis.plot(
            [point[0], point[0]],
            [0.0, width],
            [point[1], point[1]],
            color="0.55",
            linewidth=0.7,
        )


def build_geometry(args):
    if args.reservoir_stl:
        return build_profile_geometry_from_stl(args.reservoir_stl, args.v_fill_ml)
    return build_rectangular_profile_geometry(args.v_fill_ml)


def animate_reservoir(args):
    geometry = build_geometry(args)
    time_values = np.linspace(0.0, args.stop_time, args.frames)

    figure = plt.figure(figsize=(8.8, 6.8))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_box_aspect((geometry["length"], geometry["width"], geometry["height"]))

    def update(frame_index):
        axis.clear()
        time_value = float(time_values[frame_index])
        tilt_y = (
            -args.tilt_deg
            / 180
            * np.pi
            * model.rotation_signal(time_value, "single_reservoir", args.speed1, model.start_angle)
        )
        tilt_z = (
            -args.tilt_deg
            / 180
            * np.pi
            * model.rotation_signal(
                time_value,
                "single_reservoir",
                args.speed1,
                model.start_angle + model.phaseDiff,
            )
        )
        slope_length = math.tan(tilt_z)
        slope_width = math.tan(tilt_y)
        intercept = model.solve_profile_free_surface_3d(
            geometry,
            geometry["v_fill_ul"],
            slope_length,
            slope_width,
        )
        x_grid, width_grid, z_grid = build_water_surface(
            geometry,
            intercept,
            slope_length,
            slope_width,
        )

        draw_reservoir_wireframe(axis, geometry)
        axis.plot_surface(
            x_grid,
            width_grid,
            z_grid,
            color="#3a8dde",
            alpha=0.72,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        axis.scatter(
            [geometry["left_port"]["x"], geometry["right_port"]["x"]],
            [geometry["left_port_width"], geometry["right_port_width"]],
            [geometry["left_port"]["y"], geometry["right_port"]["y"]],
            color=["#d62828", "#2a9d8f"],
            s=30,
            depthshade=False,
        )

        axis.set_xlim(0.0, geometry["length"])
        axis.set_ylim(0.0, geometry["width"])
        axis.set_zlim(0.0, geometry["height"])
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_zlabel("z [mm]")
        axis.view_init(elev=24, azim=-58)
        axis.set_title(
            f"Reservoir Liquid Surface  t={time_value:.1f} s\n"
            f"tilt_y={math.degrees(tilt_y):.1f} deg, tilt_z={math.degrees(tilt_z):.1f} deg"
        )

    animation = FuncAnimation(figure, update, frames=len(time_values), interval=1000 / args.fps)
    output_path = Path(args.output).resolve()
    animation.save(output_path, writer=PillowWriter(fps=args.fps), dpi=110)
    plt.close(figure)
    return output_path


def parse_args():
    cwd = Path.cwd()
    default_stl = get_default_stl_path(cwd)
    parser = argparse.ArgumentParser(description="Generate a 3D animation of reservoir liquid levels.")
    parser.add_argument(
        "--reservoir-stl",
        type=str,
        default=None if default_stl is None else str(default_stl),
        help="Optional STL reservoir to animate. If omitted, uses the default rectangular reservoir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(get_default_output_path(cwd)),
        help="GIF output path.",
    )
    parser.add_argument("--frames", type=int, default=48, help="Number of animation frames.")
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second.")
    parser.add_argument("--stop-time", type=float, default=60.0, help="Simulated time span in seconds.")
    parser.add_argument("--tilt-deg", type=float, default=model.default_tilt, help="Tilt amplitude in degrees.")
    parser.add_argument(
        "--speed1",
        type=float,
        default=model.default_mode_settings["single_reservoir"]["speed1"],
        help="Rotator speed in rpm.",
    )
    parser.add_argument(
        "--v-fill-ml",
        type=float,
        default=model.default_reservoir_settings["v_fill_ml"],
        help="Reservoir fill volume in mL.",
    )
    cli_args = [argument for argument in sys.argv[1:] if argument != "--headless"]
    return parser.parse_args(cli_args)


def main():
    args = parse_args()
    output_path = animate_reservoir(args)
    print(f"Saved reservoir animation to {output_path}")


if __name__ == "__main__":
    main()
