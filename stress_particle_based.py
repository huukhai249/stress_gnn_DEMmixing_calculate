from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from edempy import Deck


@dataclass(frozen=True)
class StressSettings:
    start_time: float
    end_time: float
    calc_stress_tensor: bool
    calc_stress_intensity: bool
    calc_average_collision_number: bool
    calc_collision_force: bool
    generate_plots: bool
    host_particle_material: str
    extract_data_time: float
    high_impact_threshold: float


PAIR_KEY_SHIFT = np.int64(32)


@dataclass
class CollisionForceState:
    pair_keys: np.ndarray
    first_particle_ids: np.ndarray
    second_particle_ids: np.ndarray
    contact_positions: np.ndarray
    start_times: np.ndarray
    end_times: np.ndarray
    normal_force_magnitude: np.ndarray
    shear_force_magnitude: np.ndarray
    total_force_magnitude: np.ndarray


@dataclass(frozen=True)
class ParticleMetadata:
    particle_ids: np.ndarray
    particle_mass: np.ndarray
    particle_radius: np.ndarray
    index_order: np.ndarray


@dataclass(frozen=True)
class HostTypeMetadata:
    particle_types: tuple[Any, ...]


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"true", "1", "yes", "y"}


def normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def is_host_particle_type(particle_type: Any, particle_names: list[Any], host_material: str) -> bool:
    host_text = normalize_text(host_material)
    if not host_text:
        return False

    candidate_texts = {normalize_text(particle_type)}
    try:
        particle_index = int(particle_type)
    except (TypeError, ValueError):
        particle_index = None

    if particle_index is not None and 0 <= particle_index < len(particle_names):
        candidate_texts.add(normalize_text(particle_names[particle_index]))

    return host_text in candidate_texts


def find_nearest(array: np.ndarray, value: float) -> int:
    values = np.asarray(array, dtype=float)
    return int(np.abs(values - value).argmin())


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def compute_overlap_duration(
    start_times: np.ndarray,
    end_times: np.ndarray,
    interval_start: float,
    interval_end: float,
) -> np.ndarray:
    start_times = np.asarray(start_times, dtype=float)
    end_times = np.asarray(end_times, dtype=float)
    return np.maximum(0.0, np.minimum(end_times, interval_end) - np.maximum(start_times, interval_start))


def count_started_collisions(start_times: np.ndarray, interval_start: float, interval_end: float) -> int:
    if interval_end <= interval_start:
        return 0

    start_times = np.asarray(start_times, dtype=float)
    if start_times.size == 0:
        return 0

    started_after_lower = start_times >= interval_start if interval_start <= 0.0 else start_times > interval_start
    started_before_upper = start_times <= interval_end
    return int(np.count_nonzero(started_after_lower & started_before_upper))


def aggregate_specific_energy(energy_values: np.ndarray, mass_values: np.ndarray) -> float:
    energy_values = np.asarray(energy_values, dtype=float)
    mass_values = np.asarray(mass_values, dtype=float)
    valid = mass_values > 0
    if not np.any(valid):
        return 0.0
    return float(np.sum(energy_values[valid]) / np.sum(mass_values[valid]))


def accumulate_counts(indices: np.ndarray, size: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return np.zeros(size, dtype=int)
    return np.bincount(indices, minlength=size).astype(int, copy=False)


def accumulate_values(indices: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    values = np.asarray(values, dtype=float)
    if indices.size == 0:
        return np.zeros(size, dtype=float)
    return np.bincount(indices, weights=values, minlength=size).astype(float, copy=False)


def empty_collision_force_state() -> CollisionForceState:
    empty_int = np.array([], dtype=np.int64)
    empty_float = np.array([], dtype=float)
    return CollisionForceState(
        pair_keys=empty_int.copy(),
        first_particle_ids=empty_int.copy(),
        second_particle_ids=empty_int.copy(),
        contact_positions=np.empty((0, 3), dtype=float),
        start_times=empty_float.copy(),
        end_times=empty_float.copy(),
        normal_force_magnitude=empty_float.copy(),
        shear_force_magnitude=empty_float.copy(),
        total_force_magnitude=empty_float.copy(),
    )


def build_pair_keys(first_particle_ids: np.ndarray, second_particle_ids: np.ndarray) -> np.ndarray:
    first_particle_ids = np.asarray(first_particle_ids, dtype=np.int64)
    second_particle_ids = np.asarray(second_particle_ids, dtype=np.int64)
    pair_first = np.minimum(first_particle_ids, second_particle_ids)
    pair_second = np.maximum(first_particle_ids, second_particle_ids)
    return (pair_first << PAIR_KEY_SHIFT) | pair_second


def match_collisions_by_pair_and_position(
    current_pair_keys: np.ndarray,
    current_positions: np.ndarray,
    previous_pair_keys: np.ndarray,
    previous_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current_count = current_pair_keys.size
    previous_count = previous_pair_keys.size
    previous_indices_for_current = np.full(current_count, -1, dtype=int)

    if current_count == 0:
        return previous_indices_for_current, np.arange(previous_count, dtype=int)
    if previous_count == 0:
        return previous_indices_for_current, np.array([], dtype=int)

    previous_matched_mask = np.zeros(previous_count, dtype=bool)
    previous_order = np.argsort(previous_pair_keys, kind="mergesort")
    current_order = np.argsort(current_pair_keys, kind="mergesort")
    previous_sorted_keys = previous_pair_keys[previous_order]
    current_sorted_keys = current_pair_keys[current_order]

    previous_start = 0
    current_start = 0
    while previous_start < previous_count and current_start < current_count:
        previous_key = previous_sorted_keys[previous_start]
        current_key = current_sorted_keys[current_start]
        if previous_key < current_key:
            previous_start = int(np.searchsorted(previous_sorted_keys, previous_key, side="right"))
            continue
        if current_key < previous_key:
            current_start = int(np.searchsorted(current_sorted_keys, current_key, side="right"))
            continue

        previous_end = int(np.searchsorted(previous_sorted_keys, previous_key, side="right"))
        current_end = int(np.searchsorted(current_sorted_keys, current_key, side="right"))
        previous_group = previous_order[previous_start:previous_end]
        current_group = current_order[current_start:current_end]

        if previous_group.size == 1 and current_group.size == 1:
            previous_indices_for_current[current_group[0]] = previous_group[0]
            previous_matched_mask[previous_group[0]] = True
        else:
            previous_group_positions = previous_positions[previous_group]
            current_group_positions = current_positions[current_group]
            distance_matrix = np.linalg.norm(
                previous_group_positions[:, np.newaxis, :] - current_group_positions[np.newaxis, :, :],
                axis=2,
            )
            previous_used = np.zeros(previous_group.size, dtype=bool)
            current_used = np.zeros(current_group.size, dtype=bool)
            for flat_index in np.argsort(distance_matrix, axis=None):
                previous_local, current_local = np.unravel_index(flat_index, distance_matrix.shape)
                if previous_used[previous_local] or current_used[current_local]:
                    continue
                previous_used[previous_local] = True
                current_used[current_local] = True
                previous_index = previous_group[previous_local]
                current_index = current_group[current_local]
                previous_indices_for_current[current_index] = previous_index
                previous_matched_mask[previous_index] = True
                if np.count_nonzero(previous_used) == min(previous_group.size, current_group.size):
                    break

        previous_start = previous_end
        current_start = current_end

    return previous_indices_for_current, np.flatnonzero(~previous_matched_mask)


def settings_to_pairs(lines: list[str]) -> dict[str, str]:
    cleaned = [line.strip() for line in lines if line.strip()]
    if len(cleaned) % 2 != 0:
        raise ValueError("Stress_settings.txt must contain key/value pairs.")
    return {cleaned[index]: cleaned[index + 1] for index in range(0, len(cleaned), 2)}


def load_settings(settings_path: Path) -> StressSettings:
    with settings_path.open("r", encoding="utf-8") as handle:
        pairs = settings_to_pairs(handle.readlines())

    return StressSettings(
        start_time=float(pairs["Analysis_start_time (s)"]),
        end_time=float(pairs["Analysis_end_time(s)"]),
        calc_stress_tensor=parse_bool(pairs["Stress tensor calculation"]),
        calc_stress_intensity=parse_bool(pairs["Stress intensity calculation"]),
        calc_average_collision_number=parse_bool(pairs["Average collision number following time"]),
        calc_collision_force=parse_bool(pairs["Collision force"]),
        generate_plots=parse_bool(pairs["Generate_plots?"]),
        host_particle_material=pairs["Host particle material name (m)"],
        extract_data_time=float(pairs["Extract time (s)"]),
        high_impact_threshold=float(pairs.get("High-impact event threshold (N)", "0.0")),
    )


def expand_row_to_table(row_raw: dict[str, Any]) -> list[dict[str, Any]]:
    processed: dict[str, Any] = {}
    max_len = 1

    for key, value in row_raw.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            array_value = np.asarray(value)
            processed[key] = array_value
            max_len = max(max_len, len(array_value))
        else:
            processed[key] = value

    rows: list[dict[str, Any]] = []
    for index in range(max_len):
        row: dict[str, Any] = {}
        for key, value in processed.items():
            if isinstance(value, np.ndarray):
                row[key] = value[index] if index < len(value) else ""
            else:
                row[key] = value if index == 0 else ""
        rows.append(row)
    return rows


def build_particle_metadata(deck: Deck, timestep_index: int) -> ParticleMetadata:
    timestep = deck.timestep[timestep_index]
    ids: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    radii: list[np.ndarray] = []

    for particle_type in timestep.h5ParticleTypes:
        particle = timestep.particle[particle_type]
        ids.append(np.asarray(particle.getIds(), dtype=int))
        masses.append(np.asarray(particle.getMass(), dtype=float))
        radii.append(np.asarray(particle.getSphereRadii(), dtype=float))

    if not ids:
        return ParticleMetadata(
            particle_ids=np.array([], dtype=int),
            particle_mass=np.array([], dtype=float),
            particle_radius=np.array([], dtype=float),
            index_order=np.array([], dtype=int),
        )

    particle_ids = np.concatenate(ids)
    particle_mass = np.concatenate(masses)
    particle_radius = np.concatenate(radii)
    sort_order = np.argsort(particle_ids)
    particle_ids = particle_ids[sort_order]
    particle_mass = particle_mass[sort_order]
    particle_radius = particle_radius[sort_order]
    return ParticleMetadata(
        particle_ids=particle_ids,
        particle_mass=particle_mass,
        particle_radius=particle_radius,
        index_order=np.arange(particle_ids.size, dtype=int),
    )


def compute_collision_area_from_radius(first_radius: np.ndarray, second_radius: np.ndarray, mode: str = "min") -> np.ndarray:
    first_radius = np.asarray(first_radius, dtype=float)
    second_radius = np.asarray(second_radius, dtype=float)

    if mode == "min":
        radius = np.minimum(first_radius, second_radius)
    elif mode == "mean":
        radius = 0.5 * (first_radius + second_radius)
    else:
        raise ValueError("mode must be 'min' or 'mean'")

    return np.pi * radius**2


def format_time_for_filename(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "neg_").replace(".", "p")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_particle_angular_velocity_array(particle: Any) -> np.ndarray:
    method_names = (
        "getAngularVelocity",
        "getAngularVelocities",
        "getAngularVel",
        "getAngVel",
    )
    for method_name in method_names:
        method = getattr(particle, method_name, None)
        if callable(method):
            return np.asarray(method(), dtype=float)
    raise AttributeError(
        f"Could not find angular velocity getter on particle object. Tried: {', '.join(method_names)}"
    )


def angular_speed_magnitude(angular_velocity: np.ndarray) -> np.ndarray:
    angular_velocity = np.asarray(angular_velocity, dtype=float)
    if angular_velocity.size == 0:
        return np.array([], dtype=float)
    if angular_velocity.ndim == 1:
        return np.abs(angular_velocity)
    return np.linalg.norm(angular_velocity, axis=1)


def resolve_host_type_metadata(deck: Deck, timestep_index: int, host_particle: str) -> HostTypeMetadata:
    timestep = deck.timestep[timestep_index]
    particle_names = list(getattr(timestep, "particleNames", []))
    matched_types = tuple(
        particle_type
        for particle_type in timestep.h5ParticleTypes
        if is_host_particle_type(particle_type, particle_names, host_particle)
    )
    if matched_types:
        return HostTypeMetadata(particle_types=matched_types)

    available_types: list[str] = []
    for particle_type in timestep.h5ParticleTypes:
        description = str(particle_type)
        try:
            particle_index = int(particle_type)
        except (TypeError, ValueError):
            particle_index = None

        if particle_index is not None and 0 <= particle_index < len(particle_names):
            description = f"{description} ({particle_names[particle_index]})"
        available_types.append(description)
    raise ValueError(
        f"Host particle host '{host_particle}' was not found in particle types: {available_types}"
    )


def build_host_rotation_data(
    deck: Deck,
    timestep_index: int,
    host_type_metadata: HostTypeMetadata,
    particle_metadata: ParticleMetadata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestep = deck.timestep[timestep_index]
    ids: list[np.ndarray] = []
    angular_velocity: list[np.ndarray] = []

    for particle_type in host_type_metadata.particle_types:
        particle = timestep.particle[particle_type]
        ids.append(np.asarray(particle.getIds(), dtype=int))
        angular_velocity.append(get_particle_angular_velocity_array(particle))

    if not ids:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    host_ids = np.concatenate(ids)
    host_angular_velocity = np.concatenate(angular_velocity)
    host_indices, host_valid = map_particle_ids_to_indices(
        particle_metadata.index_order,
        particle_metadata.particle_ids,
        host_ids,
    )
    if not np.all(host_valid):
        missing_ids = host_ids[~host_valid][:10].tolist()
        raise ValueError(f"Could not map host particle ids to cached metadata. Sample missing ids: {missing_ids}")

    host_radii = particle_metadata.particle_radius[host_indices]
    return host_ids, host_radii, host_angular_velocity


def map_particle_ids_to_indices(
    sort_order: np.ndarray,
    sorted_particle_ids: np.ndarray,
    collision_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    particle_count = sorted_particle_ids.size
    if particle_count == 0 or collision_ids.size == 0:
        return np.array([], dtype=int), np.zeros(collision_ids.shape, dtype=bool)

    positions = np.searchsorted(sorted_particle_ids, collision_ids)

    valid = positions < particle_count
    valid[valid] &= sorted_particle_ids[positions[valid]] == collision_ids[valid]

    mapped_indices = np.empty(collision_ids.shape, dtype=int)
    mapped_indices.fill(-1)
    mapped_indices[valid] = sort_order[positions[valid]]
    return mapped_indices, valid


def analyze_timestep(
    deck: Deck,
    timestep_index: int,
    timestep_values: np.ndarray,
    settings: StressSettings,
    particle_metadata: ParticleMetadata,
    host_type_metadata: HostTypeMetadata,
    host_rotational_distance_total_by_id: dict[int, float],
    host_contact_time_total_by_id: dict[int, float],
    collision_force_state: CollisionForceState,
) -> tuple[dict[str, float], dict[str, Any] | None, dict[str, Any] | None]:
    timestep = deck.timestep[timestep_index]
    particle_ids = particle_metadata.particle_ids
    particle_mass = particle_metadata.particle_mass
    particle_radius = particle_metadata.particle_radius
    particle_count = particle_ids.size
    time_value = float(timestep_values[timestep_index])

    summary: dict[str, float] = {
        "time": time_value,
        "number_of_collisions": 0.0,
        "Ave_contact_duration_per_collision": 0.0,
        "total_collision_time": 0.0,
        "average_collision_time": 0.0,
        "collision_frequency": 0.0,
        "Ave_number_of_collisions_per_particle": 0.0,
        "Ave_TotalColForce_inSys_byParticle": 0.0,
        "Ave_TotalColForce_inSys_byCollision": 0.0,
        "Ave_Fcol_Max_per_collision": 0.0,
        "Max_Fcol_Max_per_collision": 0.0,
        "high_impact_event_fraction": 0.0,
        "Ave_SI_normal_per_collision": 0.0,
        "Ave_SI_shear_per_collision": 0.0,
        "Ave_SI_total_per_collision": 0.0,
        "Total_normal_specific_energy_by_collision": 0.0,
        "Total_shear_specific_energy_by_collision": 0.0,
        "Total_specific_energy_by_collision": 0.0,
        "Ave_SI_normal_per_particle": 0.0,
        "Ave_SI_shear_per_particle": 0.0,
        "Ave_SI_total_per_particle": 0.0,
        "Total_normal_specific_energy_by_particle": 0.0,
        "Total_shear_specific_energy_by_particle": 0.0,
        "Total_specific_energy_by_particle": 0.0,
        "Ave_normal_stress_value_per_collision": 0.0,
        "Ave_shear_stress_value_per_collision": 0.0,
        "Ave_stress_value_per_collision": 0.0,
        "Ave_normal_power_per_collision": 0.0,
        "Ave_shear_power_per_collision": 0.0,
        "Ave_power_per_collision": 0.0,
        "Ave_normal_stress_value_per_particle": 0.0,
        "Ave_shear_stress_value_per_particle": 0.0,
        "Ave_stress_value_per_particle": 0.0,
        "Ave_normal_power_per_particle": 0.0,
        "Ave_shear_power_per_particle": 0.0,
        "Ave_power_per_particle": 0.0,
        "Total_normal_power_by_collision": 0.0,
        "Total_shear_power_by_collision": 0.0,
        "Total_power_by_collision": 0.0,
        "Total_normal_power_by_particle": 0.0,
        "Total_shear_power_by_particle": 0.0,
        "Total_power_by_particle": 0.0,
        "Ave_rotational_distance_per_host_particle": 0.0,
        "Ave_rotational_distance_total_per_host_particle": 0.0,
        "Ave_cumulative_contact_time_per_host_particle": 0.0,
    }

    extracted = {
        "time": time_value,
        "particle_id": particle_ids,
        "particle_mass": particle_mass,
        "collision_number_per_particle": np.zeros(particle_count, dtype=int),
        "Ave_TotalColForce_per_particle": np.zeros(particle_count, dtype=float),
        "SI_normal_per_particle": np.zeros(particle_count, dtype=float),
        "SI_shear_per_particle": np.zeros(particle_count, dtype=float),
        "SI_total_per_particle": np.zeros(particle_count, dtype=float),
        "specific_energy_normal_per_particle": np.zeros(particle_count, dtype=float),
        "specific_energy_shear_per_particle": np.zeros(particle_count, dtype=float),
        "specific_energy_total_per_particle": np.zeros(particle_count, dtype=float),
        "normal_stress_value_per_particle": np.zeros(particle_count, dtype=float),
        "shear_stress_value_per_particle": np.zeros(particle_count, dtype=float),
        "stress_value_per_particle": np.zeros(particle_count, dtype=float),
        "normal_power_per_particle": np.zeros(particle_count, dtype=float),
        "shear_power_per_particle": np.zeros(particle_count, dtype=float),
        "power_per_particle": np.zeros(particle_count, dtype=float),
        "rotational_distance_contact_xi_per_particle": np.zeros(particle_count, dtype=int),
        "rotational_distance_per_particle": np.zeros(particle_count, dtype=float),
        "rotational_distance_total_per_particle": np.zeros(particle_count, dtype=float),
    }

    if particle_count == 0:
        return summary, extracted, None

    collision = timestep.collision.surfSurf
    pid_st = np.asarray(collision.getFirstIds(), dtype=int)
    pid_nd = np.asarray(collision.getSecondIds(), dtype=int)
    collision_count = int(collision.getNumCollisions())

    if not (len(pid_st) == len(pid_nd) == collision_count):
        raise ValueError(f"Collision id arrays mismatch at timestep {timestep_index}.")

    summary["number_of_collisions"] = float(collision_count)
    
    # Current timestep interval
    start_time_dt = float(timestep_values[timestep_index - 1]) if timestep_index > 0 else 0.0
    end_time_dt = float(timestep_values[timestep_index])
    delta_time = max(0.0, end_time_dt - start_time_dt)

    overlap_duration_per_collision = np.zeros(collision_count, dtype=float)
    contact_duration_per_collision = np.zeros(collision_count, dtype=float)
    start_time_col = np.array([], dtype=float)
    end_time_col = np.array([], dtype=float)
    new_collision_count = 0

    if collision_count > 0:
        start_time_col = np.asarray(collision.getStartTime(), dtype=float)
        end_time_col = np.asarray(collision.getEndTimes(), dtype=float)
        if len(start_time_col) != collision_count or len(end_time_col) != collision_count:
            raise ValueError(f"Collision time arrays length mismatch at timestep {timestep_index}.")

        overlap_duration_per_collision = compute_overlap_duration(start_time_col, end_time_col, start_time_dt, end_time_dt)
        contact_duration_per_collision = np.maximum(0.0, end_time_col - start_time_col)
        summary["Ave_contact_duration_per_collision"] = safe_mean(contact_duration_per_collision)
        summary["total_collision_time"] = float(np.sum(overlap_duration_per_collision))
        summary["average_collision_time"] = safe_mean(overlap_duration_per_collision)
        new_collision_count = count_started_collisions(start_time_col, start_time_dt, end_time_dt)

    summary["collision_frequency"] = float(new_collision_count / delta_time) if delta_time > 0 else 0.0

    sort_order = particle_metadata.index_order
    sorted_particle_ids = particle_ids
    first_idx_all, first_valid = map_particle_ids_to_indices(sort_order, sorted_particle_ids, pid_st)
    second_idx_all, second_valid = map_particle_ids_to_indices(sort_order, sorted_particle_ids, pid_nd)
    if not np.all(first_valid):
        missing_ids = pid_st[~first_valid][:10].tolist()
        raise ValueError(f"Could not map first collision particle ids to cached metadata. Sample missing ids: {missing_ids}")
    if not np.all(second_valid):
        missing_ids = pid_nd[~second_valid][:10].tolist()
        raise ValueError(f"Could not map second collision particle ids to cached metadata. Sample missing ids: {missing_ids}")
    pair_valid = first_valid & second_valid
    valid_first_idx = first_idx_all[pair_valid]
    valid_second_idx = second_idx_all[pair_valid]

    collision_number_per_particle = extracted["collision_number_per_particle"]
    if collision_count > 0:
        counted_indices: list[np.ndarray] = []
        if np.any(first_valid):
            counted_indices.append(first_idx_all[first_valid])
        if np.any(second_valid):
            counted_indices.append(second_idx_all[second_valid])
        if counted_indices:
            collision_number_per_particle[:] = accumulate_counts(np.concatenate(counted_indices), particle_count)

    if settings.calc_average_collision_number:
        summary["Ave_number_of_collisions_per_particle"] = float(collision_number_per_particle.mean())

    total_normal_force = None
    total_tangential_force = None
    normal_energy = None
    shear_energy = None
    dissipation_energy = None
    collision_force_mag_per_collision = np.zeros(collision_count, dtype=float)

    if settings.calc_collision_force and collision_count > 0:
        total_normal_force = np.asarray(collision.getTotalNormalForce(), dtype=float)
        total_tangential_force = np.asarray(collision.getTotalTangentialForce(), dtype=float)
        max_normal_force = np.asarray(collision.getMaxNormalForce(), dtype=float)
        max_tangential_force = np.asarray(collision.getMaxTangentialForce(), dtype=float)

        lengths = (
            len(total_normal_force),
            len(total_tangential_force),
            len(max_normal_force),
            len(max_tangential_force),
        )
        if any(length != collision_count for length in lengths):
            raise ValueError(f"Collision force arrays length mismatch at timestep {timestep_index}.")

        collision_force_mag_per_collision = np.linalg.norm(total_normal_force + total_tangential_force, axis=1)
        total_collision_force_magnitude_per_particle = np.zeros(particle_count, dtype=float)
        average_force_per_particle = extracted["Ave_TotalColForce_per_particle"]
        has_collision = collision_number_per_particle > 0

        force_indices: list[np.ndarray] = []
        force_values: list[np.ndarray] = []
        if np.any(first_valid):
            force_indices.append(first_idx_all[first_valid])
            force_values.append(collision_force_mag_per_collision[first_valid])
        if np.any(second_valid):
            force_indices.append(second_idx_all[second_valid])
            force_values.append(collision_force_mag_per_collision[second_valid])
        if force_indices:
            total_collision_force_magnitude_per_particle[:] = accumulate_values(
                np.concatenate(force_indices),
                np.concatenate(force_values),
                particle_count,
            )

        average_force_per_particle[:] = np.divide(
            total_collision_force_magnitude_per_particle,
            collision_number_per_particle,
            out=np.zeros_like(total_collision_force_magnitude_per_particle),
            where=has_collision,
        )

        summary["Ave_TotalColForce_inSys_byParticle"] = float(average_force_per_particle.mean())
        summary["Ave_TotalColForce_inSys_byCollision"] = safe_mean(collision_force_mag_per_collision)

        max_force_per_collision = np.linalg.norm(max_normal_force + max_tangential_force, axis=1)
        summary["Ave_Fcol_Max_per_collision"] = safe_mean(max_force_per_collision)
        summary["Max_Fcol_Max_per_collision"] = float(np.max(max_force_per_collision)) if max_force_per_collision.size else 0.0

        if max_force_per_collision.size > 0:
            threshold = settings.high_impact_threshold
            high_impact_count = np.sum(max_force_per_collision > threshold)
            summary["high_impact_event_fraction"] = float(high_impact_count / len(max_force_per_collision))
        else:
            summary["high_impact_event_fraction"] = 0.0

    if settings.calc_stress_intensity and np.any(pair_valid):
        normal_energy = np.asarray(collision.getNormalEnergy(), dtype=float)
        shear_energy = np.asarray(collision.getShearEnergy(), dtype=float)
        dissipation_energy = np.zeros_like(normal_energy)

        lengths = (len(normal_energy), len(shear_energy))
        if any(length != collision_count for length in lengths):
            raise ValueError(f"Collision energy arrays length mismatch at timestep {timestep_index}.")

        normal_energy_valid = normal_energy[pair_valid]
        shear_energy_valid = shear_energy[pair_valid]
        dissipation_energy_valid = dissipation_energy[pair_valid]
        total_energy_valid = normal_energy_valid + shear_energy_valid + dissipation_energy_valid

        mass_first = particle_mass[valid_first_idx]
        mass_second = particle_mass[valid_second_idx]
        reduced_mass = np.divide(
            mass_first * mass_second,
            mass_first + mass_second,
            out=np.zeros_like(mass_first),
            where=(mass_first + mass_second) > 0,
        )
        valid_reduced_mass = reduced_mass > 0

        si_normal_per_collision = np.divide(
            normal_energy_valid,
            reduced_mass,
            out=np.zeros_like(reduced_mass),
            where=valid_reduced_mass,
        )
        si_shear_per_collision = np.divide(
            shear_energy_valid,
            reduced_mass,
            out=np.zeros_like(reduced_mass),
            where=valid_reduced_mass,
        )
        si_total_per_collision = np.divide(
            total_energy_valid,
            reduced_mass,
            out=np.zeros_like(reduced_mass),
            where=valid_reduced_mass,
        )

        summary["Ave_SI_normal_per_collision"] = safe_mean(si_normal_per_collision[valid_reduced_mass])
        summary["Ave_SI_shear_per_collision"] = safe_mean(si_shear_per_collision[valid_reduced_mass])
        summary["Ave_SI_total_per_collision"] = safe_mean(si_total_per_collision[valid_reduced_mass])
        summary["Total_normal_specific_energy_by_collision"] = aggregate_specific_energy(
            normal_energy_valid,
            reduced_mass,
        )
        summary["Total_shear_specific_energy_by_collision"] = aggregate_specific_energy(
            shear_energy_valid,
            reduced_mass,
        )
        summary["Total_specific_energy_by_collision"] = aggregate_specific_energy(
            total_energy_valid,
            reduced_mass,
        )

        energy_normal_per_particle = np.zeros(particle_count, dtype=float)
        energy_shear_per_particle = np.zeros(particle_count, dtype=float)
        energy_total_per_particle = np.zeros(particle_count, dtype=float)

        half_normal = 0.5 * normal_energy_valid
        half_shear = 0.5 * shear_energy_valid
        half_total = 0.5 * total_energy_valid

        combined_pair_indices = np.concatenate((valid_first_idx, valid_second_idx))
        energy_normal_per_particle[:] = accumulate_values(
            combined_pair_indices,
            np.concatenate((half_normal, half_normal)),
            particle_count,
        )
        energy_shear_per_particle[:] = accumulate_values(
            combined_pair_indices,
            np.concatenate((half_shear, half_shear)),
            particle_count,
        )
        energy_total_per_particle[:] = accumulate_values(
            combined_pair_indices,
            np.concatenate((half_total, half_total)),
            particle_count,
        )

        positive_mass = particle_mass > 0
        extracted["SI_normal_per_particle"][:] = np.divide(
            energy_normal_per_particle,
            particle_mass,
            out=np.zeros_like(energy_normal_per_particle),
            where=positive_mass,
        )
        extracted["SI_shear_per_particle"][:] = np.divide(
            energy_shear_per_particle,
            particle_mass,
            out=np.zeros_like(energy_shear_per_particle),
            where=positive_mass,
        )
        extracted["SI_total_per_particle"][:] = np.divide(
            energy_total_per_particle,
            particle_mass,
            out=np.zeros_like(energy_total_per_particle),
            where=positive_mass,
        )
        extracted["specific_energy_normal_per_particle"][:] = extracted["SI_normal_per_particle"]
        extracted["specific_energy_shear_per_particle"][:] = extracted["SI_shear_per_particle"]
        extracted["specific_energy_total_per_particle"][:] = extracted["SI_total_per_particle"]

        summary["Ave_SI_normal_per_particle"] = float(extracted["SI_normal_per_particle"].mean())
        summary["Ave_SI_shear_per_particle"] = float(extracted["SI_shear_per_particle"].mean())
        summary["Ave_SI_total_per_particle"] = float(extracted["SI_total_per_particle"].mean())
        summary["Total_normal_specific_energy_by_particle"] = aggregate_specific_energy(
            energy_normal_per_particle,
            particle_mass,
        )
        summary["Total_shear_specific_energy_by_particle"] = aggregate_specific_energy(
            energy_shear_per_particle,
            particle_mass,
        )
        summary["Total_specific_energy_by_particle"] = aggregate_specific_energy(
            energy_total_per_particle,
            particle_mass,
        )

    collision_extracted = None

    if settings.calc_stress_tensor:
        if total_normal_force is None:
            total_normal_force = np.asarray(collision.getTotalNormalForce(), dtype=float)
            total_tangential_force = np.asarray(collision.getTotalTangentialForce(), dtype=float)
            if len(total_normal_force) != collision_count or len(total_tangential_force) != collision_count:
                raise ValueError(f"Collision force arrays length mismatch at timestep {timestep_index}.")
        if normal_energy is None:
            normal_energy = np.asarray(collision.getNormalEnergy(), dtype=float)
            shear_energy = np.asarray(collision.getShearEnergy(), dtype=float)
            dissipation_energy = np.zeros_like(normal_energy)
            if len(normal_energy) != collision_count or len(shear_energy) != collision_count:
                raise ValueError(f"Collision energy arrays length mismatch at timestep {timestep_index}.")

        first_radius = np.asarray(collision.getFirstRadius(), dtype=float)
        second_radius = np.asarray(collision.getSecondRadius(), dtype=float)
        if len(first_radius) != collision_count or len(second_radius) != collision_count:
            raise ValueError(f"Collision radius arrays length mismatch at timestep {timestep_index}.")
        contact_position = np.asarray(collision.getPosition(), dtype=float)
        if len(contact_position) != collision_count:
            raise ValueError(f"Collision position arrays length mismatch at timestep {timestep_index}.")

        total_energy = normal_energy + shear_energy + dissipation_energy
        normal_force_mag_per_collision = np.linalg.norm(total_normal_force, axis=1)
        shear_force_mag_per_collision = np.linalg.norm(total_tangential_force, axis=1)
        collision_force_mag_per_collision = np.linalg.norm(total_normal_force + total_tangential_force, axis=1)
        current_pair_keys = build_pair_keys(pid_st, pid_nd)
        previous_indices_for_current, previous_only_indices = match_collisions_by_pair_and_position(
            current_pair_keys,
            contact_position,
            collision_force_state.pair_keys,
            collision_force_state.contact_positions,
        )
        previous_normal_force = np.zeros(collision_count, dtype=float)
        previous_shear_force = np.zeros(collision_count, dtype=float)
        previous_total_force = np.zeros(collision_count, dtype=float)
        has_previous_force = previous_indices_for_current >= 0
        if np.any(has_previous_force):
            matched_previous_indices = previous_indices_for_current[has_previous_force]
            previous_normal_force[has_previous_force] = collision_force_state.normal_force_magnitude[matched_previous_indices]
            previous_shear_force[has_previous_force] = collision_force_state.shear_force_magnitude[matched_previous_indices]
            previous_total_force[has_previous_force] = collision_force_state.total_force_magnitude[matched_previous_indices]

        normal_impulse_per_collision = 0.5 * (previous_normal_force + normal_force_mag_per_collision) * overlap_duration_per_collision
        shear_impulse_per_collision = 0.5 * (previous_shear_force + shear_force_mag_per_collision) * overlap_duration_per_collision
        impulse_per_collision = 0.5 * (previous_total_force + collision_force_mag_per_collision) * overlap_duration_per_collision

        collision_area = compute_collision_area_from_radius(first_radius, second_radius, mode="mean")
        has_overlap = overlap_duration_per_collision > 0
        normal_stress_value_per_collision = np.divide(
            normal_impulse_per_collision,
            overlap_duration_per_collision * collision_area,
            out=np.zeros_like(normal_impulse_per_collision),
            where=has_overlap & (collision_area > 0),
        )
        shear_stress_value_per_collision = np.divide(
            shear_impulse_per_collision,
            overlap_duration_per_collision * collision_area,
            out=np.zeros_like(shear_impulse_per_collision),
            where=has_overlap & (collision_area > 0),
        )
        stress_value_per_collision = np.divide(
            impulse_per_collision,
            overlap_duration_per_collision * collision_area,
            out=np.zeros_like(impulse_per_collision),
            where=has_overlap & (collision_area > 0),
        )
        power_per_collision = np.divide(
            total_energy,
            delta_time,
            out=np.zeros_like(total_energy),
            where=delta_time > 0,
        )
        normal_power_per_collision = np.divide(
            normal_energy,
            delta_time,
            out=np.zeros_like(normal_energy),
            where=delta_time > 0,
        )
        shear_power_per_collision = np.divide(
            shear_energy,
            delta_time,
            out=np.zeros_like(shear_energy),
            where=delta_time > 0,
        )

        summary["Ave_normal_stress_value_per_collision"] = safe_mean(normal_stress_value_per_collision)
        summary["Ave_shear_stress_value_per_collision"] = safe_mean(shear_stress_value_per_collision)
        summary["Ave_stress_value_per_collision"] = safe_mean(stress_value_per_collision)
        summary["Ave_normal_power_per_collision"] = safe_mean(normal_power_per_collision)
        summary["Ave_shear_power_per_collision"] = safe_mean(shear_power_per_collision)
        summary["Ave_power_per_collision"] = safe_mean(power_per_collision)
        summary["Total_normal_power_by_collision"] = float(np.sum(normal_power_per_collision))
        summary["Total_shear_power_by_collision"] = float(np.sum(shear_power_per_collision))
        summary["Total_power_by_collision"] = float(np.sum(power_per_collision))

        collision_extracted = {
            "time": time_value,
            "collision_id": np.arange(collision_count, dtype=int),
            "first_particle_id": pid_st,
            "second_particle_id": pid_nd,
            "specific_energy_normal_per_collision": np.zeros(collision_count, dtype=float),
            "specific_energy_shear_per_collision": np.zeros(collision_count, dtype=float),
            "specific_energy_total_per_collision": np.zeros(collision_count, dtype=float),
            "collision_area": collision_area,
            "normal_force_magnitude_per_collision": normal_force_mag_per_collision,
            "shear_force_magnitude_per_collision": shear_force_mag_per_collision,
            "total_force_magnitude_per_collision": collision_force_mag_per_collision,
            "normal_stress_value_per_collision": normal_stress_value_per_collision,
            "shear_stress_value_per_collision": shear_stress_value_per_collision,
            "stress_value_per_collision": stress_value_per_collision,
            "normal_power_per_collision": normal_power_per_collision,
            "shear_power_per_collision": shear_power_per_collision,
            "power_per_collision": power_per_collision,
        }
        if settings.calc_stress_intensity and np.any(pair_valid):
            collision_extracted["specific_energy_normal_per_collision"][pair_valid] = si_normal_per_collision
            collision_extracted["specific_energy_shear_per_collision"][pair_valid] = si_shear_per_collision
            collision_extracted["specific_energy_total_per_collision"][pair_valid] = si_total_per_collision

        if collision_count > 0:
            normal_impulse_per_particle = np.zeros(particle_count, dtype=float)
            shear_impulse_per_particle = np.zeros(particle_count, dtype=float)
            impulse_per_particle = np.zeros(particle_count, dtype=float)
            contact_time_per_particle = np.zeros(particle_count, dtype=float)
            normal_energy_per_particle = np.zeros(particle_count, dtype=float)
            shear_energy_per_particle = np.zeros(particle_count, dtype=float)
            energy_total_per_particle = np.zeros(particle_count, dtype=float)
            impulse_indices: list[np.ndarray] = []
            normal_impulse_values: list[np.ndarray] = []
            shear_impulse_values: list[np.ndarray] = []
            total_impulse_values: list[np.ndarray] = []
            contact_time_values: list[np.ndarray] = []
            if np.any(first_valid):
                impulse_indices.append(first_idx_all[first_valid])
                normal_impulse_values.append(normal_impulse_per_collision[first_valid])
                shear_impulse_values.append(shear_impulse_per_collision[first_valid])
                total_impulse_values.append(impulse_per_collision[first_valid])
                contact_time_values.append(overlap_duration_per_collision[first_valid])
            if np.any(second_valid):
                impulse_indices.append(second_idx_all[second_valid])
                normal_impulse_values.append(normal_impulse_per_collision[second_valid])
                shear_impulse_values.append(shear_impulse_per_collision[second_valid])
                total_impulse_values.append(impulse_per_collision[second_valid])
                contact_time_values.append(overlap_duration_per_collision[second_valid])

            if previous_only_indices.size:
                tail_duration = compute_overlap_duration(
                    collision_force_state.start_times[previous_only_indices],
                    collision_force_state.end_times[previous_only_indices],
                    start_time_dt,
                    end_time_dt,
                )
                tail_valid = tail_duration > 0
                if np.any(tail_valid):
                    tail_duration = tail_duration[tail_valid]
                    tail_previous_indices = previous_only_indices[tail_valid]
                    tail_normal_impulse = 0.5 * collision_force_state.normal_force_magnitude[tail_previous_indices] * tail_duration
                    tail_shear_impulse = 0.5 * collision_force_state.shear_force_magnitude[tail_previous_indices] * tail_duration
                    tail_total_impulse = 0.5 * collision_force_state.total_force_magnitude[tail_previous_indices] * tail_duration
                    tail_first_ids = collision_force_state.first_particle_ids[tail_previous_indices]
                    tail_second_ids = collision_force_state.second_particle_ids[tail_previous_indices]

                    tail_first_idx, tail_first_valid = map_particle_ids_to_indices(sort_order, sorted_particle_ids, tail_first_ids)
                    tail_second_idx, tail_second_valid = map_particle_ids_to_indices(sort_order, sorted_particle_ids, tail_second_ids)

                    if np.any(tail_first_valid):
                        impulse_indices.append(tail_first_idx[tail_first_valid])
                        normal_impulse_values.append(tail_normal_impulse[tail_first_valid])
                        shear_impulse_values.append(tail_shear_impulse[tail_first_valid])
                        total_impulse_values.append(tail_total_impulse[tail_first_valid])
                        contact_time_values.append(tail_duration[tail_first_valid])
                    if np.any(tail_second_valid):
                        impulse_indices.append(tail_second_idx[tail_second_valid])
                        normal_impulse_values.append(tail_normal_impulse[tail_second_valid])
                        shear_impulse_values.append(tail_shear_impulse[tail_second_valid])
                        total_impulse_values.append(tail_total_impulse[tail_second_valid])
                        contact_time_values.append(tail_duration[tail_second_valid])
            if impulse_indices:
                combined_impulse_indices = np.concatenate(impulse_indices)
                normal_impulse_per_particle[:] = accumulate_values(
                    combined_impulse_indices,
                    np.concatenate(normal_impulse_values),
                    particle_count,
                )
                shear_impulse_per_particle[:] = accumulate_values(
                    combined_impulse_indices,
                    np.concatenate(shear_impulse_values),
                    particle_count,
                )
                impulse_per_particle[:] = accumulate_values(
                    combined_impulse_indices,
                    np.concatenate(total_impulse_values),
                    particle_count,
                )
                contact_time_per_particle[:] = accumulate_values(
                    combined_impulse_indices,
                    np.concatenate(contact_time_values),
                    particle_count,
                )
            if np.any(pair_valid):
                half_normal_energy = 0.5 * normal_energy[pair_valid]
                half_shear_energy = 0.5 * shear_energy[pair_valid]
                half_total_energy = 0.5 * total_energy[pair_valid]
                combined_pair_indices = np.concatenate((valid_first_idx, valid_second_idx))
                normal_energy_per_particle[:] = accumulate_values(
                    combined_pair_indices,
                    np.concatenate((half_normal_energy, half_normal_energy)),
                    particle_count,
                )
                shear_energy_per_particle[:] = accumulate_values(
                    combined_pair_indices,
                    np.concatenate((half_shear_energy, half_shear_energy)),
                    particle_count,
                )
                energy_total_per_particle[:] = accumulate_values(
                    combined_pair_indices,
                    np.concatenate((half_total_energy, half_total_energy)),
                    particle_count,
                )

            particle_area = np.pi * particle_radius**2
            extracted["normal_stress_value_per_particle"][:] = np.divide(
                normal_impulse_per_particle,
                contact_time_per_particle * particle_area,
                out=np.zeros_like(normal_impulse_per_particle),
                where=(contact_time_per_particle > 0) & (particle_area > 0),
            )
            extracted["shear_stress_value_per_particle"][:] = np.divide(
                shear_impulse_per_particle,
                contact_time_per_particle * particle_area,
                out=np.zeros_like(shear_impulse_per_particle),
                where=(contact_time_per_particle > 0) & (particle_area > 0),
            )
            extracted["stress_value_per_particle"][:] = np.divide(
                impulse_per_particle,
                contact_time_per_particle * particle_area,
                out=np.zeros_like(impulse_per_particle),
                where=(contact_time_per_particle > 0) & (particle_area > 0),
            )
            extracted["power_per_particle"][:] = np.divide(
                energy_total_per_particle,
                delta_time,
                out=np.zeros_like(energy_total_per_particle),
                where=delta_time > 0,
            )
            extracted["normal_power_per_particle"][:] = np.divide(
                normal_energy_per_particle,
                delta_time,
                out=np.zeros_like(normal_energy_per_particle),
                where=delta_time > 0,
            )
            extracted["shear_power_per_particle"][:] = np.divide(
                shear_energy_per_particle,
                delta_time,
                out=np.zeros_like(shear_energy_per_particle),
                where=delta_time > 0,
            )

            summary["Ave_normal_stress_value_per_particle"] = float(extracted["normal_stress_value_per_particle"].mean())
            summary["Ave_shear_stress_value_per_particle"] = float(extracted["shear_stress_value_per_particle"].mean())
            summary["Ave_stress_value_per_particle"] = float(extracted["stress_value_per_particle"].mean())
            summary["Ave_normal_power_per_particle"] = float(extracted["normal_power_per_particle"].mean())
            summary["Ave_shear_power_per_particle"] = float(extracted["shear_power_per_particle"].mean())
            summary["Ave_power_per_particle"] = float(extracted["power_per_particle"].mean())
            summary["Total_normal_power_by_particle"] = float(np.sum(extracted["normal_power_per_particle"]))
            summary["Total_shear_power_by_particle"] = float(np.sum(extracted["shear_power_per_particle"]))
            summary["Total_power_by_particle"] = float(np.sum(extracted["power_per_particle"]))

        collision_force_state.pair_keys = current_pair_keys
        collision_force_state.first_particle_ids = pid_st.copy()
        collision_force_state.second_particle_ids = pid_nd.copy()
        collision_force_state.contact_positions = contact_position.copy()
        collision_force_state.start_times = start_time_col.copy()
        collision_force_state.end_times = end_time_col.copy()
        collision_force_state.normal_force_magnitude = normal_force_mag_per_collision.copy()
        collision_force_state.shear_force_magnitude = shear_force_mag_per_collision.copy()
        collision_force_state.total_force_magnitude = collision_force_mag_per_collision.copy()
    else:
        collision_force_state.pair_keys = np.array([], dtype=np.int64)
        collision_force_state.first_particle_ids = np.array([], dtype=int)
        collision_force_state.second_particle_ids = np.array([], dtype=int)
        collision_force_state.contact_positions = np.empty((0, 3), dtype=float)
        collision_force_state.start_times = np.array([], dtype=float)
        collision_force_state.end_times = np.array([], dtype=float)
        collision_force_state.normal_force_magnitude = np.array([], dtype=float)
        collision_force_state.shear_force_magnitude = np.array([], dtype=float)
        collision_force_state.total_force_magnitude = np.array([], dtype=float)

    delta_time_rotation = delta_time
    host_ids, host_radii, host_angular_velocity = build_host_rotation_data(
        deck,
        timestep_index,
        host_type_metadata,
        particle_metadata,
    )
    host_omega_magnitude = angular_speed_magnitude(host_angular_velocity)
    host_rotational_distance_dt = host_radii * host_omega_magnitude * delta_time_rotation
    host_rotational_distance_total_current_in_contact = []
    host_contact_time_total_current_in_contact = []

    if host_ids.size:
        host_indices, host_valid = map_particle_ids_to_indices(sort_order, sorted_particle_ids, host_ids)
        host_contact_xi = np.zeros(host_ids.shape, dtype=int)

        if np.any(host_valid):
            valid_host_indices = host_indices[host_valid]
            host_contact_xi[host_valid] = (collision_number_per_particle[valid_host_indices] > 0).astype(int)
            extracted["rotational_distance_contact_xi_per_particle"][valid_host_indices] = host_contact_xi[host_valid]
            extracted["rotational_distance_per_particle"][valid_host_indices] = host_rotational_distance_dt[host_valid]

            for host_id, rotational_distance_dt, xi_contact in zip(
                host_ids[host_valid],
                host_rotational_distance_dt[host_valid],
                host_contact_xi[host_valid],
            ):
                previous_total = host_rotational_distance_total_by_id.get(int(host_id), 0.0)
                previous_contact_time = host_contact_time_total_by_id.get(int(host_id), 0.0)
                
                if xi_contact:
                    host_rotational_distance_total_by_id[int(host_id)] = previous_total + float(rotational_distance_dt)
                    updated_contact_time = previous_contact_time + delta_time_rotation
                    host_contact_time_total_by_id[int(host_id)] = updated_contact_time
                    host_rotational_distance_total_current_in_contact.append(host_rotational_distance_total_by_id[int(host_id)])
                    host_contact_time_total_current_in_contact.append(updated_contact_time)

            host_current_total = np.asarray(
                [host_rotational_distance_total_by_id.get(int(host_id), 0.0) for host_id in host_ids],
                dtype=float,
            )
            extracted["rotational_distance_total_per_particle"][valid_host_indices] = host_current_total[host_valid]

        summary["Ave_rotational_distance_per_host_particle"] = safe_mean(host_rotational_distance_dt)
        summary["Ave_rotational_distance_total_per_host_particle"] = safe_mean(
            np.asarray(host_rotational_distance_total_current_in_contact, dtype=float)
            if host_rotational_distance_total_current_in_contact else np.array([])
        )
        summary["Ave_cumulative_contact_time_per_host_particle"] = safe_mean(
            np.asarray(host_contact_time_total_current_in_contact, dtype=float)
            if host_contact_time_total_current_in_contact else np.array([])
        )

    return summary, extracted, collision_extracted

def process_dem_file(dem_path: Path, settings: StressSettings) -> None:
    print("-------------------------------------------------------")
    print(f"Loading particle-based analysis: {dem_path.name}")
    print("-------------------------------------------------------")

    deck = Deck(str(dem_path))
    timestep_values = np.asarray(deck.timestepValues, dtype=float)
    start_tstep = find_nearest(timestep_values, settings.start_time)
    end_tstep = find_nearest(timestep_values, settings.end_time)
    record_tstep = find_nearest(timestep_values, settings.extract_data_time)
    if start_tstep > end_tstep:
        start_tstep, end_tstep = end_tstep, start_tstep
    particle_metadata = build_particle_metadata(deck, start_tstep)
    host_type_metadata = resolve_host_type_metadata(deck, start_tstep, settings.host_particle_material)

    print("start_time:", settings.start_time)
    print("end_time:", settings.end_time)
    print("isCalStressTensor:", settings.calc_stress_tensor)
    print("isCalStressIntensity:", settings.calc_stress_intensity)
    print("isCalAveColnum:", settings.calc_average_collision_number)
    print("isCalColforce:", settings.calc_collision_force)
    print("plots:", settings.generate_plots)
    print("hostParMaterial:", settings.host_particle_material)
    print("extract_data_time:", settings.extract_data_time)
    print("high_impact_threshold:", settings.high_impact_threshold)
    print("-------------------------------------------------------")
    print(f"Processing particle-based metrics: {dem_path.name}")
    print("-------------------------------------------------------")

    data_csv_particle: list[dict[str, Any]] = []
    data_extract_particle: list[dict[str, Any]] = []
    host_rotational_distance_total_by_id: dict[int, float] = {}
    host_contact_time_total_by_id: dict[int, float] = {}
    collision_force_state = empty_collision_force_state()
    total_timesteps = end_tstep - start_tstep + 1
    progress_interval = max(1, min(50, total_timesteps))

    for timestep_index in range(start_tstep, end_tstep + 1):
        processed_count = timestep_index - start_tstep
        if (
            processed_count == 0
            or timestep_index == end_tstep
            or processed_count % progress_interval == 0
        ):
            print(
                f"Particle timestep {timestep_index} / {end_tstep} "
                f"(time={timestep_values[timestep_index]:.6f} s, step {processed_count + 1}/{total_timesteps})"
            )
        summary, particle_extracted, _collision_extracted = analyze_timestep(
            deck,
            timestep_index,
            timestep_values,
            settings,
            particle_metadata,
            host_type_metadata,
            host_rotational_distance_total_by_id,
            host_contact_time_total_by_id,
            collision_force_state,
        )
        data_csv_particle.append({
            "time": summary["time"],
            "number_of_collisions": summary["number_of_collisions"],
            "Ave_number_of_collisions_per_particle": summary["Ave_number_of_collisions_per_particle"],
            "Ave_contact_duration_per_collision": summary["Ave_contact_duration_per_collision"],
            "total_collision_time": summary["total_collision_time"],
            "average_collision_time": summary["average_collision_time"],
            "collision_frequency": summary["collision_frequency"],
            "Ave_TotalColForce_inSys_byParticle": summary["Ave_TotalColForce_inSys_byParticle"],
            "high_impact_event_fraction": summary["high_impact_event_fraction"],
            "Ave_SI_normal_per_particle": summary["Ave_SI_normal_per_particle"],
            "Ave_SI_shear_per_particle": summary["Ave_SI_shear_per_particle"],
            "Ave_SI_total_per_particle": summary["Ave_SI_total_per_particle"],
            "Total_normal_specific_energy_by_particle": summary["Total_normal_specific_energy_by_particle"],
            "Total_shear_specific_energy_by_particle": summary["Total_shear_specific_energy_by_particle"],
            "Total_specific_energy_by_particle": summary["Total_specific_energy_by_particle"],
            "Ave_normal_stress_value_per_particle": summary["Ave_normal_stress_value_per_particle"],
            "Ave_shear_stress_value_per_particle": summary["Ave_shear_stress_value_per_particle"],
            "Ave_stress_value_per_particle": summary["Ave_stress_value_per_particle"],
            "Ave_normal_power_per_particle": summary["Ave_normal_power_per_particle"],
            "Ave_shear_power_per_particle": summary["Ave_shear_power_per_particle"],
            "Ave_power_per_particle": summary["Ave_power_per_particle"],
            "Total_normal_power_by_particle": summary["Total_normal_power_by_particle"],
            "Total_shear_power_by_particle": summary["Total_shear_power_by_particle"],
            "Total_power_by_particle": summary["Total_power_by_particle"],
            "Ave_rotational_distance_per_host_particle": summary["Ave_rotational_distance_per_host_particle"],
            "Ave_rotational_distance_total_per_host_particle": summary["Ave_rotational_distance_total_per_host_particle"],
        })

        if timestep_index == record_tstep and particle_extracted is not None:
            data_extract_particle.extend(expand_row_to_table(particle_extracted))

    extract_time_text = format_time_for_filename(settings.extract_data_time)

    if data_extract_particle:
        write_csv(dem_path.with_name(f"{dem_path.stem}_particle_extracted_at_{extract_time_text}.csv"), data_extract_particle)

    if data_csv_particle:
        output_file_summary_particle = dem_path.with_name(f"{dem_path.stem}_particle_over_time.csv")
        write_csv(output_file_summary_particle, data_csv_particle)
        print(f"Particle-based data saved to {output_file_summary_particle}")


def main() -> None:
    workspace = Path.cwd()
    settings_path = workspace / "Stress_settings.txt"
    if not settings_path.exists():
        raise FileNotFoundError(f"Could not find {settings_path}")

    settings = load_settings(settings_path)
    dem_files = sorted(workspace.rglob("*.dem"))
    if not dem_files:
        raise FileNotFoundError(f"No .dem files found under {workspace}")

    for dem_path in dem_files:
        process_dem_file(dem_path, settings)


if __name__ == "__main__":
    main()
