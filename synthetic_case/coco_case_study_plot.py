import os
import re
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats


USERNAME = "cluster_username"
HOST = "cluster_login_node"
REMOTE_DATA_DIR = f"path/to/remote/COCO/data/"
LOCAL_DATA_DIR = "path/to/local/COCO/data/"
OUTPUT_PDF_DIR = f"path/to/pdf_plots/"

RUN_NAME_PATTERN = re.compile(r"^(?P<optimizer>.+)_(?P<function>f\d+)_i(?P<instance>\d+)(?:_(?P<dim>\d+)D)?$")


def parse_run_metadata(name):
    """Extract optimizer, function, instance and optional dimension from a run name."""
    match = RUN_NAME_PATTERN.match(name)
    if not match:
        return None
    return {
        "optimizer": match.group("optimizer"),
        "function": match.group("function"),
        "instance": int(match.group("instance")),
        "dim": int(match.group("dim")) if match.group("dim") is not None else None,
    }


def function_index(function_name):
    """Return the numeric part of a function name like `f15` for sorting."""
    match = re.search(r"f(\d+)", function_name)
    return int(match.group(1)) if match else None


def extract_score_series(scores_dict):
    """Return scores ordered by evaluation index."""
    indices = sorted(int(k) for k in scores_dict.keys())
    return [scores_dict[str(i)] for i in indices]


def build_mono_distance_curve(scores_dict, optimum_value):
    """Build the cumulative distance-from-optimum curve for a single-objective run."""
    values = extract_score_series(scores_dict)
    best_so_far = np.minimum.accumulate(np.asarray(values, dtype=float))
    return np.maximum(best_so_far - optimum_value, 0.0)


def build_multiobjective_series(scores_dict):
    """Extract a bi-objective run into a 2D point array and per-objective min-so-far curves."""
    values = extract_score_series(scores_dict)
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Multiobjective scores must contain exactly two values per evaluation.")
    return np.minimum.accumulate(points[:, 0]), np.minimum.accumulate(points[:, 1])


def build_multiobjective_per_objective_distance_curves(objective_1_curve, objective_2_curve, ideal_point):
    """Convert objective-wise min-so-far curves into distances from known minima."""
    return (
        np.maximum(objective_1_curve - ideal_point[0], 0.0),
        np.maximum(objective_2_curve - ideal_point[1], 0.0),
    )

def mean_and_sem_curves(curves):
    """Pad curves to the max length using forward fill, then calculate mean and SEM."""
    if not curves:
        return None, None
    
    max_len = max(len(c) for c in curves)
    padded_curves = []
    
    for c in curves:
        c_array = np.asarray(c, dtype=float)
        if len(c_array) < max_len:
            # Padding last value to the longest run
            padded = np.pad(c_array, (0, max_len - len(c_array)), mode='edge')
            padded_curves.append(padded)
        else:
            padded_curves.append(c_array)
            
    padded_matrix = np.array(padded_curves)
    mean_curve = np.mean(padded_matrix, axis=0)
    sem_curve = stats.sem(padded_matrix, axis=0) # Using std error of mean instead of std deviation
    
    return mean_curve, sem_curve

def get_problem_optimum_value(run_data):
    """Extract the known best objective value from the JSON payload."""
    best_value = (run_data.get("problem_optimum") or {}).get("best_value")
    if best_value is None:
        return None
    
    if isinstance(best_value, (list, tuple, np.ndarray)):
        return np.asarray(best_value, dtype=float) if len(best_value) == 2 else None
    
    return float(best_value)

def build_chebyshev_curve(obj1_dist, obj2_dist):
    """Compute the normalized Chebyshev distance curve to combine both objectives."""
    norm1 = np.max(obj1_dist)
    norm2 = np.max(obj2_dist)
    scale1 = norm1 if norm1 > 1e-8 else 1.0
    scale2 = norm2 if norm2 > 1e-8 else 1.0
    
    return np.maximum(obj1_dist / scale1, obj2_dist / scale2)

def fetch_data_from_cluster():
    """Fetch JSON results from the cluster into the local data folder."""
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    rsync_cmd = ["rsync", "-avz", f"{USERNAME}@{HOST}:{REMOTE_DATA_DIR}*.json", LOCAL_DATA_DIR]
    try:
        subprocess.run(rsync_cmd, check=True)
        print("Download completed!\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while downloading!\n{e}")
        exit(1)


def plot_mean_convergence(include_optimizers=None, exclude_optimizers=None):
    """Read all local JSON files and generate one PDF per function."""
    os.makedirs(OUTPUT_PDF_DIR, exist_ok=True)

    runs_by_function_dim_optimizer = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for filename in os.listdir(LOCAL_DATA_DIR):

        file_path = os.path.join(LOCAL_DATA_DIR, filename)

        if os.path.getsize(file_path) == 0:
            print(f"Warning: skipped empty file (0 byte) -> {filename}")
            continue

        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: failed JSON decoding -> {filename}")
                continue

        metadata = parse_run_metadata(data["name"])
        if metadata is None:
            print(f"Warning: run name not valid, skipping -> {filename}")
            continue

        if "scores" not in data or not data["scores"]:
            print(f"Warning: skipping data with missing scores -> {filename}")
            continue

        dim = data.get("dim", metadata["dim"])
        if dim is None:
            print(f"Warning: dimension not available, skipping -> {filename}")
            continue

        function_name = metadata["function"]
        optimizer_name = metadata["optimizer"]

        if include_optimizers is not None and optimizer_name not in include_optimizers:
            continue
        if exclude_optimizers is not None and optimizer_name in exclude_optimizers:
            continue

        if bool(data["is_moo"]):
            obj1_curve, obj2_curve = build_multiobjective_series(data["scores"])
            optimum_value = get_problem_optimum_value(data)

            if not isinstance(optimum_value, np.ndarray):
                print(f"Warning: bi-objective optimum not valid, skipping -> {filename}")
                continue
            
            obj1_dist, obj2_dist = build_multiobjective_per_objective_distance_curves(
                obj1_curve, obj2_curve, optimum_value
            )

            cheb_curve = build_chebyshev_curve(obj1_dist, obj2_dist)

            runs_by_function_dim_optimizer[function_name][dim][optimizer_name].append({
                "kind": "moo",
                "objective_1_distance_curve": obj1_dist,
                "objective_2_distance_curve": obj2_dist,
                "chebyshev_distance_curve": cheb_curve,
            })
        else:
            optimum_value = get_problem_optimum_value(data)

            if not isinstance(optimum_value, float):
                print(f"Warning: missing problem_optimum, skipping -> {filename}")
                continue

            runs_by_function_dim_optimizer[function_name][dim][optimizer_name].append({
                "kind": "soo",
                "curve": build_mono_distance_curve(data["scores"], optimum_value),
            })

    if not runs_by_function_dim_optimizer:
        print("No data found for generating plots.")
        return

    for function_name in sorted(runs_by_function_dim_optimizer.keys(),
                                key=lambda n: (0, function_index(n)) if function_index(n) is not None else (1, n)):
        dims = sorted(runs_by_function_dim_optimizer[function_name].keys())

        # Using first run to get if moo or soo
        sample_run = None
        for dim in dims:
            for optimizer_runs in runs_by_function_dim_optimizer[function_name][dim].values():
                if optimizer_runs:
                    sample_run = optimizer_runs[0]
                    break
            if sample_run is not None:
                break

        is_moo_function = sample_run is not None and sample_run["kind"] == "moo"
        metrics = ["objective_1_distance", "objective_2_distance", "chebyshev_distance"] if is_moo_function else ["score"]


        fig, axes = plt.subplots(
            nrows=len(metrics),
            ncols=len(dims),
            figsize=(10 * len(dims), 6 * len(metrics)),
            squeeze=False,
        )

        if len(metrics) > 1:
            fig.suptitle(function_name, fontsize=16, fontweight="bold")

        for dim_index, dim in enumerate(dims):
            optimizer_runs = runs_by_function_dim_optimizer[function_name][dim]

            for optimizer_name in sorted(optimizer_runs.keys()):
                runs = optimizer_runs[optimizer_name]
                metric_curves = {metric: [] for metric in metrics} # Dictionary with metrics as keys and curves as values

                for run in runs:
                    if run["kind"] == "moo":
                        metric_curves["objective_1_distance"].append(run["objective_1_distance_curve"])
                        metric_curves["objective_2_distance"].append(run["objective_2_distance_curve"])
                        metric_curves["chebyshev_distance"].append(run["chebyshev_distance_curve"])
                    else:
                        metric_curves["score"].append(run["curve"])

                for metric_index, metric_name in enumerate(metrics):
                    mean_curve, sem_curve = mean_and_sem_curves(metric_curves[metric_name])

                    if mean_curve is None:
                        continue

                    # Threshold to avoid zeros
                    MIN_DISTANCE = 1e-8
                    mean_curve = np.clip(mean_curve, a_min=MIN_DISTANCE, a_max=None)

                    ax = axes[metric_index, dim_index]
                    x_axis = np.arange(len(mean_curve))
                    
                    ax.plot(x_axis, mean_curve, label=optimizer_name.replace("_", " ").upper(), linewidth=2.5)
                    
                    lower_bound = mean_curve - sem_curve
                    lower_bound = np.clip(lower_bound, a_min=MIN_DISTANCE, a_max=None)
                    
                    ax.fill_between(x_axis, lower_bound, mean_curve + sem_curve, alpha=0.15)

            for metric_index, metric_name in enumerate(metrics):
                ax = axes[metric_index, dim_index]
                
                ax.set_yscale("log")

                ylabel_map = {
                    "score": "Mean Distance from Optimum",
                    "objective_1_distance": "Mean Distance on Objective 1",
                    "objective_2_distance": "Mean Distance on Objective 2",
                    "chebyshev_distance": "Mean Normalized Chebyshev Distance",
                }
                
                ax.set_title(
                    f"{function_name} - {dim}D" if len(metrics) == 1
                    else f"{dim}D - {metric_name.replace('_', ' ').title()}",
                    fontsize=14 if len(metrics) == 1 else 13,
                    fontweight="bold",
                )
                ax.set_xlabel("Number of Evaluations (Trials)", fontsize=12)
                ax.set_ylabel(ylabel_map.get(metric_name, "Mean of Minimum Values"), fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        if len(metrics) > 1:
            plt.subplots_adjust(top=0.94)

        output_file = os.path.join(OUTPUT_PDF_DIR, f"mean_convergence_{function_name}.pdf")
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot generated: mean_convergence_{function_name}.pdf")


if __name__ == "__main__":
    choice = int(input("1) Get JSON from server and plot\n2) Plot from local JSON\nChoice: "))
    if choice == 1:
        fetch_data_from_cluster()
    plot_mean_convergence()