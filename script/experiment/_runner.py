
"""
Run selected sweeps, retrying failed experiments in fresh processes.
"""

# %% Imports
import multiprocessing as mp
import os
from datetime import timedelta
from time import perf_counter

from tqdm import tqdm

from config import (
    ACTIVE_SWEEP,
    EXPERIMENT_SWEEPS,
    MAX_ATTEMPTS,
    PROJECT_ROOT,
    message,
    sweep_experiments,
)

# %% Sweep resolution
def _get_sweep_ids() -> list[str]:
    """Resolve the active sweep selection."""
    if ACTIVE_SWEEP == "all":
        return list(EXPERIMENT_SWEEPS)
    elif isinstance(ACTIVE_SWEEP, str):
        return [ACTIVE_SWEEP]
    elif isinstance(ACTIVE_SWEEP, list):
        return ACTIVE_SWEEP

# %% Experiment execution
def _ignore_warning():
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    
def _run_experiment(experiment: dict, progress_lock):
    """Run an experiment in an isolated worker process."""
    tqdm.set_lock(progress_lock)
    _ignore_warning()

    message(f"Worker PID: {os.getpid()}", flush=True)
    
    from script.experiment.experiment import run_experiment
    run_experiment(experiment)

def _run_with_retries(experiment: dict, context, progress_lock):
    """Run an experiment with retries and record its completion time."""
    result_path = (
        PROJECT_ROOT / "results"
        / experiment["protocol_id"] / experiment["experiment_id"]
        / "results.csv"
    )
    already_completed = result_path.exists()
    start_time = perf_counter()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        process = context.Process(
            target=_run_experiment,
            args=(experiment, progress_lock),
        )
        process.start()
        process.join()

        if process.exitcode == 0:
            if not already_completed:
                elapsed = timedelta(seconds=int(perf_counter() - start_time))
                tqdm.write(f"Completed: {experiment['experiment_id']} | {elapsed}")
            return

        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(
                f"Experiment failed after {MAX_ATTEMPTS} attempts: "
                f"{experiment['experiment_id']} "
                f"(exit code: {process.exitcode})"
            )

        tqdm.write(
            f"Retrying {experiment['experiment_id']} "
            f"({attempt + 1}/{MAX_ATTEMPTS}) "
            f"after worker exit code {process.exitcode}"
        )


# %% Main
def main():
    """Run all selected sweeps."""
    context = mp.get_context("spawn")
    progress_lock = context.RLock()
    tqdm.set_lock(progress_lock)

    for sweep_id in _get_sweep_ids():
        experiments = list(sweep_experiments(sweep_id))
    
        for experiment in tqdm(
            experiments,
            desc=f"{sweep_id}",
            unit="experiment",
            position=0,
            leave=True,
        ):
            _run_with_retries(experiment, context, progress_lock)


if __name__ == "__main__":
    main()
