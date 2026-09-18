"""Process-pool parallelism for independent repeats, shared by bo/ and
bmoo/. Each repeat (seed) only reads a candidate grid built before the pool starts, so results don't depend on how many workers run them.

Author: Titouan Brunel, Camille Cesari, Laure Leblond, Valentin Roux
"""
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def slurm_cpu_budget():
    """CPUs actually granted to this job (Slurm --cpus-per-task), falling
    back to the local core count outside Slurm."""
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(value) if value else (os.cpu_count() or 1)


def _pin_single_threaded():
    """Pool initializer: each worker gets exactly 1 BLAS/torch thread, so
    n_workers processes together use exactly n_workers cores instead of n_workers x (the multi-threaded budget the main process was given)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import torch
    torch.set_num_threads(1)


def worker_pool(n_repeats, cpus=None):
    """Process pool sized to min(cpu budget, n_repeats) -- never more
    workers than the job's Slurm allocation, and never more than there is independent work (one process per repeat seed). Uses 'spawn' (not the platform-default 'fork' on Linux) so _pin_single_threaded's env vars are read by a fresh interpreter before numpy/torch initialize their thread pools -- a fork would inherit the parent's already-initialized, already multi-threaded BLAS state instead."""
    cpus = cpus if cpus is not None else slurm_cpu_budget()
    n_workers = max(1, min(cpus, n_repeats))
    return ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_pin_single_threaded,
    )
