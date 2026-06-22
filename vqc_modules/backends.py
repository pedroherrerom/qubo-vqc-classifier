"""CUNQA and Polypus backend execution handlers and utilities."""

import concurrent.futures
import json
import logging
import os
import time

import polypus

LUSTRE_QPUS = "/mnt/lustre/scratch/nlsas/home/uvi/et/phm/.cunqa/.cunqa/qpus.json" # TODO: Move to config file or environment variable


def extract_counts(result) -> dict:
    """Normalize a Polypus execution result into a flat counts dictionary.

    Processes raw results from a Polypus execution and consolidates them into a 
    uniform format. Seamlessly handles direct dictionaries, single-element lists, 
    and multi-element lists by aggregating (summing) the counts of matching bit 
    keys across all items.

    Args:
        result (list | dict): The raw Polypus result.

    Returns:
        dict: A flat dictionary mapping bit configurations to total accumulated counts.

    Raises:
        ValueError: If the input result is neither a list nor a dictionary.
    """
    if isinstance(result, list):
        if len(result) == 1:
            return dict(result[0])
        merged = {}
        for item in result:
            for bits, count in dict(item).items():
                merged[bits] = merged.get(bits, 0) + count
        return merged
    if isinstance(result, dict):
        return result
    raise ValueError(f"Unexpected polypus result type: {type(result)}")


def resolve_qpus(requested: int, logger: logging.Logger) -> int:
    """Forward the requested QPU count to the Polypus training manager.

    Polypus manages its own QPU provisioning internally when using 
    train_infrastructure='cunqa' (via qraise). Evaluation-side QPU validation 
    is handled independently by get_cunqa_qpus() via the CUNQA Python API.
    """
    logger.info("resolve_qpus | Forwarding n_qpus=%d to polypus.qml.train (eval-side QPUs validated separately).", requested)
    return requested


def get_cunqa_qpus(actual_qpus: int, logger: logging.Logger):
    """Connect to pre-provisioned QPUs using the CUNQA Python API."""
    from cunqa.qpu import get_QPUs

    family = os.environ.get("POLYPUS_CUNQA_FAMILY", "")
    try:
        qpus = get_QPUs(family=family, co_located=True) if family else get_QPUs(co_located=True)
        if not qpus:
            logger.warning("get_QPUs() returned an empty list; falling back to local Polypus backend.")
            return None
        qpus = qpus[:actual_qpus]
        logger.info("Connected to %d pre-raised CUNQA QPU(s) (family='%s').", len(qpus), family or "any")
        return qpus
    except Exception as exc:
        logger.warning("CUNQA get_QPUs() failed (%s); falling back to local Polypus backend.", exc)
        return None


def run_batch(
    circuits: list,
    qpus: list,
    num_shots: int,
    chunk_timeout_s: int | None=None,
    *,
    timeout_per_circuit_s: float=30.0,
    min_timeout_s: int=120,
    max_timeout_s: int=1800,
    logger=None,
    ) -> list:
    """Execute a list of quantum circuits across available CUNQA QPUs in batches.

    Circuits are partitioned into chunks matching the number of available QPUs. 
    Each chunk is executed concurrently, and a timeout is enforced either via 
    an explicit override or calculated dynamically based on the chunk size.

    Args:
        circuits (list): List of quantum circuits to execute.
        qpus (list): List of available CUNQA QPUs.
        num_shots (int): Execution shots per circuit.
        chunk_timeout_s (int | None): Explicit timeout in seconds per chunk.
        timeout_per_circuit_s (float): Time budget per circuit for dynamic timeout.
        min_timeout_s (int): Minimum dynamic timeout bounds.
        max_timeout_s (int): Maximum dynamic timeout bounds.
        logger (logging.Logger | None): Standard logger instance.

    Returns:
        list: A flat list of execution results corresponding to the input circuits.

    Raises:
        RuntimeError: If gathering the results for any chunk exceeds the timeout.
    """
    import time as _time
    from cunqa.qjob import gather
    from cunqa.qpu import run as cunqa_run

    n_qpus = len(qpus)
    # Calculate total chunks to allow for accurate progress tracking
    n_chunks = (len(circuits) + n_qpus - 1) // n_qpus
    results = []
    t_batch = _time.time()

    for start in range(0, len(circuits), n_qpus):
        chunk = circuits[start:start + n_qpus]
        qpu_chunk = qpus[:len(chunk)]

        # Determine timeout: use explicit override if provided, otherwise scale by chunk size
        if chunk_timeout_s is not None:
            effective_timeout = chunk_timeout_s
        else:
            effective_timeout = int(max(min_timeout_s, min(len(chunk) * timeout_per_circuit_s, max_timeout_s)))

        chunk_number = start // n_qpus + 1
        t_chunk = _time.time()

        qjobs = cunqa_run(chunk, qpu_chunk, shots=num_shots)
        
        # Normalize single job returns into a list for consistent gathering
        if not isinstance(qjobs, list):
            qjobs = [qjobs]

        # Enforce network gather timeout using a thread to prevent silent hangs
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(gather, qjobs)
            try:
                chunk_results = future.result(timeout=effective_timeout)
            except concurrent.futures.TimeoutError as exc:
                if logger:
                    logger.error(
                        "CUNQA gather() timed out (chunk %d/%d, %d circuits, timeout=%ds).",
                        chunk_number, n_chunks, len(chunk), effective_timeout,
                    )
                raise RuntimeError(
                    f"CUNQA gather() timed out after {effective_timeout}s "
                    f"(chunk {chunk_number}/{n_chunks}, {len(chunk)} circuits, {num_shots} shots)."
                ) from exc

        results.extend(extract_counts(r.counts) for r in chunk_results)
                
        # Periodically report progress and estimated completion time
        if logger and (chunk_number % 10 == 0 or chunk_number == n_chunks):
            elapsed = _time.time() - t_batch
            rate = (start + len(chunk)) / max(elapsed, 1e-9)
            eta = (len(circuits) - start - len(chunk)) / max(rate, 1e-9)
            
            logger.info(
                "run_batch | chunk %d/%d done in %.2fs | total %.1fs | ETA %.1fs",
                chunk_number, n_chunks, _time.time() - t_chunk, elapsed, eta,
            )
            
    return results


def run_batch_local(circuits: list, num_shots: int, *, use_gpu: bool=False, parallelize: bool=True, logger=None) -> list:
    """Execute circuits on the local Aer simulator with configurable hardware targets.

    Args:
        circuits (list): Bound QuantumCircuit objects ready for execution.
        num_shots (int): Number of measurement shots per circuit.
        use_gpu (bool): Enables device='GPU' in AerSimulator.
        parallelize (bool): Triggers full CPU thread utilization (max_parallel_experiments=0).
        logger (logging.Logger | None): Standard logger instance.

    Returns:
        list[dict]: Output counts dictionary per input circuit.
    """
    import time as _time
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    
    t0 = _time.time()
    opts = {"max_parallel_experiments": 0 if parallelize else 1}
    if use_gpu:
        opts["device"] = "GPU"
        
    sim = AerSimulator(**opts)
    tqc = transpile(circuits, sim)
    result = sim.run(tqc, shots=num_shots).result()
    counts_list = [result.get_counts(i) for i in range(len(circuits))]
    
    return counts_list


def run_batch_aer(circuits: list, num_shots: int, logger=None) -> list:
    """Batched local Aer execution (parallel CPU).
    
    .. deprecated::
        Superseded by run_batch_local(parallelize=True). Maintained strictly for 
        backward compatibility with external scripts.
    """
    import time as _time
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    t0 = _time.time()
    sim = AerSimulator()
    tqc = transpile(circuits, sim)
    result = sim.run(tqc, shots=num_shots).result()
    counts_list = [result.get_counts(i) for i in range(len(circuits))]
    
    if logger:
        logger.info(
            "run_batch_aer | %d circuits | %d shots | %.2f s (%.1f circ/s)",
            len(circuits), num_shots, _time.time() - t0,
            len(circuits) / max(_time.time() - t0, 1e-9),
        )
    return counts_list