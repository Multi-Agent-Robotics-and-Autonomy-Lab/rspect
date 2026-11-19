"""GLNS (Smith & Imeson, 2017), the GTSP solver used in Step 2.2 of RSPECT.

GLNS is a Julia package. Install Julia (https://julialang.org), then GLNS::

    julia -e 'using Pkg; Pkg.add("GLNS")'

Julia compiles GLNS the first time it runs, which takes about 10 s. To pay this only
once, the module keeps a single Julia process alive for the whole Python session:
the process is started and warmed up on the first call to :func:`solve` and reused
afterwards, so later solves take milliseconds. Set the ``RSPECT_JULIA`` environment
variable if the ``julia`` executable is not on the PATH.
"""
from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

SCALE = 100       # GLNS needs integer weights: they are rounded to 1/SCALE of their unit
NO_EDGE = 10**9   # scaled weight of node pairs that are not connected


class GLNSError(RuntimeError):
    """Julia or GLNS is not available, or GLNS failed."""


@dataclass
class GTSPSolution:
    """A closed GTSP tour.

    Attributes:
        tour: Node indices (0-based), one per cluster, in cyclic order.
        cost: Tour cost, in the units of the weight matrix.
        solver_time: GLNS's own run time [s] (excludes Julia start-up and compilation).
    """

    tour: List[int]
    cost: float
    solver_time: float


def solve(weights: np.ndarray, clusters: Sequence[Sequence[int]], max_time: float, *,
          seed: int = 0) -> GTSPSolution:
    """Solve a symmetric GTSP with GLNS.

    Args:
        weights: ``(N, N)`` symmetric edge weights; ``np.inf`` marks a missing edge.
        clusters: Node indices (0-based) of each cluster; every node belongs to one cluster.
        max_time: GLNS time limit [s].
        seed: Seed of Julia's random number generator, for repeatable runs.
    """
    global _session
    if _session is None or not _session.alive:
        _session = _JuliaSession()
    solution = _session.solve(weights, clusters, max_time, seed)
    if solution.cost * SCALE >= NO_EDGE:
        raise GLNSError("GLNS returned a tour that uses a missing edge; increase its time limit")
    return solution


# ------------------------------------------------------------------------------ internals
_SERVER = """
using GLNS, Random
println("READY"); flush(stdout)
for line in eachline(stdin)
    f = split(line, '\\t')
    f[1] == "QUIT" && break
    Random.seed!(parse(Int, f[2]))
    solver(String(f[3]); output=String(f[4]), max_time=parse(Float64, f[5]), verbose=0)
    println("DONE"); flush(stdout)
end
"""

_WARMUP = (np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], dtype=float),
           [[0, 1], [2, 3]])

_session: Optional["_JuliaSession"] = None


class _JuliaSession:
    """A Julia process that runs GLNS on request (one tab-separated command per line)."""

    def __init__(self) -> None:
        julia = os.environ.get("RSPECT_JULIA", "julia")
        executable = shutil.which(julia) or (julia if os.path.isfile(julia) else None)
        if executable is None:
            raise GLNSError(f"Julia executable {julia!r} not found. Install Julia and GLNS "
                            "(julia -e 'using Pkg; Pkg.add(\"GLNS\")'), or set RSPECT_JULIA.")
        self._dir = tempfile.TemporaryDirectory(prefix="rspect_glns_")
        self._path = Path(self._dir.name)
        (self._path / "server.jl").write_text(_SERVER)
        self._log = open(self._path / "julia.log", "w+")
        self._proc = subprocess.Popen(
            [executable, "--startup-file=no", str(self._path / "server.jl")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1)
        self._count = 0
        atexit.register(self.close)
        self._wait_for("READY")
        self.solve(*_WARMUP, max_time=1.0, seed=0)  # compile GLNS once

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def solve(self, weights: np.ndarray, clusters: Sequence[Sequence[int]], max_time: float,
              seed: int) -> GTSPSolution:
        self._count += 1
        source = self._path / f"problem_{self._count}.gtsp"
        target = self._path / f"problem_{self._count}.out"
        _write_gtsplib(weights, clusters, source)
        self._proc.stdin.write(f"SOLVE\t{int(seed)}\t{source}\t{target}\t{float(max_time)!r}\n")
        self._proc.stdin.flush()
        self._wait_for("DONE")
        return _read_output(target)

    def close(self) -> None:
        if self.alive:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
        self._log.close()
        self._dir.cleanup()

    def _wait_for(self, token: str) -> None:
        while True:
            line = self._proc.stdout.readline()
            if not line:
                self._log.seek(0)
                log = self._log.read().strip()
                hint = "\nIs GLNS installed? julia -e 'using Pkg; Pkg.add(\"GLNS\")'" if "GLNS" in log else ""
                raise GLNSError(f"Julia stopped unexpectedly:\n{log}{hint}")
            if line.strip() == token:
                return


def _write_gtsplib(weights: np.ndarray, clusters: Sequence[Sequence[int]], path: Path) -> None:
    finite = np.isfinite(weights)
    scaled = np.where(finite, np.rint(np.where(finite, weights, 0.0) * SCALE), NO_EDGE).astype(np.int64)
    lines = ["NAME : rspect", "TYPE : GTSP", f"DIMENSION : {len(scaled)}",
             f"GTSP_SETS : {len(clusters)}", "EDGE_WEIGHT_TYPE : EXPLICIT",
             "EDGE_WEIGHT_FORMAT : FULL_MATRIX", "EDGE_WEIGHT_SECTION"]
    lines += [" ".join(map(str, row)) for row in scaled]
    lines.append("GTSP_SET_SECTION")
    lines += [f"{k} {' '.join(str(v + 1) for v in cluster)} -1"
              for k, cluster in enumerate(clusters, start=1)]
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n")


_PATTERNS = {key: re.compile(pattern) for key, pattern in {
    "time": r"Solver Time\s*:\s*([-+0-9.eE]+)",
    "cost": r"Tour Cost\s*:\s*([-+0-9.eE]+)",
    "tour": r"Tour\s*:\s*\[([^\]]*)\]",
}.items()}


def _read_output(path: Path) -> GTSPSolution:
    text = path.read_text()
    match = {key: pattern.search(text) for key, pattern in _PATTERNS.items()}
    if not all(match.values()):
        raise GLNSError(f"could not read the GLNS output:\n{text}")
    tour = [int(v) - 1 for v in match["tour"].group(1).replace(",", " ").split()]
    return GTSPSolution(tour, float(match["cost"].group(1)) / SCALE, float(match["time"].group(1)))
