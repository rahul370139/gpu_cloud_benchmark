"""Historical benchmark store — SQLite backend for logging and querying past runs.

Every benchmark result (full or partial) is recorded so the system can:
  1. Reuse data instead of re-running expensive benchmarks.
  2. Power the similarity-based predictor for new, unseen workloads.
  3. Track GPU performance trends over time (driver updates, etc.).
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    gpu_type        TEXT    NOT NULL,
    workload        TEXT    NOT NULL,
    model_name      TEXT,
    param_count     INTEGER,
    mode            TEXT    NOT NULL,
    batch_size      INTEGER NOT NULL,
    throughput      REAL,
    throughput_unit TEXT,
    latency_p50_ms  REAL,
    latency_p95_ms  REAL,
    latency_p99_ms  REAL,
    avg_gpu_util    REAL,
    avg_gpu_mem_mb  REAL,
    cost_per_hour   REAL    DEFAULT 0,
    benchmark_iters INTEGER,
    seed            INTEGER,
    is_partial      INTEGER DEFAULT 0,
    confidence_low  REAL,
    confidence_high REAL,
    converged       INTEGER,
    run_source      TEXT    DEFAULT 'full'
);

CREATE TABLE IF NOT EXISTS recommendations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    query_workload  TEXT,
    query_mode      TEXT,
    query_batch_size INTEGER,
    constraints_json TEXT,
    recommended_gpu TEXT,
    composite_score REAL,
    reasoning       TEXT,
    all_scores_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_workload ON benchmark_runs(workload, mode, batch_size);
CREATE INDEX IF NOT EXISTS idx_runs_gpu      ON benchmark_runs(gpu_type);
"""


class HistoryStore:
    """Thin wrapper around an SQLite database for benchmark history."""

    def __init__(self, db_path: str | Path = "data/benchmark_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("History database ready: %s", self.db_path)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_run(
        self,
        gpu_type: str,
        workload: str,
        mode: str,
        batch_size: int,
        throughput: float,
        *,
        model_name: str = "",
        param_count: int = 0,
        throughput_unit: str = "",
        latency_p50_ms: float = 0,
        latency_p95_ms: float = 0,
        latency_p99_ms: float = 0,
        avg_gpu_util: float = 0,
        avg_gpu_mem_mb: float = 0,
        cost_per_hour: float = 0,
        benchmark_iters: int = 0,
        seed: int = 42,
        is_partial: bool = False,
        confidence_low: float = 0,
        confidence_high: float = 0,
        converged: bool = True,
        run_source: str = "full",
    ) -> int:
        """Insert a single benchmark run and return its row id."""
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """INSERT INTO benchmark_runs
               (timestamp, gpu_type, workload, model_name, param_count, mode,
                batch_size, throughput, throughput_unit,
                latency_p50_ms, latency_p95_ms, latency_p99_ms,
                avg_gpu_util, avg_gpu_mem_mb, cost_per_hour,
                benchmark_iters, seed, is_partial,
                confidence_low, confidence_high, converged, run_source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts, gpu_type, workload, model_name, param_count, mode,
                batch_size, throughput, throughput_unit,
                latency_p50_ms, latency_p95_ms, latency_p99_ms,
                avg_gpu_util, avg_gpu_mem_mb, cost_per_hour,
                benchmark_iters, seed, int(is_partial),
                confidence_low, confidence_high, int(converged), run_source,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_benchmark_results(
        self, results: list[dict], gpu_type: str, cost_per_hour: float = 0
    ) -> int:
        """Bulk-insert results from runner.run_full_benchmark()."""
        count = 0
        for r in results:
            if "error" in r:
                continue
            self.log_run(
                gpu_type=gpu_type,
                workload=r.get("workload", ""),
                mode=r.get("mode", ""),
                batch_size=r.get("batch_size", 0),
                throughput=r.get("throughput", 0),
                model_name=r.get("model_name", ""),
                param_count=r.get("param_count", 0),
                throughput_unit=r.get("throughput_unit", ""),
                latency_p50_ms=r.get("latency_p50_ms", 0),
                latency_p95_ms=r.get("latency_p95_ms", 0),
                latency_p99_ms=r.get("latency_p99_ms", 0),
                avg_gpu_util=r.get("avg_gpu_utilization_pct", 0),
                avg_gpu_mem_mb=r.get("avg_gpu_memory_used_mb", 0),
                cost_per_hour=cost_per_hour,
                benchmark_iters=r.get("benchmark_iters", 0),
                seed=r.get("seed", 42),
                run_source="full",
            )
            count += 1
        logger.info("Logged %d runs to history (gpu=%s)", count, gpu_type)
        return count

    def log_partial_result(self, pr, cost_per_hour: float = 0) -> int:
        """Log a PartialResult dataclass."""
        return self.log_run(
            gpu_type=pr.gpu_type,
            workload=pr.workload,
            mode=pr.mode,
            batch_size=pr.batch_size,
            throughput=pr.estimated_throughput,
            throughput_unit=pr.throughput_unit,
            param_count=pr.param_count,
            latency_p50_ms=pr.latency_p50_ms,
            latency_p95_ms=pr.latency_p95_ms,
            latency_p99_ms=pr.latency_p99_ms,
            avg_gpu_util=pr.avg_gpu_util_pct,
            avg_gpu_mem_mb=pr.avg_gpu_mem_mb,
            cost_per_hour=cost_per_hour,
            benchmark_iters=pr.iterations_run,
            is_partial=True,
            confidence_low=pr.confidence_low,
            confidence_high=pr.confidence_high,
            converged=pr.converged,
            run_source="partial",
        )

    def log_recommendation(
        self,
        workload: str,
        mode: str,
        batch_size: int,
        constraints_json: str,
        recommended_gpu: str,
        composite_score: float,
        reasoning: str,
        all_scores_json: str,
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """INSERT INTO recommendations
               (timestamp, query_workload, query_mode, query_batch_size,
                constraints_json, recommended_gpu, composite_score,
                reasoning, all_scores_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ts, workload, mode, batch_size, constraints_json,
             recommended_gpu, composite_score, reasoning, all_scores_json),
        )
        self._conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_runs(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM benchmark_runs ORDER BY timestamp DESC", self._conn)

    def get_runs_for_workload(self, workload: str, mode: str = "") -> pd.DataFrame:
        if mode:
            return pd.read_sql_query(
                "SELECT * FROM benchmark_runs WHERE workload=? AND mode=? ORDER BY timestamp DESC",
                self._conn, params=(workload, mode),
            )
        return pd.read_sql_query(
            "SELECT * FROM benchmark_runs WHERE workload=? ORDER BY timestamp DESC",
            self._conn, params=(workload,),
        )

    def get_distinct_gpus(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT gpu_type FROM benchmark_runs").fetchall()
        return [r["gpu_type"] for r in rows]

    def get_distinct_workloads(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT workload FROM benchmark_runs").fetchall()
        return [r["workload"] for r in rows]

    def get_run_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM benchmark_runs").fetchone()
        return row["c"]

    def get_latest_runs_per_gpu(
        self, workload: str, mode: str, batch_size: int
    ) -> pd.DataFrame:
        """Return the most recent run for each GPU type matching the query."""
        return pd.read_sql_query(
            """SELECT * FROM benchmark_runs
               WHERE workload=? AND mode=? AND batch_size=?
               GROUP BY gpu_type
               HAVING timestamp = MAX(timestamp)
               ORDER BY throughput DESC""",
            self._conn, params=(workload, mode, batch_size),
        )

    def summary_stats(self) -> dict:
        """Quick stats for display."""
        return {
            "total_runs": self.get_run_count(),
            "gpus_benchmarked": len(self.get_distinct_gpus()),
            "workloads_benchmarked": len(self.get_distinct_workloads()),
            "distinct_gpus": self.get_distinct_gpus(),
            "distinct_workloads": self.get_distinct_workloads(),
        }
