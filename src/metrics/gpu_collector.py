"""Background GPU metrics collector using pynvml with nvidia-smi fallback."""

import csv
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GpuSnapshot:
    timestamp: float
    gpu_index: int
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_draw_w: float
    sm_clock_mhz: float


class GpuCollector:
    """Polls GPU metrics at a fixed interval in a background thread."""

    def __init__(self, interval_sec: float = 0.5, gpu_index: int = 0):
        self.interval_sec = interval_sec
        self.gpu_index = gpu_index
        self.snapshots: list[GpuSnapshot] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_available = False
        self._handle = None

    def _init_nvml(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml_available = True
            logger.info("pynvml initialized for GPU %d", self.gpu_index)
            return True
        except Exception as e:
            logger.warning("pynvml unavailable (%s), falling back to nvidia-smi", e)
            return False

    def _read_nvml(self) -> GpuSnapshot | None:
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW -> W
            clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
            return GpuSnapshot(
                timestamp=time.time(),
                gpu_index=self.gpu_index,
                utilization_pct=util.gpu,
                memory_used_mb=mem.used / (1024 ** 2),
                memory_total_mb=mem.total / (1024 ** 2),
                temperature_c=temp,
                power_draw_w=power,
                sm_clock_mhz=clock,
            )
        except Exception as e:
            logger.debug("pynvml read error: %s", e)
            return None

    def _read_nvidia_smi(self) -> GpuSnapshot | None:
        fields = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm"
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}",
                 f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            vals = [v.strip() for v in result.stdout.strip().split(",")]
            return GpuSnapshot(
                timestamp=time.time(),
                gpu_index=self.gpu_index,
                utilization_pct=float(vals[0]),
                memory_used_mb=float(vals[1]),
                memory_total_mb=float(vals[2]),
                temperature_c=float(vals[3]),
                power_draw_w=float(vals[4]),
                sm_clock_mhz=float(vals[5]),
            )
        except Exception as e:
            logger.debug("nvidia-smi read error: %s", e)
            return None

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            snap = None
            if self._nvml_available:
                snap = self._read_nvml()
            if snap is None:
                snap = self._read_nvidia_smi()
            if snap is not None:
                self.snapshots.append(snap)
            self._stop_event.wait(self.interval_sec)

    def start(self) -> None:
        self._init_nvml()
        self.snapshots.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("GPU collector started (interval=%.1fs)", self.interval_sec)

    def stop(self) -> list[GpuSnapshot]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
        logger.info("GPU collector stopped, %d snapshots captured", len(self.snapshots))
        return self.snapshots

    def save_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(GpuSnapshot.__dataclass_fields__))
            writer.writeheader()
            for snap in self.snapshots:
                writer.writerow(asdict(snap))
        logger.info("GPU metrics saved to %s (%d rows)", path, len(self.snapshots))
        return path
