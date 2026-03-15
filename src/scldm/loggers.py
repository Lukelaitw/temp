# Copyright (c) 2025. Custom CSV logger that merges metrics by step (one row per step).

from __future__ import annotations

import csv
import os
from typing import Any, Optional, Union

from torch import Tensor
from typing_extensions import override

from lightning.fabric.loggers.logger import Logger
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.logger import _add_prefix, _convert_params
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers.logger import Logger as PLLogger


def _to_scalar(value: Any) -> float:
    if isinstance(value, Tensor):
        return value.item()
    return float(value)


class _ExperimentWriterMergeByStep:
    """Writes metrics so that each step is exactly one row (merge multiple log calls per step)."""

    NAME_METRICS_FILE = "metrics.csv"
    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._fs = get_filesystem(log_dir)
        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)
        self._buffer: dict[int, dict[str, float]] = {}
        self._metrics_keys: list[str] = []
        self._fs.makedirs(self.log_dir, exist_ok=True)

    def log_metrics(self, metrics_dict: dict[str, Any], step: Optional[int] = None) -> None:
        if step is None:
            step = 0
        row = {k: _to_scalar(v) for k, v in metrics_dict.items()}
        row["step"] = step
        if step not in self._buffer:
            self._buffer[step] = {}
        self._buffer[step].update(row)
        for k in row:
            if k not in self._metrics_keys:
                self._metrics_keys.append(k)
                self._metrics_keys.sort()

    def save(self) -> None:
        if not self._buffer:
            return
        # Load existing rows from file (step -> row)
        steps_dict: dict[int, dict[str, float]] = {}
        file_exists = self._fs.isfile(self.metrics_file_path)
        if file_exists:
            with self._fs.open(self.metrics_file_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    s = r.get("step")
                    if s is not None and s != "":
                        try:
                            step = int(float(s))
                            steps_dict[step] = {k: v for k, v in r.items()}
                        except (ValueError, TypeError):
                            pass
        # Merge buffer into steps_dict (buffer wins)
        for step, row in self._buffer.items():
            if step not in steps_dict:
                steps_dict[step] = {}
            for k, v in row.items():
                if v is not None and (isinstance(v, str) and v != "") or (not isinstance(v, str)):
                    steps_dict[step][k] = v
        # Update global key list
        for row in steps_dict.values():
            for k in row:
                if k not in self._metrics_keys:
                    self._metrics_keys.append(k)
        self._metrics_keys.sort()
        # Write all rows in step order
        sorted_steps = sorted(steps_dict.keys())
        rows_to_write = [steps_dict[s] for s in sorted_steps]
        with self._fs.open(self.metrics_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._metrics_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_to_write)
        self._buffer.clear()


class CSVLoggerMergeByStep(PLLogger):
    """CSV logger that writes one row per step by merging all metrics for the same step.

    Drop-in replacement for :class:`pytorch_lightning.loggers.CSVLogger` with the same
    constructor args. Use this in config as:

        logger:
          csv:
            _target_: scldm.loggers.CSVLoggerMergeByStep
            save_dir: ${paths.base_output_path}/csv_logger
            name: ${experiment_name}
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ) -> None:
        super().__init__()
        save_dir = os.fspath(save_dir)
        self._save_dir = save_dir
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._fs = get_filesystem(save_dir)
        self._experiment: Optional[_ExperimentWriterMergeByStep] = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def version(self) -> Union[int, str]:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        return os.path.join(self._save_dir, self._name)

    @property
    @override
    def log_dir(self) -> str:
        ver = self.version
        version_str = ver if isinstance(ver, str) else f"version_{ver}"
        return os.path.join(self.root_dir, version_str)

    @property
    @override
    def save_dir(self) -> str:
        return self._save_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> _ExperimentWriterMergeByStep:
        if self._experiment is not None:
            return self._experiment
        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = _ExperimentWriterMergeByStep(log_dir=self.log_dir)
        return self._experiment

    @override
    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Union[Tensor, float]], step: Optional[int] = None) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is None:
            step = 0
        self.experiment.log_metrics(metrics, step)
        if (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Any = None) -> None:
        if params is None:
            return
        params = _convert_params(params)
        if not params:
            return
        hparams_file = os.path.join(self.log_dir, _ExperimentWriterMergeByStep.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, params)

    @override
    @rank_zero_only
    def save(self) -> None:
        super().save()
        if self._experiment is not None:
            self._experiment.save()

    @override
    @rank_zero_only
    def finalize(self, status: str = "") -> None:
        if self._experiment is None:
            return
        self.save()

    def _get_next_version(self) -> int:
        from lightning.fabric.utilities.cloud_io import _is_dir
        versions_root = self.root_dir
        if not _is_dir(self._fs, versions_root, strict=True):
            return 0
        existing = []
        for d in self._fs.listdir(versions_root):
            name = os.path.basename(d["name"])
            if name.startswith("version_") and name[8:].isdigit():
                existing.append(int(name[8:]))
        return max(existing, default=-1) + 1
