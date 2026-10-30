from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def quantile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class ResearchArchive:
    """Read-only access to the frozen replay and empirical risk-confidence mapping."""

    def __init__(self) -> None:
        self.replay = json.loads((DATA / "replay.json").read_text(encoding="utf-8"))
        self.topology = json.loads((DATA / "topology.json").read_text(encoding="utf-8"))
        self.metrics = json.loads((DATA / "metrics.json").read_text(encoding="utf-8"))
        self.nodes = self.replay["nodeOrder"]
        self.origins = self.replay["origins"]
        self.offsets = self.replay["originOffsets"]
        self.temperature = self.replay["temperature"]
        self.prediction = self.replay["prediction"]

    @property
    def frame_count(self) -> int:
        return len(self.origins)

    @staticmethod
    def node_metadata(node_id: str, index: int) -> dict:
        rack = node_id.split("_")[0]
        code = node_id.split("_")[1]
        side = "Front" if code[2] == "F" else "Back"
        level = {"B": "Lower", "M": "Middle", "T": "Upper"}[code[-1]]
        rack_index = index // 6
        row = 0 if rack.startswith("R05") else 1
        column = int(rack[-2:]) - 1
        x = float(column)
        y = float(row * 3 + (-0.34 if side == "Front" else 0.34))
        z = {"Lower": 0.55, "Middle": 1.45, "Upper": 2.35}[level]
        return {"id": node_id, "index": index, "rack": rack, "rack_index": rack_index,
                "side": side, "level": level, "x": x, "y": y, "z": z}

    def current_temperatures(self, frame: int) -> list[float]:
        return self.temperature[self.offsets[frame]]

    def forecast(self, frame: int, horizon: int) -> list[float]:
        return self.prediction[frame][horizon - 1]

    @lru_cache(maxsize=250_000)
    def risk_summary(self, frame: int, node: int, horizon: int) -> dict:
        history = [row[node] for row in self.temperature if row[node] is not None]
        center = quantile(history, 0.5)
        deviations = [abs(value - center) for value in history]
        radius = max(0.05, quantile(deviations, 0.95) or 0.05)
        residuals = []
        for index, offset in enumerate(self.offsets):
            target = offset + horizon
            if target >= len(self.temperature):
                continue
            actual = self.temperature[target][node]
            predicted = self.prediction[index][horizon - 1][node]
            if actual is not None and predicted is not None:
                residuals.append(actual - predicted)
        predicted = self.prediction[frame][horizon - 1][node]
        samples = [predicted + residual for residual in residuals]
        probability = sum(abs(value - center) > radius for value in samples) / max(len(samples), 1)
        return {
            "probability": probability,
            "lower": quantile(samples, 0.05),
            "upper": quantile(samples, 0.95),
            "prediction": predicted,
            "sample_count": len(samples),
        }

    @lru_cache(maxsize=20_000)
    def observation_evidence(self, frame: int, node: int) -> dict:
        offset = self.offsets[frame]
        series = [row[node] for row in self.temperature]
        center = median(series)
        mad = median(abs(value - center) for value in series)
        scale = max(0.02, 1.4826022185 * mad)
        z_series = [abs(value - center) / scale for value in series]

        def rank(values: list[float], value: float) -> float:
            return sum(item <= value for item in values) / max(len(values), 1)

        statistical = rank(z_series, z_series[offset])
        edges = self.topology["directed_edges"]
        neighbours = sorted({endpoint for edge in edges
                             if edge["source_index"] == node or edge["target_index"] == node
                             for endpoint in (edge["source_index"], edge["target_index"])
                             if endpoint != node})
        neighbour_series = []
        for time_index in range(len(self.temperature)):
            near = [abs(self.temperature[time_index][idx] - median(row[idx] for row in self.temperature))
                    for idx in neighbours]
            neighbour_series.append(sum(near) / max(len(near), 1))
        graph_series = [math.sqrt(0.1 * own ** 2 + 0.9 * own * near)
                        for own, near in zip(z_series, neighbour_series)]
        graph = rank(graph_series, graph_series[offset])
        dynamic_series = []
        for time_index in range(5, len(self.temperature)):
            changes = [self.temperature[time_index][idx] - self.temperature[time_index - 5][idx]
                       for idx in range(len(self.nodes))]
            dynamic_series.append(math.sqrt(sum(value * value for value in changes) / len(changes)))
        dynamic = rank(dynamic_series, dynamic_series[max(0, offset - 5)])
        d5 = series[offset] - series[max(0, offset - 5)]
        d15 = series[offset] - series[max(0, offset - 15)]
        positive5 = [series[i] - series[i - 5] for i in range(5, len(series)) if series[i] - series[i - 5] > 0]
        positive15 = [series[i] - series[i - 15] for i in range(15, len(series)) if series[i] - series[i - 15] > 0]
        sequence = math.sqrt((rank(positive5, d5) if d5 > 0 else 0) *
                             (rank(positive15, d15) if d15 > 0 else 0))
        return {"statistical": statistical, "graph": graph, "dynamic": dynamic, "sequence": sequence}
