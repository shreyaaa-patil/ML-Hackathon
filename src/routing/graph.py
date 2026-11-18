"""Directed road graph utilities for PathSnap route decoding."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.data.geo_utils import bearing_deg, haversine_m
from src.data.loader import load_arcs, load_nodes


@dataclass(frozen=True)
class Arc:
    """Geometry and endpoint information for one source arc ID."""

    arc_id: int
    from_node_id: int
    to_node_id: int
    length_m: float
    bearing_deg: float


class RoadGraph:
    """A NetworkX directed multigraph that preserves the input arc IDs."""

    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame):
        required_nodes = {"node_id", "longitude", "latitude"}
        required_arcs = {"arc_id", "from_node_id", "to_node_id"}
        if missing := required_nodes.difference(nodes.columns):
            raise ValueError(f"nodes is missing columns: {sorted(missing)}")
        if missing := required_arcs.difference(arcs.columns):
            raise ValueError(f"arcs is missing columns: {sorted(missing)}")

        self.graph = nx.MultiDiGraph()
        self.arcs: dict[int, Arc] = {}
        self.node_coordinates: dict[int, tuple[float, float]] = {}
        for row in nodes.itertuples(index=False):
            node_id = int(row.node_id)
            lon, lat = float(row.longitude), float(row.latitude)
            if not np.isfinite(lon) or not np.isfinite(lat):
                continue
            self.node_coordinates[node_id] = (lon, lat)
            self.graph.add_node(node_id, longitude=lon, latitude=lat)

        for row in arcs.itertuples(index=False):
            arc_id = int(row.arc_id)
            source, target = int(row.from_node_id), int(row.to_node_id)
            if source not in self.node_coordinates or target not in self.node_coordinates:
                continue
            lon1, lat1 = self.node_coordinates[source]
            lon2, lat2 = self.node_coordinates[target]
            length = float(haversine_m(lon1, lat1, lon2, lat2))
            heading = float(bearing_deg(lon1, lat1, lon2, lat2))
            arc = Arc(arc_id, source, target, length, heading)
            self.arcs[arc_id] = arc
            self.graph.add_edge(
                source, target, key=arc_id, arc_id=arc_id,
                length_m=length, bearing_deg=heading,
            )

        if not self.arcs:
            raise ValueError("No valid road arcs could be loaded")
        self._connector_cache: dict[tuple[int, int], tuple[float, tuple[int, ...]] | None] = {}

    def transition(self, source_arc_id: int, target_arc_id: int,
                   source_fraction: float = 0.5, target_fraction: float = 0.5
                   ) -> tuple[float, tuple[int, ...]] | None:
        """Return network distance and traversed arc IDs, or None if unreachable.

        Fractions are projected positions along the candidate arcs, from 0 at
        the from-node to 1 at the to-node. The returned path includes both
        candidate arcs and any connecting arcs between them.
        """
        a_id, b_id = int(source_arc_id), int(target_arc_id)
        if a_id not in self.arcs or b_id not in self.arcs:
            return None
        fa = float(np.clip(source_fraction, 0.0, 1.0))
        fb = float(np.clip(target_fraction, 0.0, 1.0))
        a, b = self.arcs[a_id], self.arcs[b_id]
        if a_id == b_id:
            result = (abs(fb-fa) * a.length_m, (a_id,))
        else:
            connector_key = (a_id, b_id)
            if connector_key not in self._connector_cache:
                try:
                    middle_distance, middle_nodes = nx.bidirectional_dijkstra(
                        self.graph, a.to_node_id, b.from_node_id, weight="length_m"
                    )
                    middle_arcs: list[int] = []
                    for u, v in zip(middle_nodes, middle_nodes[1:]):
                        options = self.graph.get_edge_data(u, v)
                        edge = min(options.values(), key=lambda item: item["length_m"])
                        middle_arcs.append(int(edge["arc_id"]))
                    self._connector_cache[connector_key] = (
                        middle_distance, tuple([a_id, *middle_arcs, b_id]),
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    self._connector_cache[connector_key] = None
            connector = self._connector_cache[connector_key]
            if connector is None:
                result = None
            else:
                middle_distance, path = connector
                distance = ((1.0-fa)*a.length_m + middle_distance + fb*b.length_m)
                result = (float(distance), path)

        return result


def _as_frame(source: pd.DataFrame | str | Path, kind: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        if kind == "nodes" and "node_id" not in frame:
            frame["node_id"] = np.arange(len(frame), dtype=np.int64)
        if kind == "arcs" and "arc_id" not in frame:
            frame["arc_id"] = np.arange(len(frame), dtype=np.int64)
        return frame
    path = Path(source)
    return load_nodes(path) if kind == "nodes" else load_arcs(path)


def load_road_graph(nodes: pd.DataFrame | str | Path,
                    arcs: pd.DataFrame | str | Path) -> RoadGraph:
    """Load .nodes/.arcs files or loader-compatible DataFrames."""
    return RoadGraph(_as_frame(nodes, "nodes"), _as_frame(arcs, "arcs"))


def load_trip_graph(trip_dir: str | Path) -> RoadGraph:
    """Find the unique .nodes and .arcs files in a trip directory and load them."""
    root = Path(trip_dir)
    node_files, arc_files = list(root.glob("*.nodes")), list(root.glob("*.arcs"))
    if len(node_files) != 1 or len(arc_files) != 1:
        raise FileNotFoundError(
            f"Expected one .nodes and one .arcs file in {root}; "
            f"found {len(node_files)} and {len(arc_files)}"
        )
    return load_road_graph(node_files[0], arc_files[0])
