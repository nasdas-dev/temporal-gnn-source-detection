"""Scientific temporal-network reduction utilities.

The reduction layer creates smaller TSIR artifacts before the quadratic
``[n_nodes, runs, n_nodes]`` simulation tensors are generated.  It keeps the
policy explicit and reportable so reduced experiments can be described as
representative temporal subnetworks rather than ad-hoc truncations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml


DEFAULT_RUNTIME_TARGET_S = 3600
DEFAULT_SAMPLE_REFERENCE = "students"
DEFAULT_SAMPLE_BUDGET_FACTOR = 72


@dataclass(frozen=True)
class NetworkStats:
    """Basic full-network size statistics used by reduction policies."""

    n_nodes: int
    n_edges: int
    n_contacts: int
    t_max: int
    directed: bool

    @property
    def node_edge_cost(self) -> int:
        return self.n_nodes * max(self.n_edges, 1)


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict-like object or Config object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def cfg_to_dict(value: Any) -> Any:
    """Convert Config-like objects to plain JSON/YAML-friendly containers."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: cfg_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cfg_to_dict(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: cfg_to_dict(v) for k, v in vars(value).items()}
    return value


def largest_component(G: nx.Graph) -> tuple[nx.Graph, int]:
    """Return the largest weak/undirected component and removed node count."""
    if G.number_of_nodes() == 0:
        return G.copy(), 0
    components = nx.weakly_connected_components(G) if G.is_directed() else nx.connected_components(G)
    largest = max(components, key=len)
    removed = G.number_of_nodes() - len(largest)
    return G.subgraph(largest).copy(), removed


def directed_from_meta(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"yes", "true", "1"}
    return bool(value)


def read_network_metadata(network: str) -> dict[str, Any]:
    path = Path("nwk") / f"{network}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Missing network metadata: {path}")
    with open(path) as f:
        meta = yaml.safe_load(f) or {}
    t_max = meta.get("time_steps", meta.get("t_max"))
    if t_max is None:
        raise ValueError(f"Cannot determine t_max/time_steps for {network} from {path}")
    directed = directed_from_meta(meta.get("directed", False))
    return {"t_max": int(t_max), "directed": bool(directed), **meta}


def read_full_network_stats(network: str) -> NetworkStats:
    """Read only enough of ``nwk/<network>.csv`` to compute full LCC stats."""
    meta = read_network_metadata(network)
    t_max = int(meta["t_max"])
    directed = bool(meta["directed"])
    graph = nx.DiGraph() if directed else nx.Graph()
    seen_times: dict[tuple[int, int], set[int]] = {}
    csv_path = Path("nwk") / f"{network}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing network CSV: {csv_path}")

    with open(csv_path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) != 3:
                continue
            u, v, t = map(int, fields)
            if t > t_max or u == v:
                continue
            key = (u, v) if directed else tuple(sorted((u, v)))
            times = seen_times.setdefault(key, set())
            if t in times:
                continue
            times.add(t)
            if graph.has_edge(u, v):
                graph.edges[u, v]["times"].append(t)
            else:
                graph.add_edge(u, v, times=[t])

    graph, _ = largest_component(graph)
    contacts = sum(len(data.get("times", [])) for _, _, data in graph.edges(data=True))
    return NetworkStats(
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        n_contacts=contacts,
        t_max=t_max,
        directed=directed,
    )


def auto_node_edge_budget(value: Any) -> int | None:
    """Resolve symbolic node-edge budgets such as ``auto_students_div72``."""
    if value in (None, "", False):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    raw = str(value)
    if raw == "auto_students_div72":
        stats = read_full_network_stats(DEFAULT_SAMPLE_REFERENCE)
        return int(stats.node_edge_cost // DEFAULT_SAMPLE_BUDGET_FACTOR)
    if raw.startswith("auto_") and "_div" in raw:
        prefix, divisor = raw.rsplit("_div", 1)
        reference = prefix.removeprefix("auto_")
        stats = read_full_network_stats(reference)
        return int(stats.node_edge_cost // float(divisor))
    return int(raw)


def _graph_contacts(H: nx.Graph) -> int:
    return sum(len(data.get("times", [])) for _, _, data in H.edges(data=True))


def _incident_activity(H: nx.Graph) -> dict[int, int]:
    activity = {n: 0 for n in H.nodes()}
    for u, v, data in H.edges(data=True):
        count = len(data.get("times", []))
        activity[u] += count
        activity[v] += count
    return activity


def _temporal_neighbors(H: nx.Graph, node: int) -> set[int]:
    if H.is_directed():
        return set(H.successors(node)) | set(H.predecessors(node))
    return set(H.neighbors(node))


def _contact_count_between(H: nx.Graph, u: int, v: int) -> int:
    count = 0
    if H.has_edge(u, v):
        count += len(H.edges[u, v].get("times", []))
    if H.is_directed() and H.has_edge(v, u):
        count += len(H.edges[v, u].get("times", []))
    return count


def _static_edges_to_selected(H: nx.Graph, node: int, selected: set[int]) -> int:
    count = 0
    for other in selected:
        if H.is_directed():
            if H.has_edge(node, other):
                count += 1
            if H.has_edge(other, node):
                count += 1
        elif H.has_edge(node, other):
            count += 1
    return count


def activity_snowball_sample(H: nx.Graph, sample_cfg: Any) -> nx.Graph:
    """Connected activity-biased node sampling under an optional cost budget."""
    n_original = H.number_of_nodes()
    if n_original == 0:
        return H.copy()

    max_cost = auto_node_edge_budget(cfg_get(sample_cfg, "max_node_edge_cost", None))
    target_nodes = cfg_get(sample_cfg, "target_nodes", None)
    min_nodes = int(cfg_get(sample_cfg, "min_nodes", 5))
    seed = int(cfg_get(sample_cfg, "seed", 42))
    rng = np.random.default_rng(seed)

    activity = _incident_activity(H)
    tie_break = {n: float(rng.random()) for n in H.nodes()}
    start = max(H.nodes(), key=lambda n: (activity[n], H.degree(n), tie_break[n]))

    selected = {start}
    frontier = _temporal_neighbors(H, start) - selected
    selected_edge_count = 0

    def reached_target() -> bool:
        if target_nodes is not None and len(selected) >= int(target_nodes):
            return True
        if target_nodes is None and max_cost is None:
            return True
        return False

    while not reached_target():
        pool = frontier or (set(H.nodes()) - selected)
        if not pool:
            break

        candidates = []
        for node in pool:
            add_edges = _static_edges_to_selected(H, node, selected)
            if add_edges == 0 and selected:
                continue
            new_n = len(selected) + 1
            new_m = selected_edge_count + add_edges
            new_cost = new_n * max(new_m, 1)
            within_cost = max_cost is None or new_cost <= float(max_cost)
            within_target = target_nodes is None or new_n <= int(target_nodes)
            forced_min = new_n <= min_nodes
            if (within_cost and within_target) or forced_min:
                connection_activity = sum(_contact_count_between(H, node, s) for s in selected)
                candidates.append((
                    connection_activity,
                    add_edges,
                    activity[node],
                    H.degree(node),
                    tie_break[node],
                    node,
                ))
        if not candidates:
            break

        *_, picked = max(candidates)
        selected_edge_count += _static_edges_to_selected(H, picked, selected)
        selected.add(picked)
        frontier.discard(picked)
        frontier |= _temporal_neighbors(H, picked) - selected

    return H.subgraph(selected).copy()


def _rank_bins(values: dict[int, int], n_bins: int) -> dict[int, int]:
    nodes = sorted(values, key=lambda n: (values[n], n))
    if not nodes:
        return {}
    n_bins = max(1, int(n_bins))
    return {
        node: min(n_bins - 1, int(i * n_bins / len(nodes)))
        for i, node in enumerate(nodes)
    }


def _proportional_quotas(bin_by_node: dict[int, int], target_nodes: int) -> dict[int, int]:
    counts = Counter(bin_by_node.values())
    total = sum(counts.values())
    if total == 0:
        return {}

    raw = {b: target_nodes * count / total for b, count in counts.items()}
    quotas = {b: min(counts[b], int(raw[b])) for b in counts}
    for b in counts:
        if counts[b] > 0 and quotas[b] == 0 and target_nodes >= len(counts):
            quotas[b] = 1

    remaining = target_nodes - sum(quotas.values())
    order = sorted(counts, key=lambda b: (raw[b] - int(raw[b]), counts[b]), reverse=True)
    while remaining > 0 and order:
        progressed = False
        for b in order:
            if quotas[b] < counts[b]:
                quotas[b] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return quotas


def balanced_activity_snowball_sample(H: nx.Graph, sample_cfg: Any) -> nx.Graph:
    """Connected sampling that preserves temporal activity and degree strata."""
    n_original = H.number_of_nodes()
    if n_original == 0:
        return H.copy()

    target_nodes = cfg_get(sample_cfg, "target_nodes", None)
    if target_nodes is None:
        target_nodes = n_original
    target_nodes = min(max(1, int(target_nodes)), n_original)
    max_cost = auto_node_edge_budget(cfg_get(sample_cfg, "max_node_edge_cost", None))
    min_nodes = int(cfg_get(sample_cfg, "min_nodes", 5))
    n_bins = int(cfg_get(sample_cfg, "stratification_bins", 4))
    seed = int(cfg_get(sample_cfg, "seed", 42))
    rng = np.random.default_rng(seed)

    activity = _incident_activity(H)
    degree = dict(H.degree())
    activity_bin = _rank_bins(activity, n_bins)
    degree_bin = _rank_bins(degree, n_bins)
    activity_quota = _proportional_quotas(activity_bin, target_nodes)
    degree_quota = _proportional_quotas(degree_bin, target_nodes)
    tie_break = {n: float(rng.random()) for n in H.nodes()}

    start = max(H.nodes(), key=lambda n: (activity[n], degree[n], tie_break[n]))
    selected = {start}
    selected_edge_count = 0
    selected_activity_bins = Counter([activity_bin[start]])
    selected_degree_bins = Counter([degree_bin[start]])
    frontier = _temporal_neighbors(H, start) - selected

    while len(selected) < target_nodes:
        pool = frontier or (set(H.nodes()) - selected)
        if not pool:
            break

        candidates = []
        for node in pool:
            add_edges = _static_edges_to_selected(H, node, selected)
            if add_edges == 0 and selected:
                continue
            new_n = len(selected) + 1
            new_m = selected_edge_count + add_edges
            new_cost = new_n * max(new_m, 1)
            within_cost = max_cost is None or new_cost <= float(max_cost)
            if not within_cost and new_n > min_nodes:
                continue

            a_bin = activity_bin[node]
            d_bin = degree_bin[node]
            a_deficit = max(0, activity_quota.get(a_bin, 0) - selected_activity_bins[a_bin])
            d_deficit = max(0, degree_quota.get(d_bin, 0) - selected_degree_bins[d_bin])
            connection_activity = sum(_contact_count_between(H, node, s) for s in selected)
            candidates.append((
                int(a_deficit > 0),
                int(d_deficit > 0),
                a_deficit + d_deficit,
                connection_activity,
                add_edges,
                activity[node],
                degree[node],
                tie_break[node],
                node,
            ))
        if not candidates:
            break

        *_, picked = max(candidates)
        selected_edge_count += _static_edges_to_selected(H, picked, selected)
        selected.add(picked)
        selected_activity_bins[activity_bin[picked]] += 1
        selected_degree_bins[degree_bin[picked]] += 1
        frontier.discard(picked)
        frontier |= _temporal_neighbors(H, picked) - selected

    return H.subgraph(selected).copy()


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Small scipy-free two-sample Kolmogorov-Smirnov distance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 1.0
    values = np.sort(np.unique(np.concatenate([a, b])))
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    cdf_a = np.searchsorted(a_sorted, values, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, values, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _burstiness(times: list[int]) -> float:
    if len(times) < 3:
        return 0.0
    diffs = np.diff(np.sort(np.asarray(times, dtype=np.int64)))
    if diffs.size == 0:
        return 0.0
    mean = float(np.mean(diffs))
    std = float(np.std(diffs))
    denom = std + mean
    return float((std - mean) / denom) if denom > 0 else 0.0


def _feature_vectors(H: nx.Graph) -> dict[str, np.ndarray | float | int]:
    activity = np.asarray(list(_incident_activity(H).values()), dtype=float)
    degree = np.asarray([deg for _, deg in H.degree()], dtype=float)
    times = [int(t) for _, _, data in H.edges(data=True) for t in data.get("times", [])]
    t_span = max(times) - min(times) + 1 if times else 1
    return {
        "activity": activity,
        "degree": degree,
        "contact_rate": float(len(times) / max(t_span, 1)),
        "active_nodes": int(np.sum(activity > 0)) if activity.size else 0,
        "burstiness": _burstiness(times),
        "inter_event": np.diff(np.sort(np.asarray(times, dtype=np.int64))) if len(times) >= 2 else np.asarray([], dtype=float),
    }


def _graph_rows(H: nx.Graph) -> np.ndarray:
    rows: list[tuple[int, int, int]] = []
    for u, v, data in H.edges(data=True):
        rows.extend((int(u), int(v), int(t)) for t in data.get("times", []))
    if not rows:
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.asarray(rows, dtype=np.int64)
    return arr[arr[:, 2].argsort()]


def _time_window_steps(raw_steps: Any, time_cfg: Any, nwk_cfg: Any, full_t_max: int) -> int:
    if raw_steps not in (None, "auto", ""):
        return max(1, int(raw_steps))
    max_days = int(cfg_get(time_cfg, "max_steps_days", 365))
    granularity = str(cfg_get(nwk_cfg, "time_granularity", "") or "").lower()
    if "hour" in granularity:
        steps = max_days * 24
    else:
        steps = max_days
    return max(1, min(int(steps), int(full_t_max) + 1))


def select_representative_window(
    H: nx.Graph,
    time_cfg: Any,
    nwk_cfg: Any,
) -> dict[str, Any] | None:
    """Choose a contiguous window that best matches full-network statistics."""
    rows = _graph_rows(H)
    if rows.size == 0:
        return None
    t_min = int(rows[:, 2].min())
    t_max = int(rows[:, 2].max())
    full_t = int(cfg_get(nwk_cfg, "t_max", t_max))
    apply_gt = int(cfg_get(time_cfg, "apply_if_time_steps_gt", 1000))
    if full_t <= apply_gt:
        return None

    window_steps = _time_window_steps(cfg_get(time_cfg, "max_steps", None), time_cfg, nwk_cfg, full_t)
    if window_steps >= full_t + 1:
        return None

    start_min = t_min
    start_max = max(t_min, t_max - window_steps + 1)
    if start_max <= start_min:
        candidate_starts = [start_min]
    else:
        n_candidates = int(cfg_get(time_cfg, "candidate_windows", 64))
        candidate_starts = np.linspace(start_min, start_max, num=min(n_candidates, start_max - start_min + 1))
        candidate_starts = sorted(set(int(round(x)) for x in candidate_starts))

    full_features = _feature_vectors(H)
    best: dict[str, Any] | None = None
    for start in candidate_starts:
        stop = start + window_steps - 1
        window = restrict_time_window(H, start, stop, reindex=False)
        if window.number_of_edges() == 0:
            continue
        features = _feature_vectors(window)
        contact_rate_delta = abs(float(features["contact_rate"]) - float(full_features["contact_rate"])) / max(float(full_features["contact_rate"]), 1e-12)
        active_delta = abs(int(features["active_nodes"]) - int(full_features["active_nodes"])) / max(int(full_features["active_nodes"]), 1)
        score = (
            contact_rate_delta
            + active_delta
            + _ks_distance(np.asarray(full_features["activity"]), np.asarray(features["activity"]))
            + _ks_distance(np.asarray(full_features["degree"]), np.asarray(features["degree"]))
            + _ks_distance(np.asarray(full_features["inter_event"]), np.asarray(features["inter_event"]))
            + abs(float(features["burstiness"]) - float(full_features["burstiness"]))
        )
        if best is None or score < float(best["score"]):
            best = {
                "start": int(start),
                "end": int(stop),
                "window_steps": int(window_steps),
                "score": float(score),
                "contact_rate_delta": float(contact_rate_delta),
                "active_node_delta": float(active_delta),
                "degree_ks": _ks_distance(np.asarray(full_features["degree"]), np.asarray(features["degree"])),
                "activity_ks": _ks_distance(np.asarray(full_features["activity"]), np.asarray(features["activity"])),
                "inter_event_ks": _ks_distance(np.asarray(full_features["inter_event"]), np.asarray(features["inter_event"])),
                "burstiness_delta": float(abs(float(features["burstiness"]) - float(full_features["burstiness"]))),
            }
    return best


def restrict_time_window(H: nx.Graph, start: int, end: int, reindex: bool = True) -> nx.Graph:
    """Return a graph containing only contacts in ``[start, end]``."""
    G = nx.DiGraph() if H.is_directed() else nx.Graph()
    G.add_nodes_from(H.nodes(data=True))
    offset = int(start) if reindex else 0
    for u, v, data in H.edges(data=True):
        times = [
            int(t) - offset
            for t in data.get("times", [])
            if int(start) <= int(t) <= int(end)
        ]
        if times:
            G.add_edge(u, v, **{**data, "times": sorted(set(times))})
    G.graph.update(H.graph)
    return G


def _sample_graph(H: nx.Graph, sample_cfg: Any) -> nx.Graph:
    method = cfg_get(sample_cfg, "method", None)
    if method in (None, "none", "None", False):
        return H
    if method == "activity_snowball":
        return activity_snowball_sample(H, sample_cfg)
    if method == "balanced_activity_snowball":
        return balanced_activity_snowball_sample(H, sample_cfg)
    raise ValueError(f"Unknown network sampling method: {method}")


def sample_temporal_network(H: nx.Graph, sample_cfg: Any) -> nx.Graph:
    """Backward-compatible node-only sampling wrapper for ``nwk.sample``."""
    return apply_node_sampling(H, sample_cfg)[0]


def apply_node_sampling(H: nx.Graph, sample_cfg: Any) -> tuple[nx.Graph, dict[str, Any]]:
    """Apply configured node sampling and return sampled graph plus report."""
    method = cfg_get(sample_cfg, "method", None)
    if method in (None, "none", "None", False):
        return H, {}

    original_nodes = H.number_of_nodes()
    original_edges = H.number_of_edges()
    original_contacts = _graph_contacts(H)
    original_features = _feature_vectors(H)
    sampled = _sample_graph(H, sample_cfg)
    sampled, removed = largest_component(sampled)

    relabeled = False
    if sampled.number_of_nodes() < original_nodes:
        sampled = nx.convert_node_labels_to_integers(
            sampled,
            label_attribute="sample_id",
            ordering="sorted",
        )
        relabeled = True

    sampled_nodes = sampled.number_of_nodes()
    sampled_edges = sampled.number_of_edges()
    sampled_contacts = _graph_contacts(sampled)
    original_cost = original_nodes * max(original_edges, 1)
    sampled_cost = sampled_nodes * max(sampled_edges, 1)
    sampled_features = _feature_vectors(sampled)
    report = {
        "method": method,
        "seed": int(cfg_get(sample_cfg, "seed", 42)),
        "original_nodes": int(original_nodes),
        "original_edges": int(original_edges),
        "original_contacts": int(original_contacts),
        "sampled_nodes": int(sampled_nodes),
        "sampled_edges": int(sampled_edges),
        "sampled_contacts": int(sampled_contacts),
        "node_edge_cost_reduction": float(original_cost / max(sampled_cost, 1)),
        "max_node_edge_cost": auto_node_edge_budget(cfg_get(sample_cfg, "max_node_edge_cost", None)),
        "target_nodes": cfg_get(sample_cfg, "target_nodes", None),
        "stratification_bins": cfg_get(sample_cfg, "stratification_bins", None),
        "removed_disconnected_sampled_nodes": int(removed),
        "relabelled": bool(relabeled),
        "degree_ks": _ks_distance(np.asarray(original_features["degree"]), np.asarray(sampled_features["degree"])),
        "activity_ks": _ks_distance(np.asarray(original_features["activity"]), np.asarray(sampled_features["activity"])),
        "inter_event_ks": _ks_distance(np.asarray(original_features["inter_event"]), np.asarray(sampled_features["inter_event"])),
        "burstiness_delta": float(abs(float(original_features["burstiness"]) - float(sampled_features["burstiness"]))),
        "active_node_fraction": float(int(sampled_features["active_nodes"]) / max(sampled_nodes, 1)),
        "original_ids": [
            sampled.nodes[n].get("old_id", sampled.nodes[n].get("sample_id", n))
            for n in sorted(sampled.nodes())
        ],
    }
    sampled.graph["sample"] = report
    return sampled, report


def _legacy_sample_to_reduction(sample_cfg: Any) -> dict[str, Any]:
    sample = cfg_to_dict(sample_cfg) or {}
    return {
        "enabled": True,
        "preset": "legacy_sample",
        "node": sample,
    }


def normalize_reduction_cfg(nwk_cfg: Any) -> dict[str, Any] | None:
    reduction = cfg_get(nwk_cfg, "reduction", None)
    sample = cfg_get(nwk_cfg, "sample", None)
    if reduction is None and sample is not None:
        return _legacy_sample_to_reduction(sample)
    return cfg_to_dict(reduction)


def reduction_enabled(reduction_cfg: dict[str, Any] | None) -> bool:
    if not reduction_cfg:
        return False
    enabled = reduction_cfg.get("enabled", True)
    return enabled not in (False, "false", "False", "none", "None", "off", "disabled")


def make_reduction_id(network: str, reduction_cfg: dict[str, Any], report: dict[str, Any]) -> str:
    preset = str(reduction_cfg.get("preset", "custom"))
    node = reduction_cfg.get("node") or {}
    time_cfg = report.get("time") or {}
    n = node.get("target_nodes", "full")
    seed = node.get("seed", reduction_cfg.get("seed", 42))
    w = time_cfg.get("window_steps", "full")
    return f"{network}_{preset}_n{n}_w{w}_s{seed}"


def apply_network_reduction(H: nx.Graph, nwk_cfg: Any) -> tuple[nx.Graph, dict[str, Any]]:
    """Apply configured temporal and node reductions to a temporal graph."""
    reduction_cfg = normalize_reduction_cfg(nwk_cfg)
    if not reduction_enabled(reduction_cfg):
        return H, {}

    network = str(cfg_get(nwk_cfg, "name", "network"))
    original = {
        "nodes": int(H.number_of_nodes()),
        "edges": int(H.number_of_edges()),
        "contacts": int(_graph_contacts(H)),
        "t_max": int(cfg_get(nwk_cfg, "t_max", 0)),
    }
    report: dict[str, Any] = {
        "enabled": True,
        "preset": reduction_cfg.get("preset", "custom"),
        "runtime_target_s": int(reduction_cfg.get("runtime_target_s", DEFAULT_RUNTIME_TARGET_S)),
        "network": network,
        "original": original,
    }
    G = H

    time_cfg = reduction_cfg.get("time") or {}
    if cfg_get(time_cfg, "method", None) == "representative_window":
        selection = select_representative_window(G, time_cfg, nwk_cfg)
        if selection is not None:
            reindex = bool(cfg_get(time_cfg, "reindex_to_zero", True))
            G = restrict_time_window(G, int(selection["start"]), int(selection["end"]), reindex=reindex)
            G, removed = largest_component(G)
            if reindex:
                reduced_t_max = int(selection["window_steps"]) - 1
            else:
                reduced_t_max = int(selection["end"])
            report["time"] = {
                **selection,
                "method": "representative_window",
                "reindex_to_zero": bool(reindex),
                "removed_disconnected_nodes": int(removed),
                "reduced_t_max": int(reduced_t_max),
            }

    node_cfg = reduction_cfg.get("node") or {}
    apply_if = int(cfg_get(node_cfg, "apply_if_nodes_gt", 0))
    if cfg_get(node_cfg, "method", None) and G.number_of_nodes() > apply_if:
        G, node_report = apply_node_sampling(G, node_cfg)
        report["node"] = node_report

    reduced = {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "contacts": int(_graph_contacts(G)),
        "t_max": int(report.get("time", {}).get("reduced_t_max", cfg_get(nwk_cfg, "t_max", 0))),
    }
    report["reduced"] = reduced
    report["node_edge_cost_reduction"] = float(
        (original["nodes"] * max(original["edges"], 1))
        / max(reduced["nodes"] * max(reduced["edges"], 1), 1)
    )
    report["reduction_id"] = make_reduction_id(network, reduction_cfg, report)
    G.graph["reduction_report"] = report
    G.graph["sample"] = report.get("node", {})
    return G, report

