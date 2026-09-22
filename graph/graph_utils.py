import numpy as np
import pandas as pd


def build_topk_graph(correlation, top_k, positive_only=True, add_self_loops=True):
    node_names = correlation.columns.tolist()
    directed = np.zeros((len(node_names), len(node_names)), dtype=np.float32)

    for index, node_name in enumerate(node_names):
        candidates = correlation.loc[node_name].copy()
        candidates.loc[node_name] = np.nan
        if positive_only:
            candidates = candidates[candidates > 0]
        candidates = candidates.dropna().sort_values(ascending=False).head(top_k)
        for neighbor_name, value in candidates.items():
            neighbor_index = node_names.index(neighbor_name)
            directed[index, neighbor_index] = float(value)

    weighted = np.maximum(directed, directed.T)
    binary = (weighted > 0).astype(np.float32)
    if add_self_loops:
        np.fill_diagonal(weighted, 1.0)
        np.fill_diagonal(binary, 1.0)
    return weighted, binary, directed


def build_and_save_graph(train_data, station_names, output_dir, summary_name, top_k=5,
                         remove_all_zero=True, positive_only=True, add_self_loops=True,
                         zero_eps=1e-8):
    if train_data.columns.astype(str).tolist() != station_names:
        raise RuntimeError("训练时间序列与节点顺序文件不一致。")
    if not np.isfinite(train_data.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("构图训练数据存在NaN或Inf。")

    graph_data = train_data.copy()
    original_steps = len(graph_data)
    if remove_all_zero:
        graph_data = graph_data.loc[graph_data.abs().max(axis=1) > zero_eps].copy()
    if graph_data.empty:
        raise RuntimeError("删除全零时间点后没有可用构图数据。")

    correlation = graph_data.corr(method="pearson")
    if correlation.isna().any().any():
        bad_nodes = correlation.columns[correlation.isna().any(axis=0)].tolist()
        raise RuntimeError(f"相关矩阵存在NaN，异常节点：{bad_nodes}")

    weighted, binary, directed = build_topk_graph(
        correlation, top_k, positive_only, add_self_loops
    )
    binary_no_self = binary.copy()
    np.fill_diagonal(binary_no_self, 0)
    degree = binary_no_self.sum(axis=1).astype(int)
    edge_count = int(binary_no_self.sum() / 2)
    node_count = len(station_names)
    density = edge_count / (node_count * (node_count - 1) / 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    correlation.to_csv(output_dir / "correlation_matrix.csv")
    np.save(output_dir / "adjacency_weighted.npy", weighted)
    np.save(output_dir / "adjacency_binary.npy", binary)
    pd.DataFrame(weighted, index=station_names, columns=station_names).to_csv(
        output_dir / "adjacency_weighted.csv"
    )
    pd.DataFrame(binary.astype(int), index=station_names, columns=station_names).to_csv(
        output_dir / "adjacency_binary.csv"
    )
    graph_data.to_csv(output_dir / "graph_input_timeseries.csv")

    edges = []
    for source in range(node_count):
        for target in range(source + 1, node_count):
            if binary_no_self[source, target] > 0:
                edges.append({
                    "source_index": source, "source": station_names[source],
                    "target_index": target, "target": station_names[target],
                    "correlation": float(weighted[source, target]),
                })
    pd.DataFrame(edges).to_csv(output_dir / "edge_list.csv", index=False)
    pd.DataFrame({
        "node_index": np.arange(node_count), "NodeID": station_names, "degree": degree,
    }).to_csv(output_dir / "node_degree.csv", index=False)

    graph_info = pd.DataFrame([{
        "node_count": node_count, "top_k": top_k,
        "edge_count_no_self_loop": edge_count,
        "min_degree": int(degree.min()), "max_degree": int(degree.max()),
        "mean_degree": float(degree.mean()), "graph_density": density,
        "train_time_steps_original": original_steps,
        "train_time_steps_for_graph": len(graph_data),
        "removed_all_zero_steps": original_steps - len(graph_data),
        "remove_all_zero_timestamps": remove_all_zero,
        "positive_correlation_only": positive_only,
        "self_loop": add_self_loops, "correlation_method": "pearson",
    }])
    graph_info.to_csv(output_dir / "graph_info.csv", index=False)
    graph_info.to_csv(output_dir / summary_name, index=False)

    return {
        "weighted": weighted, "binary": binary, "directed": directed,
        "edge_count": edge_count, "degree": degree, "density": density,
        "original_steps": original_steps, "graph_steps": len(graph_data),
    }
