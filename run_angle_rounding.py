"""
Entry points for large scale parallel calculation functions.
Used for angle rounding heuristic
"""

import os
from functools import partial
from os import path

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
from random import sample

from src.data_processing import merge_dfs, numpy_str_to_array
from src.graph_utils import get_max_edge_depth, is_isomorphic
from src.parallel import optimize_expectation_parallel, WorkerFourier, WorkerStandard, WorkerBaseQAOA, WorkerInterp, WorkerGreedy, WorkerMA, WorkerLinear, WorkerCombined, WorkerConstant, WorkerRandomCircuit


def generate_graphs():
    """
    Generate the graphs with the specific number of nodes and a given depth.
    """
    num_graphs = 1000
    max_attempts = 10 ** 10
    nodes = 9
    depth = 4
    edge_prob = 0.1
    out_path = f'graphs/main/nodes_{nodes}/depth_{depth}'

    graphs = np.empty(num_graphs, dtype=object)
    graphs_generated = 0
    for i in range(max_attempts):
        next_graph = nx.gnp_random_graph(nodes, edge_prob)
        if not nx.is_connected(next_graph):
            continue
        if get_max_edge_depth(next_graph) != depth:
            continue
        if is_isomorphic(next_graph, graphs[:graphs_generated]):
            continue
        graphs[graphs_generated] = next_graph
        graphs_generated += 1
        print(f'{graphs_generated}')
        if graphs_generated == num_graphs:
            break
    else:
        raise 'Failed to generate connected set'
    print('Generation done')

    for i in range(len(graphs)):
        nx.write_gml(graphs[i], f'{out_path}/{i}.gml')


def remove_max_degree_edge(g):
    """
    Take in a graph $g$ and remove the edge $uv$ if one of $u$ or $v$ has maximum degree.
    """
    nodes = 8
    graph_path = f'graphs/main/all_{nodes}/graph_{g}/{g}.gml'
    graph = nx.read_gml(graph_path)

    vertex_degrees = {v: graph.degree(v) for v in graph.nodes}
    max_degree = max(vertex_degrees.values())
    max_degree_vertices = [v for v in graph.nodes if vertex_degrees[v] == max_degree]

    out_path = f'graphs/main/all_{nodes}/graph_{g}/angle_rounding_gamma'

    if not path.exists(out_path):
        os.makedirs(out_path)

    new_graph = graph.copy()

    # Remove the edge with max degree vertex
    for v in max_degree_vertices:
        for u in graph[v]:
            if (u, v) in new_graph.edges:
                new_graph.remove_edge(v, u)

    nx.write_gml(new_graph, f'{out_path}/0.gml')


def init_dataframe(data_path: str, worker: WorkerBaseQAOA, out_path: str, random_type=None):
    if isinstance(worker, WorkerRandomCircuit):
        paths = [(f'{data_path}graph_{i}/{i}.gml', f'{data_path}graph_{i}/{random_type}/{os.fsdecode(j)}')
                 for i in range(11117) for j in sorted(os.listdir(os.fsencode(f'{data_path}graph_{i}/{random_type}')))]
        index = pd.MultiIndex.from_tuples(paths, names=["path", "random_path"])
        df = DataFrame(index=index)

    elif worker.initial_guess_from is None:
        paths = [f'{data_path}/{i}.gml' for i in range(11117)]
        df = DataFrame(paths).set_axis(['path'], axis=1).set_index('path')

    elif isinstance(worker, (WorkerInterp, WorkerFourier, WorkerGreedy, WorkerCombined)) or hasattr(worker, 'guess_provider') and isinstance(worker.guess_provider, WorkerInterp):
        df = pd.read_csv(f'{data_path}/output/{worker.search_space}/random/p_1/out.csv', index_col=0)
        prev_nfev = df.filter(regex=r'r_\d_nfev').sum(axis=1).astype(int)
        df = df.filter(regex='r_10').rename(columns=lambda name: f'p_1{name[4:]}')
        df['p_1_nfev'] += prev_nfev

        if isinstance(worker, (WorkerInterp, WorkerFourier)):
            df = df.rename(columns={'p_1_angles': 'p_1_angles_unperturbed'})
            df['p_1_angles_best'] = df['p_1_angles_unperturbed']

    elif isinstance(worker, WorkerMA):
        df = pd.read_csv(f'{data_path}/output/qaoa/constant/0.2/out.csv', index_col=0)
        # df = df.filter(regex=r'p_\d+_angles').rename(columns=lambda name: f'{name[:-7]}_starting_angles')
        df = df.filter(regex=r'p_\d+_angles')

    else:
        raise Exception('No init for this worker')
    df.to_csv(out_path)


def run_graphs_parallel():
    nodes = 8

    num_workers = 5
    # convergence_threshold = 0.9995
    convergence_threshold = 1.1
    reader = partial(nx.read_gml, destringizer=int)
    p = 1

    # for p in ps:
    random_type = 'angle_rounding_gamma'
    out_path = 'results/angle_rounding_gamma_ma/out.csv'
    out_col = f'p_{p}'
    # initial_guess_from = None if p == 1 else f'p_{p - 1}'
    # initial_guess_from = f'p_{p}'
    # transfer_from = None if p == 1 else f'p_{p - 1}'
    # transfer_p = None if p == 1 else p - 1

    # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, initial_guess_from=None, search_space='qaoa')
    worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, initial_guess_from=None, search_space='ma')

    data_path = f'graphs/main/all_{nodes}/'

    def rows_func(df): return np.ones((df.shape[0], 1), dtype=bool) if p == 1 else df[f'p_{p - 1}'] < convergence_threshold

    out_folder = path.split(out_path)[0]
    if not path.exists(out_folder):
        os.makedirs(path.split(out_path)[0])
    if not path.exists(out_path):
        init_dataframe(data_path, worker, out_path, random_type)

    optimize_expectation_parallel(out_path, rows_func, num_workers, worker)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # generate_graphs()
    # for g in range(11117):
    #     remove_max_degree_edge(g)
    #     print(f'g = {g}')
    run_graphs_parallel()


