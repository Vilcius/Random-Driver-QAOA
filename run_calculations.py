"""
Entry points for large scale parallel calculation functions.
"""

import os
from functools import partial
from os import path

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame

from src.data_processing import numpy_str_to_array
from src.graph_utils import generate_graphs, generate_random_edge_graphs, generate_random_subgraphs, generate_remove_triangle_graphs, generate_remove_random_edge_from_max_degree_vertex, remove_2_random_edges_from_max_degree_vertex_other, remove_all_edges_from_max_degree_vertex
from src.parallel import optimize_expectation_parallel, WorkerRandomCircuit



def init_dataframe(data_path: str, worker: WorkerBaseQAOA, out_path: str, random_type=None):
    if isinstance(worker, WorkerRandomCircuit):
        paths = [(f'{data_path}graph_{i}/{i}.gml', f'{data_path}graph_{i}/{random_type}/{os.fsdecode(j)}')
                 for i in range(11117) for j in sorted(os.listdir(os.fsencode(f'{data_path}graph_{i}/{random_type}')))]
        paths = [(f'{data_path}graph_{i}/{i}.gml', f'{data_path}graph_{i}/{i}.gml') for i in range(11117)]
        index = pd.MultiIndex.from_tuples(paths, names=["path", "random_path"])
        df = DataFrame(index=index)

    elif worker.initial_guess_from is None:
        paths = [f'{data_path}/{i}.gml' for i in range(11117)]
        df = DataFrame(paths).set_axis(['path'], axis=1).set_index('path')

    else:
        raise Exception('No init for this worker')
    df.to_csv(out_path)


def run_graphs_parallel():
    nodes = [8]

    num_workers = 20
    # convergence_threshold = 0.9995
    convergence_threshold = 1.1
    reader = partial(nx.read_gml, destringizer=int)
    p = 1

    for random_type in ['random', 'pseudo_random', 'remove_triangle', 'max_degree_most', 'max_degree_2_most_other', 'max_degree_all']:
        out_path = f'results/random_circuit/{random_type}/out_ma.csv'
        out_col = f'p_{p}'

        worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, initial_guess_from=None, search_space='qaoa')
        # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, initial_guess_from=None, search_space='ma')

        for node in nodes:
            data_path = f'graphs/main/all_{node}/'

            rows_func = lambda df: np.ones((df.shape[0], 1), dtype=bool) if p == 1 else df[f'p_{p - 1}'] < convergence_threshold

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
        # print(f'g = {g}')
        # generate_random_edge_graphs(g)
        # generate_random_subgraphs(g)
        # generate_remove_triangle_graphs(g)
    # for g in range(11117):
    #     print(f'g = {g}')
        # generate_remove_random_edge_from_max_degree_vertex(g)
        # remove_2_random_edges_from_max_degree_vertex_other(g)
        # remove_all_edges_from_max_degree_vertex(g)
    run_graphs_parallel()
    # run_merge()
    # run_correct()
