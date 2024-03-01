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
from random import sample

from src.data_processing import merge_dfs, numpy_str_to_array
from src.graph_utils import get_max_edge_depth, is_isomorphic
from src.parallel import optimize_expectation_parallel, WorkerFourier, WorkerStandard, WorkerBaseQAOA, WorkerInterp, WorkerGreedy, WorkerMA, WorkerLinear, WorkerCombined, WorkerConstant, WorkerRandomCircuit


def generate_graphs():
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

    # print('Calculating depth')
    # depths = [get_max_edge_depth(graph) for graph in graphs]
    # histogram = np.histogram(depths, bins=range(1, nodes))
    # print(histogram)
    # return

    # print('Checking isomorphisms')
    # isomorphisms = find_non_isomorphic(graphs)
    # print(f'Number of non-isomorphic: {sum(isomorphisms)}')

    for i in range(len(graphs)):
        nx.write_gml(graphs[i], f'{out_path}/{i}.gml')


def generate_random_edge_graphs(g):
    num_graphs = 10
    max_attempts = 10 ** 3
    nodes = 8
    graph_path = f'graphs/main/all_{nodes}/graph_{g}/{g}.gml'
    out_path = f'graphs/main/all_{nodes}/graph_{g}/random'
    graph = nx.read_gml(graph_path)
    m = nx.number_of_edges(graph)

    graphs = np.empty(num_graphs+1, dtype=object)
    graphs[0] = graph
    graphs_generated = 1
    for i in range(max_attempts):
        next_graph = nx.gnm_random_graph(nodes, m)
        if not nx.is_connected(next_graph):
            continue
        if is_isomorphic(next_graph, graphs[:graphs_generated]):
            continue
        graphs[graphs_generated] = next_graph
        graphs_generated += 1
        print(f'{graphs_generated-1}')
        if graphs_generated == num_graphs+1:
            break
    # else:
    #     raise 'Failed to generate connected set'
    print('Generation done')

    for i in range(graphs_generated-1):
        nx.write_gml(graphs[i+1], f'{out_path}/{i}.gml')


def generate_random_subgraphs(g):
    num_graphs = 10
    max_attempts = 100
    nodes = 8
    graph_path = f'graphs/main/all_{nodes}/graph_{g}/{g}.gml'
    graph = nx.read_gml(graph_path)
    m = nx.number_of_edges(graph)

    out_path = f'graphs/main/all_{nodes}/graph_{g}/pseudo_random'

    edge_frac = [1/4, 1/3, 1/2, 2/3, 3/4]
    edge_count = [int(np.ceil(m * i)) for i in edge_frac]

    graphs = np.empty(num_graphs*len(edge_count), dtype=object)
    total_generated = 0
    for mm in edge_count:
        graphs_generated = 0
        for i in range(max_attempts):
            next_graph = nx.Graph()
            next_graph.add_nodes_from(graph.nodes)
            next_graph.add_edges_from(sample(list(graph.edges),mm))
            # if not nx.is_connected(next_graph):
            #     continue
            if is_isomorphic(next_graph, graphs[:total_generated]):
                continue
            graphs[total_generated] = next_graph
            graphs_generated += 1
            total_generated += 1
            print(f'{total_generated}')
            if graphs_generated == num_graphs:
                break
    # else:
    #     raise 'Failed to generate connected set'
    print('Generation done')

    for i in range(total_generated):
        nx.write_gml(graphs[i], f'{out_path}/{i}.gml')


def generate_remove_triangle_graphs(g):
    num_graphs = 4
    # max_attempts = 10 ** 4
    nodes = 8
    graph_path = f'graphs/main/all_{nodes}/graph_{g}/{g}.gml'
    graph = nx.read_gml(graph_path)

    out_path = f'graphs/main/all_{nodes}/graph_{g}/remove_triangle'

    graphs = np.empty(num_graphs, dtype=object)
    triangles = [clique for clique in nx.enumerate_all_cliques(graph) if len(clique) == 3]
    graphs_generated = 0

    cases = []

    if len(triangles) > 0:
        if len(triangles) == 1:
            common = {(i, j): sorted(nx.common_neighbors(graph, i, j)) for (i, j) in graph.edges}
            ncommon = {e: common[e] for e in common if len(common[e]) > 0}
            sorted_common = sorted(ncommon, key=lambda e: len(ncommon[e]), reverse=True)

            cases.append('most')
            next_graph = graph.copy()
            removed_edge = sample(sorted_common, 1)[0]
            next_graph.remove_edge(removed_edge[0], removed_edge[1])

            graphs[graphs_generated] = next_graph
            print(f'{cases[graphs_generated]}')
            graphs_generated += 1

        else:
            for case in ['all', 'random']:
                if case == 'random':
                    common = {(i, j): sorted(nx.common_neighbors(graph, i, j)) for (i, j) in graph.edges}
                    ncommon = {e: common[e] for e in common if len(common[e]) > 0}
                    sorted_common = sorted(ncommon, key=lambda e: len(ncommon[e]), reverse=True)

                    cases.append('random')
                    next_graph = graph.copy()
                    removed_edge = sample(sorted_common, 1)[0]
                    next_graph.remove_edge(removed_edge[0], removed_edge[1])

                    graphs[graphs_generated] = next_graph
                    print(f'{cases[graphs_generated]}')
                    graphs_generated += 1

                else:
                    common = {(i, j): sorted(nx.common_neighbors(graph, i, j)) for (i, j) in graph.edges}
                    ncommon = {e: common[e] for e in common if len(common[e]) > 0}
                    sorted_common = sorted(ncommon, key=lambda e: len(ncommon[e]), reverse=True)

                    n_removed = 0
                    next_graph = graph.copy()

                    while len(ncommon) > 0:
                        removed_edge = sorted_common[0]
                        next_graph.remove_edge(removed_edge[0], removed_edge[1])
                        n_removed += 1

                        if n_removed == 1:
                            cases.append('most')
                            graphs[graphs_generated] = next_graph.copy()
                            print(f'{cases[graphs_generated]}')
                            graphs_generated += 1

                        if n_removed == 2:
                            cases.append('2_most')
                            graphs[graphs_generated] = next_graph.copy()
                            print(f'{cases[graphs_generated]}')
                            graphs_generated += 1

                        common = {(i, j): sorted(nx.common_neighbors(next_graph, i, j)) for (i, j) in next_graph.edges}
                        ncommon = {e: common[e] for e in common if len(common[e]) > 0}
                        sorted_common = sorted( ncommon, key=lambda e: len(ncommon[e]), reverse=True)

                    cases.append('all')
                    graphs[graphs_generated] = next_graph
                    print(f'{cases[graphs_generated]}')
                    graphs_generated += 1

    print('Generation done')

    for i, c in enumerate(cases):
        nx.write_gml(graphs[i], f'{out_path}/{c}.gml')


def generate_remove_random_edge_from_max_degree_vertex(g):
    num_graphs = 5
    nodes = 8
    max_attempts = 20
    graph_path = f'graphs/main/all_{nodes}/graph_{g}/{g}.gml'
    graph = nx.read_gml(graph_path)

    out_path = f'graphs/main/all_{nodes}/graph_{g}/max_degree_most'
    if not path.exists(out_path):
        os.makedirs(out_path)

    graphs = np.empty(num_graphs+1, dtype=object)
    graphs[0] = graph
    graphs_generated = 1

    max_degree = max(dict(graph.degree).values())
    max_degree_vertices = [v for v, d in graph.degree if d == max_degree]
    for i in range(max_attempts):
        max_degree_vertex = sample(max_degree_vertices, 1)[0]
        edges = list(graph.edges(max_degree_vertex))

        next_graph = graph.copy()
        edge = sample(edges, 1)[0]
        next_graph.remove_edge(*edge)

        if is_isomorphic(next_graph, graphs[:graphs_generated]):
            continue
        graphs[graphs_generated] = next_graph
        graphs_generated += 1
        print(f'{graphs_generated-1}')
        if graphs_generated == num_graphs+1:
            break
    print('Generation done')

    for i in range(graphs_generated-1):
        nx.write_gml(graphs[i+1], f'{out_path}/{i}.gml')


def remove_2_random_edges_from_max_degree_vertex(graph):
    num_graphs = 5
    nodes = 8
    max_attempts = 20
    graph_path = f'graphs/main/all_{nodes}/graph_{g}/{g}.gml'
    graph = nx.read_gml(graph_path)

    out_path = f'graphs/main/all_{nodes}/graph_{g}/max_degree_2_most'
    if not path.exists(out_path):
        os.makedirs(out_path)

    graphs = np.empty(num_graphs+1, dtype=object)
    graphs[0] = graph
    graphs_generated = 1

    max_degree = max(dict(graph.degree).values())
    max_degree_vertices = [v for v, d in graph.degree if d == max_degree]
    for i in range(max_attempts):
        max_degree_vertex = sample(max_degree_vertices, 1)[0]
        edges = list(graph.edges(max_degree_vertex))

        next_graph = graph.copy()
        edge = sample(edges, 2)
        next_graph.remove_edge(*edge[0])
        next_graph.remove_edge(*edge[1])

        if is_isomorphic(next_graph, graphs[:graphs_generated]):
            continue
        graphs[graphs_generated] = next_graph
        graphs_generated += 1
        print(f'{graphs_generated-1}')
        if graphs_generated == num_graphs+1:
            break
    print('Generation done')

    for i in range(graphs_generated-1):
        nx.write_gml(graphs[i+1], f'{out_path}/{i}.gml')


def remove_2_random_edges_from_max_degree_vertex_other(graph):
    num_graphs = 5
    nodes = 8
    max_attempts = 20
    graph_path = f'graphs/main/all_{nodes}/graph_{g}/{g}.gml'
    graph = nx.read_gml(graph_path)

    out_path = f'graphs/main/all_{nodes}/graph_{g}/max_degree_2_most_other'
    if not path.exists(out_path):
        os.makedirs(out_path)

    graphs = np.empty(num_graphs+1, dtype=object)
    graphs[0] = graph
    graphs_generated = 1

    for i in range(max_attempts):
        next_graph = graph.copy()
        for i in range(2):
            max_degree = max(dict(next_graph.degree).values())
            max_degree_vertices = [v for v, d in next_graph.degree if d == max_degree]
            max_degree_vertex = sample(max_degree_vertices, 1)[0]
            edges = list(next_graph.edges(max_degree_vertex))

            edge = sample(edges, 1)[0]
            next_graph.remove_edge(*edge)

        if is_isomorphic(next_graph, graphs[:graphs_generated]):
            continue
        graphs[graphs_generated] = next_graph
        graphs_generated += 1
        print(f'{graphs_generated-1}')
        if graphs_generated == num_graphs+1:
            break
    print('Generation done')

    for i in range(graphs_generated-1):
        nx.write_gml(graphs[i+1], f'{out_path}/{i}.gml')


def remove_all_edges_from_max_degree_vertex(graph):
    num_graphs = 5
    nodes = 8
    max_attempts = 20
    graph_path = f'graphs/main/all_{nodes}/graph_{g}/{g}.gml'
    graph = nx.read_gml(graph_path)

    out_path = f'graphs/main/all_{nodes}/graph_{g}/max_degree_all'
    if not path.exists(out_path):
        os.makedirs(out_path)

    graphs = np.empty(num_graphs+1, dtype=object)
    graphs[0] = graph
    graphs_generated = 1

    max_degree = max(dict(graph.degree).values())
    max_degree_vertices = [v for v, d in graph.degree if d == max_degree]
    for i in range(max_attempts):
        max_degree_vertex = sample(max_degree_vertices, 1)[0]
        edges = list(graph.edges(max_degree_vertex))

        next_graph = graph.copy()
        next_graph.remove_edges_from(edges)

        if is_isomorphic(next_graph, graphs[:graphs_generated]):
            continue
        graphs[graphs_generated] = next_graph
        graphs_generated += 1
        print(f'{graphs_generated-1}')
        if graphs_generated == num_graphs+1:
            break
    print('Generation done')

    for i in range(graphs_generated-1):
        nx.write_gml(graphs[i+1], f'{out_path}/{i}.gml')


def init_dataframe(data_path: str, worker: WorkerBaseQAOA, out_path: str, random_type=None):
    if isinstance(worker, WorkerRandomCircuit):
        paths = [(f'{data_path}graph_{i}/{i}.gml', f'{data_path}graph_{i}/{random_type}/{os.fsdecode(j)}')
                 for i in range(11117) for j in sorted(os.listdir(os.fsencode(f'{data_path}graph_{i}/{random_type}')))]
        # paths = [(f'{data_path}graph_{i}/{i}.gml', f'{data_path}graph_{i}/{i}.gml') for i in range(11117)]
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
    # nodes = list(range(12, 13))
    nodes = [8]
    # depths = list(range(3, 4))
    # ps = list(range(1, 2))
    ps = [1]

    num_workers = 20
    # convergence_threshold = 0.9995
    convergence_threshold = 1.1
    reader = partial(nx.read_gml, destringizer=int)
    p = 1

    # for p in ps:
    # for random_type in ['random', 'pseudo_random', 'remove_triangle']:
    # for random_type in ['max_degree_most', 'max_degree_2_most', 'max_degree_all']:
    for random_type in ['max_degree_2_most_other']:
        out_path = f'results/random_circuit/{random_type}/out_ma.csv'
        out_col = f'p_{p}'

        # initial_guess_from = None if p == 1 else f'p_{p - 1}'
        # initial_guess_from = f'p_{p}'
        # transfer_from = None if p == 1 else f'p_{p - 1}'
        # transfer_p = None if p == 1 else p - 1
        # worker = WorkerStandard(reader=reader, p=p, out_col=f'p_1', initial_guess_from=None, transfer_from=None, transfer_p=None, search_space='qaoa')
        # worker_constant = WorkerConstant(reader=reader, p=p, out_col=out_col, initial_guess_from=None, transfer_from=transfer_from, transfer_p=transfer_p)
        # worker_tqa = WorkerLinear(reader=reader, p=p, out_col=out_col, initial_guess_from=None, transfer_from=transfer_from, transfer_p=transfer_p, search_space='tqa')
        # worker_interp = WorkerInterp(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p, alpha=0.6)
        # worker_fourier = WorkerFourier(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p, alpha=0.6)
        # worker_greedy = WorkerGreedy(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p)
        # worker_combined = WorkerCombined(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p,
        #                                  workers=[worker_interp, worker_greedy], restart_shares=[0.5, 0.5])
        # worker_ma = WorkerMA(reader=reader, p=p, out_col=out_col, initial_guess_from=initial_guess_from, transfer_from=transfer_from, transfer_p=transfer_p,
        #                      guess_provider=None, guess_format='qaoa')
        # worker = worker_ma

        worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, initial_guess_from=None, search_space='qaoa')
        # worker = WorkerRandomCircuit(reader=reader, p=p, out_col=out_col, initial_guess_from=None, search_space='ma')

        for node in nodes:
            # node_depths = [3] if node < 12 else depths
            # for depth in node_depths:
            data_path = f'graphs/main/all_{node}/'

            rows_func = lambda df: np.ones((df.shape[0], 1), dtype=bool) if p == 1 else df[f'p_{p - 1}'] < convergence_threshold
            # rows_func = lambda df: (df[f'p_{p - 1}'] < convergence_threshold) & (df[f'p_{p}'] - df[f'p_{p - 1}'] < 1e-3)
            # rows_func = lambda df: (df[f'p_{p}'] < convergence_threshold) & ((df[f'p_{p}_nfev'] == 1000 * p) | (df[f'p_{p}'] < df[f'p_{p - 1}']))

            # mask = np.zeros((1000, 1), dtype=bool)
            # mask[:] = True
            # rows_func = lambda df: mask

            out_folder = path.split(out_path)[0]
            if not path.exists(out_folder):
                os.makedirs(path.split(out_path)[0])
            if not path.exists(out_path):
                init_dataframe(data_path, worker, out_path, random_type)

            optimize_expectation_parallel(out_path, rows_func, num_workers, worker)


def run_correct():
    nodes = list(range(9, 13))
    depths = list(range(3, 7))
    for node in nodes:
        node_depths = [3] if node < 12 else depths
        for depth in node_depths:
            data_path = f'graphs/new/nodes_{node}/depth_{depth}/output/qaoa/random/p_1/out.csv'
            df = pd.read_csv(data_path, index_col=0)
            for r in range(1, 11):
                for i in range(1000):
                    angles = numpy_str_to_array(df.loc[f'graphs/new/nodes_{node}/depth_{depth}//{i}.gml', f'r_{r}_angles'])
                    angles = angles[angles != 0]
                    df.loc[f'graphs/new/nodes_{node}/depth_{depth}//{i}.gml', f'r_{r}_angles'] = str(angles)
            df.to_csv(data_path)


def run_merge():
    copy_better = True
    nodes = [9]
    depths = [3, 4, 5, 6]
    methods = ['ma']
    ps_all = {'qaoa': list(range(1, 12)), 'ma': list(range(1, 6))}
    convergence_threshold = 0.9995
    for method in methods:
        ps = ps_all[method]
        for node in nodes:
            node_depths = [3] if node < 12 else depths
            for depth in node_depths:
                base_path = f'graphs/new/nodes_{node}/depth_{depth}/output/{method}/random'
                # restarts = [1] * len(ps)
                restarts = ps
                merge_dfs(base_path, ps, restarts, convergence_threshold, f'{base_path}/attempts_p/out.csv', copy_better)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    # generate_graphs()
    # for g in range(11117):
        # print(f'g = {g}')
        # generate_random_edge_graphs(g)
        # generate_random_subgraphs(g)
        # generate_remove_triangle_graphs(g)
    for g in range(11117):
        print(f'g = {g}')
        # generate_remove_random_edge_from_max_degree_vertex(g)
        remove_2_random_edges_from_max_degree_vertex_other(g)
        # remove_all_edges_from_max_degree_vertex(g)
    run_graphs_parallel()
    # run_merge()
    # run_correct()
