"""
Graph utilities.
"""
from queue import SimpleQueue
from typing import Sequence

import os
from os import path

from random import sample
import networkx as nx
import numpy as np
from networkx import Graph
from numpy import ndarray


def edge_bfs(graph: Graph, starting_edge: tuple) -> dict[tuple, int]:
    """
    Carries out edge BFS from the specified edge and returns distances to all other edges.
    :param graph: Graph for BFS.
    :param starting_edge: Starting edge.
    :return: Distances to all edges starting from the given edge.
    """
    if starting_edge[0] > starting_edge[1]:
        starting_edge = starting_edge[::-1]
    distances = {starting_edge: 0}
    bfs_queue = SimpleQueue()
    bfs_queue.put(starting_edge)
    while not bfs_queue.empty():
        parent_edge = bfs_queue.get()
        for node in parent_edge:
            for edge in graph.edges(node):
                if edge[0] > edge[1]:
                    edge = edge[::-1]
                if edge not in distances:
                    distances[edge] = distances[parent_edge] + 1
                    bfs_queue.put(edge)
    return distances


def get_max_edge_depth(graph: Graph) -> int:
    """
    Returns worst case depth of edge BFS.
    :param graph: Graph in question.
    :return: Worst case depth of edge BFS.
    """
    depths = []
    for edge in graph.edges:
        distances = edge_bfs(graph, edge)
        depths.append(max(distances.values()))
    return max(depths)


def get_edge_diameter(graph: Graph) -> int:
    """
    Returns edge diameter of the graph, i.e. maximum number of BFS layers necessary to discover all edges.
    :param graph: Graph.
    :return: Edge diameter.
    """
    peripheral_nodes = nx.periphery(graph)
    diameter = nx.diameter(graph)
    for node in peripheral_nodes:
        last_edge = list(nx.edge_bfs(graph, node))[-1]
        if nx.shortest_path_length(graph, node, last_edge[0]) == diameter and nx.shortest_path_length(graph, node, last_edge[1]) == diameter:
            return diameter + 1
    return diameter


def get_node_indices(graph: Graph) -> dict:
    """
    Returns a dict that maps node labels to their indices.
    :param graph: Graph.
    :return: Dict that maps node labels to their indices.
    """
    return {node: i for i, node in enumerate(graph.nodes)}


def get_index_edge_list(graph: Graph, edge_list: list[tuple[int, int]] = None) -> ndarray:
    """
    Returns 2D array of edges specified by pairs of node indices in the order of graph.nodes instead of node labels.
    :param graph: Graph to consider.
    :param edge_list: List of edges that should be taken into account. If None, then all edges are taken into account.
    :return: 2D array of size graph.edges x 2, where each edge is specified by node indices instead of labels.
    """
    if edge_list is None:
        edge_list = graph.edges

    node_indices = get_node_indices(graph)
    index_edge_list = []
    for edge in edge_list:
        index_edge_list.append([node_indices[edge[0]], node_indices[edge[1]]])
    return np.array(index_edge_list)


def read_graph_xqaoa(path):
    """
    Reads a graph in XQAOA format
    :param path: Path to graph file.
    :return: Read graph.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = lines[2:]
    graph = nx.read_edgelist(lines, delimiter=',', nodetype=int)
    nx.set_edge_attributes(graph, 1, 'weight')
    return graph


def is_isomorphic(graph: Graph, other_graphs: Sequence) -> bool:
    """
    Checks if given graph is isomorphic to any of the other graphs.
    :param graph: Graph to check.
    :param other_graphs: Graphs to check against.
    :return: True if the graph is isomorphic to any of the graphs, False otherwise.
    """
    for i in range(len(other_graphs)):
        if nx.is_isomorphic(graph, other_graphs[i]):
            return True
    return False


def find_non_isomorphic(graphs: Sequence) -> list[bool]:
    """
    Finds non-isomorphic graphs among the given iterable.
    :param graphs: Graphs to search.
    :return: Boolean list such that if True elements are taken from graphs, then none of them will be isomorphic.
    """
    res = [True] * len(graphs)
    for pivot in range(len(graphs)):
        if not res[pivot]:
            continue
        for i in range(pivot + 1, len(graphs)):
            if nx.is_isomorphic(graphs[pivot], graphs[i]):
                res[i] = False
    return res


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
            next_graph.add_edges_from(sample(list(graph.edges), mm))
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
                        sorted_common = sorted(ncommon, key=lambda e: len(ncommon[e]), reverse=True)

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


def remove_2_random_edges_from_max_degree_vertex_other(graph):
    num_graphs = 5
    nodes = 8
    max_attempts = 20
    graph_path = f'graphs/main/all_{nodes}/graph_{graph}/{graph}.gml'
    graph = nx.read_gml(graph_path)

    out_path = f'graphs/main/all_{nodes}/graph_{graph}/max_degree_2_most_other'
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
    graph_path = f'graphs/main/all_{nodes}/graph_{graph}/{graph}.gml'
    graph = nx.read_gml(graph_path)

    out_path = f'graphs/main/all_{nodes}/graph_{graph}/max_degree_all'
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

