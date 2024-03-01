"""
Entry points for test single core uses.
"""
import logging
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from random import sample

from src.graph_utils import get_max_edge_depth, is_isomorphic
from src.data_processing import numpy_str_to_array
from src.optimization import optimize_qaoa_angles, Evaluator
from src.preprocessing import evaluate_graph_cut


def run_add_graph():
    # k = 8
    # n = 15
    # g = nx.random_regular_graph(k, n)

    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    # g.add_edge(0, 3)
    # g.add_edge(1, 4)
    # g.add_edge(1, 5)
    # g.add_edge(4, 5)
    # g.add_edge(2, 6)

    cut_vals = evaluate_graph_cut(g)
    g.graph['max_cut'] = int(max(cut_vals))
    nx.write_gml(g, f'graphs/simple/n7_e7.gml')


def run_point():
    # graph = nx.complete_graph(5)
    graph = nx.read_gml('graphs/nodes_8/ed_4/20.gml', destringizer=int)
    p = 3

    starting_point = np.array([-0.000, -0.000, 0.000, -0.000, 0.250, 0.000, 0.000, 0.000, 0.000, -0.000, 0.000, -0.250, 0.250, -0.000, 0.000, 0.500, 0.000, 0.000, -0.250, 0.000,
                               -0.250, 0.000, 0.250, -0.000, 0.250, 0.000, 0.250, 0.250, 0.000, 0.250, 0.000, -0.250, -0.000, -0.000, 0.250, 0.250, 0.000, 0.000, 0.250, 0.000,
                               0.250, 0.000, 0.000, 0.000, -0.250, 0.000, -0.000, 0.000, -0.000, -0.250, -0.250]) * np.pi

    starting_point = np.array([-0.25, 0.5, 0.25, -0., 0.5, 0.5, 0.5, 0., 0.5, -0.25, 0.5, 0.5, -0.25, 0.5, 0., 0., -0.25, 0.25, 0.5, -0., 0.5, 0., -0.25, -0.25, 0.25, 0.25, 0.,
                               0.25, -0.25, 0.5, 0., -0.25, 0.5, -0., 0., 0., 0., -0.25, 0.5, -0.25, -0., -0.25, 0.5, 0.5, 0., 0., -0.25, -0.25, -0.25, 0.5, 0.25]) * np.pi

    # target_vals = evaluate_graph_cut(graph)
    # driver_term_vals = np.array([evaluate_z_term(np.array([term]), len(graph)) for term in range(len(graph))])
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p)
    res = -evaluator.func(starting_point)

    # res = evaluate_angles_ma_qiskit(angles, graph, p)

    print(f'Expectation: {res}')


def run_optimization(n):
    # graph = nx.read_gml("/home/vilcius//Papers/angle_analysis_ma_qaoa/code/MA-QAOA/graphs/main/all_8/graph_3354/3354.gml", destringizer=int)
    # graph_random = nx.read_gml("/home/vilcius//Papers/angle_analysis_ma_qaoa/code/MA-QAOA/graphs/main/all_8/graph_3354/pseudo_random/18.gml", destringizer=int)
    # graph = nx.read_gml('graphs/main/nodes_9/depth_3/0.gml', destringizer=int)
    # graph = nx.complete_graph(3)
    # graph = read_graph_xqaoa('graphs/xqaoa/G6#128_1.csv')

    graph = nx.star_graph(n-1)
    graph_random = nx.Graph()
    graph_random.add_nodes_from(graph.nodes)
    graph_random.add_edges_from([(i, i+1) for i in range(1,n-1)] + [(n-1, 1)])

    p = 1
    search_space = 'qaoa'

    # target_vals = evaluate_graph_cut(graph)
    # edges = [edge for edge in get_index_edge_list(graph)]
    # # driver_term_vals = np.array([evaluate_z_term(edge, len(graph)) for edge in edges])
    # driver_term_vals = np.array([evaluate_z_term(np.array([node]), len(graph)) for node in range(len(graph))])
    # evaluator = Evaluator.get_evaluator_general(target_vals, driver_term_vals, p, use_multi_angle=use_multi_angle)

    # target_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # target_term_coeffs = [-1 / 2] * len(graph.edges) + [len(graph.edges) / 2]
    # # driver_terms = [set(edge) for edge in get_index_edge_list(graph)]
    # driver_terms = [set(term) for term in it.combinations(range(len(graph)), 1)]
    # # driver_terms = [set(term) for term in it.chain(it.combinations(range(len(graph)), 1), it.combinations(range(len(graph)), 2))]
    # evaluator = Evaluator.get_evaluator_general_subsets(len(graph), target_terms, target_term_coeffs, driver_terms, p)

    # evaluator = Evaluator.get_evaluator_standard_maxcut(graph, p, search_space=search_space)
    evaluator = Evaluator.get_evaluator_random_circuit_maxcut_analytical(graph, graph_random)
    # evaluator = Evaluator.get_evaluator_qiskit_fast(graph, p, search_space)
    # evaluator = Evaluator.get_evaluator_standard_maxcut_analytical(graph, use_multi_angle=True)

    # starting_point = numpy_str_to_array('[-0.17993277 -1.30073361 -1.08469108 -1.59744761]')
    # starting_point = convert_angles_qaoa_to_ma(starting_point, len(graph.edges), len(graph))

    # result = optimize_qaoa_angles(evaluator, starting_angles=starting_point)
    result = optimize_qaoa_angles(evaluator, num_restarts=100, objective_tolerance=1, normalize_angles=False)

    print(f'Best achieved objective: {-result.fun}')
    print(f'Maximizing angles: {repr(result.x)}')

    # expectations = calc_per_edge_expectation(angles_best, driver_term_vals, p, graph, use_multi_angle=use_multi_angle)
    print('Done')


def run_draw_graph(graph):
    # graph = nx.read_gml('graphs/main/all_8/graph_40/pseudo_random/25.gml', destringizer=int)
    # graph = nx.read_gml(gf, destringizer=int)
    nx.draw(graph, with_labels=True)
    plt.show()

def generate_random_subgraphs(g):
    num_graphs = 10
    max_attempts = 40
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

if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger('QAOA')
    np.set_printoptions(linewidth=160, formatter={'float': lambda x: '{:.16f}'.format(x)})

    # Select procedure to run below
    start = time.perf_counter()
    # run_add_graph()
    # run_point()
    for i in range(3,11):
        run_optimization(i)

        print()
    # generate_random_subgraphs(50)
    # n=4
    # graph = nx.star_graph(n-1)
    # graph_random = nx.Graph()
    # graph_random.add_nodes_from(graph.nodes)
    # graph_random.add_edges_from([(i, i+1) for i in range(1,n-1)] + [(n-1, 1)])
    # run_draw_graph(graph)
    # run_draw_graph(graph_random)
    end = time.perf_counter()
    print(f'Elapsed time: {end - start}')
