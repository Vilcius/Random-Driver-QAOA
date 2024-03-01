r"""°°°
# Imports #
°°°"""
#|%%--%%| <uRfF0A5PZo|fH3x81tUZb>

# TODO: find graph highest increase/decrease
# TODO: update averages (at least one better)
# TODO: write results in LaTeX
# TODO: analyse p_1 = 1 graphs

# |%%--%%| <fH3x81tUZb|UUDEsgzKcH>

# from dot2tex import dottex
# from nxpd import nxpdParams
# from nxpd import draw
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import re
import seaborn as sns
# import math
import os
import json
import networkx as nx
from IPython.display import display
from rich_dataframe import prettify
pd.set_option('display.max_columns', 105)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 750)

# |%%--%%| <UUDEsgzKcH|EUuknvcviE>
r"""°°°
Make the $C_{\text{max}}$ dataframe
°°°"""
# |%%--%%| <EUuknvcviE|wk3TkaLFk6>

with open('/home/vilcius/Papers/random_circuit_maxcut/results/qaoa_cmax_n=8.txt') as f:
    f_lines = f.readlines()
    integers = []
    for line in f_lines:
        values = line.strip().split()
        integers.append([int(values[0]) - 1, int(values[1])])

cmax_df = pd.DataFrame(integers, columns=['graph_num', 'maxcut'])
cmax_df

#|%%--%%| <wk3TkaLFk6|2YqPfT3O5c>
r"""°°°
## Make the DataFrames ##
°°°"""
#|%%--%%| <2YqPfT3O5c|3yd7kUDAWs>
r"""°°°
QAOA DataFrame
°°°"""
#|%%--%%| <3yd7kUDAWs|cgmwJLJIXp>

df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/qaoa_df.txt'
result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/qaoa/out.csv'
if os.path.exists(df_filename):
    qaoa_df = pd.read_csv(df_filename)
else:
    qaoa_df = pd.read_csv(result_filename)
    # qaoa_df['graph_num'] = qaoa_df['path'].str.extract(r'graph_(\d+)')
    # qaoa_df['graph_num'] = qaoa_df['graph_num'].astype(int)
    qaoa_df.rename(columns={'p_1': 'qaoa p_1', 'p_1_angles': 'qaoa p1_angles', 'p_1_nfev': 'qaoa p_1 nfev'}, inplace=True)
    qaoa_df.drop(['random_path'], axis=1, inplace=True)
    qaoa_df.to_csv(df_filename, index=False)


#|%%--%%| <cgmwJLJIXp|L1GBRf8dTI>

prettify(qaoa_df.sort_values('qaoa p_1', ascending=False), col_limit=15, row_limit=25)

#|%%--%%| <L1GBRf8dTI|P0E8YhhKq5>
r"""°°°
Triangle Removal DataFrame
°°°"""
#|%%--%%| <P0E8YhhKq5|3tKHSfBqE1>

df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/triangle_removed_df.csv'
result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/remove_triangle/out.csv'
if os.path.exists(df_filename):
    tri_df = pd.read_csv(df_filename)
else:
    tri_df = pd.read_csv(result_filename)
    tri_df['graph_num'] = tri_df['path'].str.extract(r'graph_(\d+)')
    tri_df['graph_num'] = tri_df['graph_num'].astype(int)
    tri_df['case'] = tri_df['random_path'].str.extract(r'(\w+).gml')
    tri_df = tri_df.merge(cmax_df, how='left')
    tri_df.to_csv(df_filename, index=False)
# prettify(tri_df, col_limit=15, row_limit=25)

#|%%--%%| <3tKHSfBqE1|POaffRY0j3>

prettify(tri_df.sort_values('p_1', ascending=False), col_limit=18, row_limit=10)

#|%%--%%| <POaffRY0j3|WpYjfYuFKj>
r"""°°°
Random DataFrame
°°°"""
#|%%--%%| <WpYjfYuFKj|DJs1aMfqQ4>

df_filename = '/home/vilcius/Papers/random_circuit_maxcut/results/random_df.csv'
result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/random/out.csv'
if os.path.exists(df_filename):
    random_df = pd.read_csv(df_filename)
else:
    random_df = pd.read_csv(result_filename)
    random_df['graph_num'] = random_df['path'].str.extract(r'graph_(\d+)')
    random_df['graph_num'] = random_df['graph_num'].astype(int)
    random_df['random_num'] = random_df['random_path'].str.extract(r'(\d+).gml')
    random_df['random_num'] = random_df['random_num'].astype(int)
    random_df = random_df.merge(cmax_df, how='left')
    random_df.to_csv(df_filename, index=False)
# prettify(random_df, col_limit=15, row_limit=25)

#|%%--%%| <DJs1aMfqQ4|RbsGdMt6GV>

prettify(random_df.sort_values('p_1', ascending=False), col_limit=18, row_limit=10)

#|%%--%%| <RbsGdMt6GV|iIAHxCUjxr>
r"""°°°
Subgraph DataFrame
°°°"""
# |%%--%%| <iIAHxCUjxr|GmDT9ib1Nw>

df_filename = '/home/vilcius/Papers/random_circuit_maxcut/results/pseudo_random_df.csv'
result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/subgraph/out.csv'
if os.path.exists(df_filename):
    subgraph_df = pd.read_csv(df_filename)
else:
    subgraph_df = pd.read_csv(result_filename)
    subgraph_df['graph_num'] = subgraph_df['path'].str.extract(r'graph_(\d+)')
    subgraph_df['graph_num'] = subgraph_df['graph_num'].astype(int)
    subgraph_df['random_num'] = subgraph_df['random_path'].str.extract(r'(\d+).gml')
    subgraph_df['random_num'] = subgraph_df['random_num'].astype(int)
    subgraph_df = subgraph_df.merge(cmax_df, how='left')
    subgraph_df.to_csv(df_filename, index=False)
# prettify(subgraph_df, col_limit=15, row_limit=25)

#|%%--%%| <GmDT9ib1Nw|g1f3yc9DPs>

prettify(subgraph_df.sort_values('p_1', ascending=False), col_limit=18, row_limit=10)

#|%%--%%| <g1f3yc9DPs|7zJAgOdVm0>
r"""°°°
Random Max Degree Most Dataframe
°°°"""
#|%%--%%| <7zJAgOdVm0|W4DloxD9hq>

df_filename = '/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_most_df.csv'
result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/max_degree_most/out_ma.csv'
if os.path.exists(df_filename):
    max_degree_most_df = pd.read_csv(df_filename)
else:
    max_degree_most_df = pd.read_csv(result_filename)
    max_degree_most_df['graph_num'] = max_degree_most_df['path'].str.extract(r'graph_(\d+)')
    max_degree_most_df['graph_num'] = max_degree_most_df['graph_num'].astype(int)
    max_degree_most_df['random_num'] = max_degree_most_df['random_path'].str.extract(r'(\d+).gml')
    max_degree_most_df['random_num'] = max_degree_most_df['random_num'].astype(int)
    max_degree_most_df = max_degree_most_df.merge(cmax_df, how='left')
    max_degree_most_df.to_csv(df_filename, index=False)
# prettify(max_degree_most_df, col_limit=15, row_limit=25)


#|%%--%%| <W4DloxD9hq|zK5cnOIS5E>
r"""°°°
Random Max Degree 2 Most Dataframe
°°°"""
#|%%--%%| <zK5cnOIS5E|uxMHAEwZ72>

df_filename = '/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_2_most_df.csv'
result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/max_degree_2_most/out_ma.csv'
if os.path.exists(df_filename):
    max_degree_2_most_df = pd.read_csv(df_filename)
else:
    max_degree_2_most_df = pd.read_csv(result_filename)
    max_degree_2_most_df['graph_num'] = max_degree_2_most_df['path'].str.extract(r'graph_(\d+)')
    max_degree_2_most_df['graph_num'] = max_degree_2_most_df['graph_num'].astype(int)
    max_degree_2_most_df['random_num'] = max_degree_2_most_df['random_path'].str.extract(r'(\d+).gml')
    max_degree_2_most_df['random_num'] = max_degree_2_most_df['random_num'].astype(int)
    max_degree_2_most_df = max_degree_2_most_df.merge(cmax_df, how='left')
    max_degree_2_most_df.to_csv(df_filename, index=False)
# prettify(max_degree_2_most_df, col_limit=15, row_limit=25)


#|%%--%%| <uxMHAEwZ72|kJBqay5iTY>

df_filename = '/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_2_most_other_df.csv'
result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/max_degree_2_most_other/out_ma.csv'
if os.path.exists(df_filename):
    max_degree_2_most_other_df = pd.read_csv(df_filename)
else:
    max_degree_2_most_other_df = pd.read_csv(result_filename)
    max_degree_2_most_other_df['graph_num'] = max_degree_2_most_other_df['path'].str.extract(r'graph_(\d+)')
    max_degree_2_most_other_df['graph_num'] = max_degree_2_most_other_df['graph_num'].astype(int)
    max_degree_2_most_other_df['random_num'] = max_degree_2_most_other_df['random_path'].str.extract(r'(\d+).gml')
    max_degree_2_most_other_df['random_num'] = max_degree_2_most_other_df['random_num'].astype(int)
    max_degree_2_most_other_df = max_degree_2_most_other_df.merge(cmax_df, how='left')
    max_degree_2_most_other_df.to_csv(df_filename, index=False)
# prettify(max_degree_2_most_other_df, col_limit=15, row_limit=25)


#|%%--%%| <kJBqay5iTY|d03SNZcs3R>
r"""°°°
Random Max Degree All DataFrame
°°°"""
#|%%--%%| <d03SNZcs3R|0MGVaXd6Ia>

df_filename = '/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_all_df.csv'
result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/max_degree_all/out_ma.csv'
if os.path.exists(df_filename):
    max_degree_all_df = pd.read_csv(df_filename)
else:
    max_degree_all_df = pd.read_csv(result_filename)
    max_degree_all_df['graph_num'] = max_degree_all_df['path'].str.extract(r'graph_(\d+)')
    max_degree_all_df['graph_num'] = max_degree_all_df['graph_num'].astype(int)
    max_degree_all_df['random_num'] = max_degree_all_df['random_path'].str.extract(r'(\d+).gml')
    max_degree_all_df['random_num'] = max_degree_all_df['random_num'].astype(int)
    max_degree_all_df = max_degree_all_df.merge(cmax_df, how='left')
    max_degree_all_df.to_csv(df_filename, index=False)
# prettify(max_degree_all_df, col_limit=15, row_limit=25)


#|%%--%%| <0MGVaXd6Ia|V8QjgigAC6>
r"""°°°
## Combine DataFrames ##
°°°"""
#|%%--%%| <V8QjgigAC6|db4unoXj0U>
r"""°°°
QAOA Values
°°°"""
#|%%--%%| <db4unoXj0U|124Vu4aS4m>

max_p1_qaoa = qaoa_df['qaoa p_1'].max()
min_p1_qaoa = qaoa_df['qaoa p_1'].min()
average_p1_qaoa = qaoa_df['qaoa p_1'].mean()

#|%%--%%| <124Vu4aS4m|Zu8Kmn6KT2>
r"""°°°
Triangle Removal Compare QAOA
°°°"""
#|%%--%%| <Zu8Kmn6KT2|SBf7jzulfk>

tri_qaoa_df = tri_df.copy()
tri_qaoa_df = tri_qaoa_df.merge(qaoa_df, how='left')
tri_qaoa_df['p_1 diff'] = tri_qaoa_df['p_1'] - tri_qaoa_df['qaoa p_1']
prettify(tri_qaoa_df.sort_values('p_1 diff', ascending=False), col_limit=15, row_limit=25)

qaoa_tri_df = qaoa_df.copy()
qaoa_tri_df = qaoa_tri_df.merge(tri_df, how='left')
prettify(qaoa_tri_df)

#|%%--%%| <SBf7jzulfk|vDyQwyHNfr>

# what percetage of graph nums are such that p_1 is greater than qaoa p_1
percent_better_tri = len(tri_qaoa_df[tri_qaoa_df['p_1'] > tri_qaoa_df['qaoa p_1']]) / len(tri_qaoa_df)
average_diff_tri = tri_qaoa_df['p_1 diff'].mean()
percent_graph_better_tri = len(tri_qaoa_df[tri_qaoa_df['p_1'] > tri_qaoa_df['qaoa p_1']]['graph_num'].unique()) / len(tri_qaoa_df['graph_num'].unique())
print(f'{percent_better_tri}')
print(f'{average_diff_tri}')
print(f'{percent_graph_better_tri}')

#|%%--%%| <vDyQwyHNfr|L31y3Md6UV>

percent_better_most = tri_qaoa_df[(tri_qaoa_df['case'] == 'most') & (tri_qaoa_df['p_1'] > tri_qaoa_df['qaoa p_1'])].shape[0] / tri_qaoa_df[tri_qaoa_df['case'] == 'most'].shape[0]
percent_better_2_most = tri_qaoa_df[(tri_qaoa_df['case'] == '2_most') & (tri_qaoa_df['p_1'] > tri_qaoa_df['qaoa p_1'])].shape[0] / tri_qaoa_df[tri_qaoa_df['case'] == '2_most'].shape[0]
percent_better_all = tri_qaoa_df[(tri_qaoa_df['case'] == 'all') & (tri_qaoa_df['p_1'] > tri_qaoa_df['qaoa p_1'])].shape[0] / tri_qaoa_df[tri_qaoa_df['case'] == 'all'].shape[0]
percent_better_random = tri_qaoa_df[(tri_qaoa_df['case'] == 'random') & (tri_qaoa_df['p_1'] > tri_qaoa_df['qaoa p_1'])].shape[0] / tri_qaoa_df[tri_qaoa_df['case'] == 'random'].shape[0]

print(f'percent better most: {np.round(percent_better_most * 100, 4)}%')
print(f'percent better 2_most: {np.round(percent_better_2_most * 100, 4)}%')
print(f'percent better all: {np.round(percent_better_all * 100, 4)}%')
print(f'percent better random: {np.round(percent_better_random * 100, 4)}%')

average_p_1_most = tri_qaoa_df[tri_qaoa_df['case'] == 'most']['p_1'].mean()
average_p_1_2_most = tri_qaoa_df[tri_qaoa_df['case'] == '2_most']['p_1'].mean()
average_p_1_all = tri_qaoa_df[tri_qaoa_df['case'] == 'all']['p_1'].mean()
average_p_1_random = tri_qaoa_df[tri_qaoa_df['case'] == 'random']['p_1'].mean()

print()
print(f'average p_1 most: {np.round(average_p_1_most, 4)}')
print(f'average p_1 2_most: {np.round(average_p_1_2_most, 4)}')
print(f'average p_1 all: {np.round(average_p_1_all, 4)}')
print(f'average p_1 random: {np.round(average_p_1_random, 4)}')

average_qaoa_p_1_most = tri_qaoa_df[tri_qaoa_df['case'] == 'most']['qaoa p_1'].mean()
average_qaoa_p_1_2_most = tri_qaoa_df[tri_qaoa_df['case'] == '2_most']['qaoa p_1'].mean()
average_qaoa_p_1_all = tri_qaoa_df[tri_qaoa_df['case'] == 'all']['qaoa p_1'].mean()
average_qaoa_p_1_random = tri_qaoa_df[tri_qaoa_df['case'] == 'random']['qaoa p_1'].mean()

print()
print(f'average qaoa p_1 most: {np.round(average_qaoa_p_1_most, 4)}')
print(f'average qaoa p_1 2_most: {np.round(average_qaoa_p_1_2_most, 4)}')
print(f'average qaoa p_1 all: {np.round(average_qaoa_p_1_all, 4)}')
print(f'average qaoa p_1 random: {np.round(average_qaoa_p_1_random, 4)}')

print()
print(f'max p_1 most: {np.round(tri_qaoa_df[tri_qaoa_df["case"] == "most"]["p_1"].max(), 3)}')
print(f'max p_1 2_most: {np.round(tri_qaoa_df[tri_qaoa_df["case"] == "2_most"]["p_1"].max(), 3)}')
print(f'max p_1 all: {np.round(tri_qaoa_df[tri_qaoa_df["case"] == "all"]["p_1"].max(), 3)}')
print(f'max p_1 random: {np.round(tri_qaoa_df[tri_qaoa_df["case"] == "random"]["p_1"].max(), 3)}')

print()
print(f'min p_1 most: {np.round(tri_qaoa_df[tri_qaoa_df["case"] == "most"]["p_1"].min(), 3)}')
print(f'min p_1 2_most: {np.round(tri_qaoa_df[tri_qaoa_df["case"] == "2_most"]["p_1"].min(), 3)}')
print(f'min p_1 all: {np.round(tri_qaoa_df[tri_qaoa_df["case"] == "all"]["p_1"].min(), 3)}')
print(f'min p_1 random: {np.round(tri_qaoa_df[tri_qaoa_df["case"] == "random"]["p_1"].min(), 3)}')

print()
print(f'number of graphs most: {tri_qaoa_df[tri_qaoa_df["case"] == "most"].shape[0]}')
print(f'number of graphs 2_most: {tri_qaoa_df[tri_qaoa_df["case"] == "2_most"].shape[0]}')
print(f'number of graphs all: {tri_qaoa_df[tri_qaoa_df["case"] == "all"].shape[0]}')
print(f'number of graphs random: {tri_qaoa_df[tri_qaoa_df["case"] == "random"].shape[0]}')

print()
print('number of unique graphs most:', len(tri_qaoa_df[tri_qaoa_df["case"] == "most"]["graph_num"].unique()))
print('number of unique graphs 2_most:', len(tri_qaoa_df[tri_qaoa_df["case"] == "2_most"]["graph_num"].unique()))
print('number of unique graphs all:', len(tri_qaoa_df[tri_qaoa_df["case"] == "all"]["graph_num"].unique()))
print('number of unique graphs random:', len(tri_qaoa_df[tri_qaoa_df["case"] == "random"]["graph_num"].unique()))

#|%%--%%| <L31y3Md6UV|9JNniYHwge>
r"""°°°
Random Compare QAOA
°°°"""
#|%%--%%| <9JNniYHwge|QprobCgPDT>

random_qaoa_df = random_df.copy()
random_qaoa_df = random_qaoa_df.merge(qaoa_df, how='left')
random_qaoa_df['p_1 diff'] = random_qaoa_df['p_1'] - random_qaoa_df['qaoa p_1']
prettify(random_qaoa_df.sort_values('p_1 diff', ascending=False), col_limit=15, row_limit=25)

#|%%--%%| <QprobCgPDT|e97vAYdZ9W>

percent_better_random = len(random_qaoa_df[random_qaoa_df['p_1'] > random_qaoa_df['qaoa p_1']]) / len(random_qaoa_df)
average_diff_random = random_qaoa_df['p_1 diff'].mean()
# graph nums that are better
percent_graph_better_random = len(random_qaoa_df[random_qaoa_df['p_1'] > random_qaoa_df['qaoa p_1']]['graph_num'].unique()) / len(random_qaoa_df['graph_num'].unique())
print(f'{percent_better_random}')
print(f'{average_diff_random}')
print(f'{percent_graph_better_random}')

#|%%--%%| <e97vAYdZ9W|Nb3bGLMtty>
r"""°°°
Subgraph Compare QAOA
°°°"""
#|%%--%%| <Nb3bGLMtty|PiDT40CHM2>

subgraph_qaoa_df = subgraph_df.copy()
subgraph_qaoa_df = subgraph_qaoa_df.merge(qaoa_df, how='left')
subgraph_qaoa_df['p_1 diff'] = subgraph_qaoa_df['p_1'] - subgraph_qaoa_df['qaoa p_1']
prettify(subgraph_qaoa_df.sort_values('p_1 diff', ascending=False), col_limit=15, row_limit=25)

#|%%--%%| <PiDT40CHM2|FoLjFNPIx2>

percent_better_pseudo = len(subgraph_qaoa_df[subgraph_qaoa_df['p_1'] > subgraph_qaoa_df['qaoa p_1']]) / len(subgraph_qaoa_df)
average_diff_pseudo = subgraph_qaoa_df['p_1 diff'].mean()
percent_graph_better_pseudo = len(subgraph_qaoa_df[subgraph_qaoa_df['p_1'] > subgraph_qaoa_df['qaoa p_1']]['graph_num'].unique()) / len(subgraph_qaoa_df['graph_num'].unique())
print(f'{percent_better_pseudo}')
print(f'{average_diff_pseudo}')
print(f'{percent_graph_better_pseudo}')

#|%%--%%| <FoLjFNPIx2|kla4C30unj>
r"""°°°
Max Degree Most Compare QAOA
°°°"""
#|%%--%%| <kla4C30unj|23PPWxmp1B>

degree_most_qaoa_df = max_degree_most_df.copy()
degree_most_qaoa_df = degree_most_qaoa_df.merge(qaoa_df, how='left')
degree_most_qaoa_df['p_1 diff'] = degree_most_qaoa_df['p_1'] - degree_most_qaoa_df['qaoa p_1']
prettify(degree_most_qaoa_df.sort_values('p_1 diff', ascending=False), col_limit=15, row_limit=25)

#|%%--%%| <23PPWxmp1B|DzDIc96xm5>

percent_better_most = len(degree_most_qaoa_df[degree_most_qaoa_df['p_1'] > degree_most_qaoa_df['qaoa p_1']]) / len(degree_most_qaoa_df)
average_diff_most = degree_most_qaoa_df['p_1 diff'].mean()
percent_graph_better_most = len(degree_most_qaoa_df[degree_most_qaoa_df['p_1'] > degree_most_qaoa_df['qaoa p_1']]['graph_num'].unique()) / len(degree_most_qaoa_df['graph_num'].unique())
print(f'{percent_better_most}')
print(f'{average_diff_most}')
print(f'{percent_graph_better_most}')

#|%%--%%| <DzDIc96xm5|shTqNQBROw>
r"""°°°
Max Degree 2 Most Compare QAOA
°°°"""
#|%%--%%| <shTqNQBROw|zEy8J9Vf6D>

degree_2_most_qaoa_df = max_degree_2_most_df.copy()
degree_2_most_qaoa_df = degree_2_most_qaoa_df.merge(qaoa_df, how='left')
degree_2_most_qaoa_df['p_1 diff'] = degree_2_most_qaoa_df['p_1'] - degree_2_most_qaoa_df['qaoa p_1']
prettify(degree_2_most_qaoa_df.sort_values('p_1 diff', ascending=False), col_limit=15, row_limit=25)

#|%%--%%| <zEy8J9Vf6D|NKxfOwwK5I>

degree_2_most_other_qaoa_df = max_degree_2_most_other_df.copy()
degree_2_most_other_qaoa_df = degree_2_most_other_qaoa_df.merge(qaoa_df, how='left')
degree_2_most_other_qaoa_df['p_1 diff'] = degree_2_most_other_qaoa_df['p_1'] - degree_2_most_other_qaoa_df['qaoa p_1']
prettify(degree_2_most_other_qaoa_df.sort_values('p_1 diff', ascending=False), col_limit=15, row_limit=25)

#|%%--%%| <NKxfOwwK5I|wU5yC6qUUs>

percent_better_2_most = len(degree_2_most_qaoa_df[degree_2_most_qaoa_df['p_1'] > degree_2_most_qaoa_df['qaoa p_1']]) / len(degree_2_most_qaoa_df)
average_diff_2_most = degree_2_most_qaoa_df['p_1 diff'].mean()
percent_graph_better_2_most = len(degree_2_most_qaoa_df[degree_2_most_qaoa_df['p_1'] > degree_2_most_qaoa_df['qaoa p_1']]['graph_num'].unique()) / len(degree_2_most_qaoa_df['graph_num'].unique())
print(f'{percent_better_2_most}')
print(f'{average_diff_2_most}')
print(f'{percent_graph_better_2_most}')

#|%%--%%| <wU5yC6qUUs|OjSVMTKq1n>

percent_better_2_most_other = len(degree_2_most_other_qaoa_df[degree_2_most_other_qaoa_df['p_1'] > degree_2_most_other_qaoa_df['qaoa p_1']]) / len(degree_2_most_other_qaoa_df)
average_diff_2_most_other = degree_2_most_other_qaoa_df['p_1 diff'].mean()
percent_graph_better_2_most_other = len(degree_2_most_other_qaoa_df[degree_2_most_other_qaoa_df['p_1'] > degree_2_most_other_qaoa_df['qaoa p_1']]['graph_num'].unique()) / len(degree_2_most_other_qaoa_df['graph_num'].unique())
print(f'{percent_better_2_most_other}')
print(f'{average_diff_2_most_other}')
print(f'{percent_graph_better_2_most_other}')


#|%%--%%| <OjSVMTKq1n|bAiHl2Mog7>
r"""°°°
Max Degree All Compare QAOA
°°°"""
#|%%--%%| <bAiHl2Mog7|4U4Al3iW8G>

degree_all_qaoa_df = max_degree_all_df.copy()
degree_all_qaoa_df = degree_all_qaoa_df.merge(qaoa_df, how='left')
degree_all_qaoa_df['p_1 diff'] = degree_all_qaoa_df['p_1'] - degree_all_qaoa_df['qaoa p_1']
prettify(degree_all_qaoa_df.sort_values('p_1 diff', ascending=False), col_limit=15, row_limit=25)

#|%%--%%| <4U4Al3iW8G|7fgAZkdarY>

percent_better_all = len(degree_all_qaoa_df[degree_all_qaoa_df['p_1'] > degree_all_qaoa_df['qaoa p_1']]) / len(degree_all_qaoa_df)
average_diff_all = degree_all_qaoa_df['p_1 diff'].mean()
percent_graph_better_all = len(degree_all_qaoa_df[degree_all_qaoa_df['p_1'] > degree_all_qaoa_df['qaoa p_1']]['graph_num'].unique()) / len(degree_all_qaoa_df['graph_num'].unique())
print(f'{percent_better_all}')
print(f'{average_diff_all}')
print(f'{percent_graph_better_all}')

#|%%--%%| <7fgAZkdarY|7T8NUhk32I>
r"""°°°
Combine Max Degree DataFrames
°°°"""
#|%%--%%| <7T8NUhk32I|LHLxnLl0IY>

degree_df = pd.concat([degree_most_qaoa_df, degree_2_most_other_qaoa_df, degree_all_qaoa_df])
percent_better_degree_graph = len(degree_df[degree_df['p_1'] > degree_df['qaoa p_1']]['graph_num'].unique()) / len(degree_df['graph_num'].unique())
print(f'{percent_better_degree_graph}')

#|%%--%%| <LHLxnLl0IY|yktwXdmAtg>
r"""°°°
Compare results
°°°"""
#|%%--%%| <yktwXdmAtg|PnF7cXnGXS>

max_p1_tri = tri_qaoa_df['p_1'].max()
max_p1_random = random_qaoa_df['p_1'].max()
max_p1_pseudo = subgraph_qaoa_df['p_1'].max()

min_p1_tri = tri_qaoa_df['p_1'].min()
min_p1_random = random_qaoa_df['p_1'].min()
min_p1_pseudo = subgraph_qaoa_df['p_1'].min()

average_p1_tri = tri_qaoa_df['p_1'].mean()
average_p1_random = random_qaoa_df['p_1'].mean()
average_p1_pseudo = subgraph_qaoa_df['p_1'].mean()

max_diff_tri = tri_qaoa_df['p_1 diff'].max()
max_diff_random = random_qaoa_df['p_1 diff'].max()
max_diff_pseudo = subgraph_qaoa_df['p_1 diff'].max()

min_diff_tri = tri_qaoa_df['p_1 diff'].min()
min_diff_random = random_qaoa_df['p_1 diff'].min()
min_diff_pseudo = subgraph_qaoa_df['p_1 diff'].min()

max_p1_degree_most = degree_most_qaoa_df['p_1'].max()
max_p1_degree_2_most = degree_2_most_qaoa_df['p_1'].max()
max_p1_degree_all = degree_all_qaoa_df['p_1'].max()

min_p1_degree_most = degree_most_qaoa_df['p_1'].min()
min_p1_degree_2_most = degree_2_most_qaoa_df['p_1'].min()
min_p1_degree_all = degree_all_qaoa_df['p_1'].min()

average_p1_degree_most = degree_most_qaoa_df['p_1'].mean()
average_p1_degree_2_most = degree_2_most_qaoa_df['p_1'].mean()
average_p1_degree_all = degree_all_qaoa_df['p_1'].mean()

average_diff_degree_most = degree_most_qaoa_df['p_1 diff'].mean()
average_diff_degree_2_most = degree_2_most_qaoa_df['p_1 diff'].mean()
average_diff_degree_all = degree_all_qaoa_df['p_1 diff'].mean()

print(f'For Triangle Removal:\npercent better: {percent_better_tri}, percent graph better: {percent_graph_better_tri},\nmax p_1: {max_p1_tri}, min p_1: {min_p1_tri}, average p_1: {average_p1_tri},\nmax diff: {max_diff_tri}, min diff: {min_diff_tri}, average_dif: {average_diff_tri}\n')
print(f'For Random:\npercent better: {percent_better_random}, percent graph better: {percent_graph_better_random},\nmax p_1: {max_p1_random}, min p_1: {min_p1_random}, average p_1: {average_p1_random},\nmax diff: {max_diff_random}, min diff: {min_diff_random}, average_dif: {average_diff_random}\n')
print(f'For Subgraph:\npercent better: {percent_better_pseudo}, percent graph better: {percent_graph_better_pseudo},\nmax p_1: {max_p1_pseudo}, min p_1: {min_p1_pseudo}, average p_1: {average_p1_pseudo},\nmax diff: {max_diff_pseudo}, min diff: {min_diff_pseudo}, average_dif: {average_diff_pseudo}\n')
#|%%--%%| <PnF7cXnGXS|Xf1IoQA9PS>

max_p1_degree_most = degree_most_qaoa_df['p_1'].max()
max_p1_degree_2_most = degree_2_most_qaoa_df['p_1'].max()
max_p1_degree_2_most_other = degree_2_most_other_qaoa_df['p_1'].max()
max_p1_degree_all = degree_all_qaoa_df['p_1'].max()

min_p1_degree_most = degree_most_qaoa_df['p_1'].min()
min_p1_degree_2_most = degree_2_most_qaoa_df['p_1'].min()
min_p1_degree_2_most_other = degree_2_most_other_qaoa_df['p_1'].min()
min_p1_degree_all = degree_all_qaoa_df['p_1'].min()

average_p1_degree_most = degree_most_qaoa_df['p_1'].mean()
average_p1_degree_2_most = degree_2_most_qaoa_df['p_1'].mean()
average_p1_degree_2_most_other = degree_2_most_other_qaoa_df['p_1'].mean()
average_p1_degree_all = degree_all_qaoa_df['p_1'].mean()

average_diff_degree_most = degree_most_qaoa_df['p_1 diff'].mean()
average_diff_degree_2_most = degree_2_most_qaoa_df['p_1 diff'].mean()
average_diff_degree_2_most_other = degree_2_most_other_qaoa_df['p_1 diff'].mean()
average_diff_degree_all = degree_all_qaoa_df['p_1 diff'].mean()

print(f'For Max Degree Most:\npercent better: {round(percent_better_most * 100, 3)}, percent graph better: {round(percent_graph_better_most * 100, 3)},\nmax p_1: {round(max_p1_degree_most, 3)}, min p_1: {round(min_p1_degree_most, 3)}, average p_1: {round(average_p1_degree_most, 3)},\naverage diff: {round(average_diff_degree_most, 3)}\n')
print(f'For Max Degree 2 Most:\npercent better: {round(percent_better_2_most * 100, 3)}, percent graph better: {round(percent_graph_better_2_most * 100, 3)},\nmax p_1: {round(max_p1_degree_2_most, 3)}, min p_1: {round(min_p1_degree_2_most, 3)}, average p_1: {round(average_p1_degree_2_most, 3)},\naverage diff: {round(average_diff_degree_2_most, 3)}\n')
print(f'For Max Degree 2 most_other:\npercent better: {round(percent_better_2_most_other * 100, 3)}, percent graph better: {round(percent_graph_better_2_most_other * 100, 3)},\nmax p_1: {round(max_p1_degree_2_most_other, 3)}, min p_1: {round(min_p1_degree_2_most_other, 3)}, average p_1: {round(average_p1_degree_2_most_other, 3)},\naverage diff: {round(average_diff_degree_2_most_other, 3)}\n')
print(f'For Max Degree All:\npercent better: {round(percent_better_all * 100, 3)}, percent graph better: {round(percent_graph_better_all * 100, 3)},\nmax p_1: {round(max_p1_degree_all, 3)}, min p_1: {round(min_p1_degree_all, 3)}, average p_1: {round(average_p1_degree_all, 3)},\naverage diff: {round(average_diff_degree_all, 3)}\n')

#|%%--%%| <Xf1IoQA9PS|XgTjE2wKoq>
r"""°°°
Graph Properties
°°°"""
#|%%--%%| <XgTjE2wKoq|tALySgoZbP>

# Properties to test:
# - average degree
# - bipartite
# - connectedness
# - density
# - diameter
# - eulerian
# - number of 5 cycles
# - number of edges
# - number of triangles
# - planarity
#

graph_path = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/'


def apply_max_degree(row):
    G = nx.read_gml(graph_path+row['path'])
    return max([G.degree(n) for n in G.nodes()])


def apply_max_degree_random(row):
    G = nx.read_gml(graph_path+row['random_path'])
    return max([G.degree(n) for n in G.nodes()])


def apply_bipartite(row):
    return nx.algorithms.bipartite.is_bipartite(nx.read_gml(graph_path+row['path']))


def apply_bipartite_random(row):
    return nx.algorithms.bipartite.is_bipartite(nx.read_gml(graph_path+row['random_path']))


def apply_connected(row):
    return nx.is_connected(nx.read_gml(graph_path+row['path']))


def apply_connected_random(row):
    return nx.is_connected(nx.read_gml(graph_path+row['random_path']))


def apply_density(row):
    return nx.density(nx.read_gml(graph_path+row['path']))


def apply_density_random(row):
    return nx.density(nx.read_gml(graph_path+row['random_path']))


def apply_diameter(row):
    return nx.diameter(nx.read_gml(graph_path+row['path']))


def apply_diameter_random(row):
    if not nx.is_connected(nx.read_gml(graph_path+row['random_path'])):
        return 0
    return nx.diameter(nx.read_gml(graph_path+row['random_path']))


def apply_eulerian(row):
    return nx.is_eulerian(nx.read_gml(graph_path+row['path']))


def apply_eulerian_random(row):
    if not nx.is_connected(nx.read_gml(graph_path+row['random_path'])):
        return False
    return nx.is_eulerian(nx.read_gml(graph_path+row['random_path']))


def apply_num_5_cycles(row):
    return len([c for c in nx.simple_cycles(nx.read_gml(graph_path+row['path'])) if len(c) == 5])


def apply_num_5_cycles_random(row):
    return len([c for c in nx.simple_cycles(nx.read_gml(graph_path+row['random_path'])) if len(c) == 5])


def apply_num_edges(row):
    return nx.read_gml(graph_path+row['path']).number_of_edges()


def apply_num_edges_random(row):
    return nx.read_gml(graph_path+row['random_path']).number_of_edges()


def apply_num_triangles(row):
    return len([c for c in nx.enumerate_all_cliques(nx.read_gml(graph_path+row['path'])) if len(c) == 3])


def apply_num_triangles_random(row):
    return len([c for c in nx.enumerate_all_cliques(nx.read_gml(graph_path+row['random_path'])) if len(c) == 3])


def apply_planar(row):
    return nx.is_planar(nx.read_gml(graph_path+row['path']))


def apply_planar_random(row):
    return nx.is_planar(nx.read_gml(graph_path+row['random_path']))


#|%%--%%| <tALySgoZbP|injuKFC7Ss>
r"""°°°
Load Random Circuit dataframes
°°°"""
#|%%--%%| <injuKFC7Ss|Y0vKeQmTWq>


df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/triangle_removed_qaoa_df.csv'
# result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/remove_triangle/out.csv'
if os.path.exists(df_filename):
    tri_qaoa_df = pd.read_csv(df_filename)
else:
    tri_qaoa_df['max_degree'] = tri_qaoa_df.apply(apply_max_degree, axis=1)
    tri_qaoa_df['max_degree_random'] = tri_qaoa_df.apply(apply_max_degree_random, axis=1)
    tri_qaoa_df['num_edges'] = tri_qaoa_df.apply(apply_num_edges, axis=1)
    tri_qaoa_df['num_edges_random'] = tri_qaoa_df.apply(apply_num_edges_random, axis=1)
    tri_qaoa_df['num_triangles'] = tri_qaoa_df.apply(apply_num_triangles, axis=1)
    tri_qaoa_df['num_triangles_random'] = tri_qaoa_df.apply(apply_num_triangles_random, axis=1)
    tri_qaoa_df['diameter'] = tri_qaoa_df.apply(apply_diameter, axis=1)
    tri_qaoa_df['diameter_random'] = tri_qaoa_df.apply(apply_diameter_random, axis=1)
    tri_qaoa_df.to_csv(df_filename, index=False)

# prettify(tri_qaoa_df, col_limit=25, row_limit=25)


#|%%--%%| <Y0vKeQmTWq|w9iETPkCgL>

df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/random_qaoa_df.csv'
# result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/remove_triangle/out.csv'
if os.path.exists(df_filename):
    random_qaoa_df = pd.read_csv(df_filename)
else:
    random_qaoa_df['max_degree'] = random_qaoa_df.apply(apply_max_degree, axis=1)
    random_qaoa_df['max_degree_random'] = random_qaoa_df.apply(apply_max_degree_random, axis=1)
    random_qaoa_df['num_edges'] = random_qaoa_df.apply(apply_num_edges, axis=1)
    random_qaoa_df['num_edges_random'] = random_qaoa_df.apply(apply_num_edges_random, axis=1)
    random_qaoa_df['num_triangles'] = random_qaoa_df.apply(apply_num_triangles, axis=1)
    random_qaoa_df['num_triangles_random'] = random_qaoa_df.apply(apply_num_triangles_random, axis=1)
    random_qaoa_df['diameter'] = random_qaoa_df.apply(apply_diameter, axis=1)
    random_qaoa_df['diameter_random'] = random_qaoa_df.apply(apply_diameter_random, axis=1)
    random_qaoa_df.to_csv(df_filename, index=False)

# prettify(random_qaoa_df, col_limit=25, row_limit=25)

#|%%--%%| <w9iETPkCgL|C6pt7fIx0g>

df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/subgraph_qaoa_df.csv'
# result_filename = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/results/random_circuit/remove_triangle/out.csv'
if os.path.exists(df_filename):
    subgraph_qaoa_df = pd.read_csv(df_filename)
else:
    subgraph_qaoa_df['max_degree'] = subgraph_qaoa_df.apply(apply_max_degree, axis=1)
    subgraph_qaoa_df['max_degree_random'] = subgraph_qaoa_df.apply(apply_max_degree_random, axis=1)
    subgraph_qaoa_df['num_edges'] = subgraph_qaoa_df.apply(apply_num_edges, axis=1)
    subgraph_qaoa_df['num_edges_random'] = subgraph_qaoa_df.apply(apply_num_edges_random, axis=1)
    subgraph_qaoa_df['num_triangles'] = subgraph_qaoa_df.apply(apply_num_triangles, axis=1)
    subgraph_qaoa_df['num_triangles_random'] = subgraph_qaoa_df.apply(apply_num_triangles_random, axis=1)
    subgraph_qaoa_df['diameter'] = subgraph_qaoa_df.apply(apply_diameter, axis=1)
    subgraph_qaoa_df['diameter_random'] = subgraph_qaoa_df.apply(apply_diameter_random, axis=1)
    subgraph_qaoa_df.to_csv(df_filename, index=False)

# prettify(subgraph_qaoa_df, col_limit=25, row_limit=25)


#|%%--%%| <C6pt7fIx0g|vU3j0bh0W3>
r"""°°°
Load Random Max Degree Dataframes
°°°"""
#|%%--%%| <vU3j0bh0W3|WDypEmGt7c>

df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_most_qaoa_df.csv'
if os.path.exists(df_filename):
    degree_most_qaoa_df = pd.read_csv(df_filename)
else:
    degree_most_qaoa_df['max_degree'] = degree_most_qaoa_df.apply(apply_max_degree, axis=1)
    degree_most_qaoa_df['max_degree_random'] = degree_most_qaoa_df.apply(apply_max_degree_random, axis=1)
    degree_most_qaoa_df['num_edges'] = degree_most_qaoa_df.apply(apply_num_edges, axis=1)
    degree_most_qaoa_df['num_edges_random'] = degree_most_qaoa_df.apply(apply_num_edges_random, axis=1)
    degree_most_qaoa_df['num_triangles'] = degree_most_qaoa_df.apply(apply_num_triangles, axis=1)
    degree_most_qaoa_df['num_triangles_random'] = degree_most_qaoa_df.apply(apply_num_triangles_random, axis=1)
    degree_most_qaoa_df['diameter'] = degree_most_qaoa_df.apply(apply_diameter, axis=1)
    degree_most_qaoa_df['diameter_random'] = degree_most_qaoa_df.apply(apply_diameter_random, axis=1)
    degree_most_qaoa_df.to_csv(df_filename, index=False)

# prettify(degree_most_qaoa_df, col_limit=25, row_limit=25)

#|%%--%%| <WDypEmGt7c|yhFu2mySbc>

df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_2_most_qaoa_df.csv'
if os.path.exists(df_filename):
    degree_2_most_qaoa_df = pd.read_csv(df_filename)
else:
    degree_2_most_qaoa_df['max_degree'] = degree_2_most_qaoa_df.apply(apply_max_degree, axis=1)
    degree_2_most_qaoa_df['max_degree_random'] = degree_2_most_qaoa_df.apply(apply_max_degree_random, axis=1)
    degree_2_most_qaoa_df['num_edges'] = degree_2_most_qaoa_df.apply(apply_num_edges, axis=1)
    degree_2_most_qaoa_df['num_edges_random'] = degree_2_most_qaoa_df.apply(apply_num_edges_random, axis=1)
    degree_2_most_qaoa_df['num_triangles'] = degree_2_most_qaoa_df.apply(apply_num_triangles, axis=1)
    degree_2_most_qaoa_df['num_triangles_random'] = degree_2_most_qaoa_df.apply(apply_num_triangles_random, axis=1)
    degree_2_most_qaoa_df['diameter'] = degree_2_most_qaoa_df.apply(apply_diameter, axis=1)
    degree_2_most_qaoa_df['diameter_random'] = degree_2_most_qaoa_df.apply(apply_diameter_random, axis=1)
    degree_2_most_qaoa_df.to_csv(df_filename, index=False)


#|%%--%%| <yhFu2mySbc|cfPidyzH6x>

df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_2_most_other_qaoa_df.csv'
if os.path.exists(df_filename):
    degree_2_most_other_qaoa_df = pd.read_csv(df_filename)
else:
    degree_2_most_other_qaoa_df['max_degree'] = degree_2_most_other_qaoa_df.apply(apply_max_degree, axis=1)
    degree_2_most_other_qaoa_df['max_degree_random'] = degree_2_most_other_qaoa_df.apply(apply_max_degree_random, axis=1)
    degree_2_most_other_qaoa_df['num_edges'] = degree_2_most_other_qaoa_df.apply(apply_num_edges, axis=1)
    degree_2_most_other_qaoa_df['num_edges_random'] = degree_2_most_other_qaoa_df.apply(apply_num_edges_random, axis=1)
    degree_2_most_other_qaoa_df['num_triangles'] = degree_2_most_other_qaoa_df.apply(apply_num_triangles, axis=1)
    degree_2_most_other_qaoa_df['num_triangles_random'] = degree_2_most_other_qaoa_df.apply(apply_num_triangles_random, axis=1)
    degree_2_most_other_qaoa_df['diameter'] = degree_2_most_other_qaoa_df.apply(apply_diameter, axis=1)
    degree_2_most_other_qaoa_df['diameter_random'] = degree_2_most_other_qaoa_df.apply(apply_diameter_random, axis=1)
    degree_2_most_other_qaoa_df.to_csv(df_filename, index=False)

# prettify(degree_2_most_qaoa_df, col_limit=25, row_limit=25)

#|%%--%%| <cfPidyzH6x|CVqrloQUN7>

df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_all_qaoa_df.csv'
if os.path.exists(df_filename):
    degree_all_qaoa_df = pd.read_csv(df_filename)
else:
    degree_all_qaoa_df['max_degree'] = degree_all_qaoa_df.apply(apply_max_degree, axis=1)
    degree_all_qaoa_df['max_degree_random'] = degree_all_qaoa_df.apply(apply_max_degree_random, axis=1)
    degree_all_qaoa_df['num_edges'] = degree_all_qaoa_df.apply(apply_num_edges, axis=1)
    degree_all_qaoa_df['num_edges_random'] = degree_all_qaoa_df.apply(apply_num_edges_random, axis=1)
    degree_all_qaoa_df['num_triangles'] = degree_all_qaoa_df.apply(apply_num_triangles, axis=1)
    degree_all_qaoa_df['num_triangles_random'] = degree_all_qaoa_df.apply(apply_num_triangles_random, axis=1)
    degree_all_qaoa_df['diameter'] = degree_all_qaoa_df.apply(apply_diameter, axis=1)
    degree_all_qaoa_df['diameter_random'] = degree_all_qaoa_df.apply(apply_diameter_random, axis=1)
    degree_all_qaoa_df.to_csv(df_filename, index=False)

# prettify(degree_all_qaoa_df, col_limit=25, row_limit=25)

#|%%--%%| <CVqrloQUN7|ZOlovZggXc>
r"""°°°
Grouping the graph properties
°°°"""
#|%%--%%| <ZOlovZggXc|ym8rbpuMoq>

# group the numeber of graphs properties that have p_1 diff greater than 0
max_degree_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('max_degree').count()['path']
max_degree_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('max_degree').count()['path']
max_degree_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('max_degree_random').count()['path']
max_degree_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('max_degree_random').count()['path']
bipartite_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('bipartite').count()['path']
bipartite_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('bipartite').count()['path']
bipartite_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('bipartite_random').count()['path']
bipartite_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('bipartite_random').count()['path']
connected_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('connected').count()['path']
connected_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('connected').count()['path']
connected_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('connected_random').count()['path']
connected_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('connected_random').count()['path']
density_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('density').count()['path']
density_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('density').count()['path']
density_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('density_random').count()['path']
density_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('density_random').count()['path']
diameter_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('diameter').count()['path']
diameter_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('diameter').count()['path']
diameter_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('diameter_random').count()['path']
diameter_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('diameter_random').count()['path']
eulerian_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('eulerian').count()['path']
eulerian_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('eulerian').count()['path']
eulerian_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('eulerian_random').count()['path']
eulerian_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('eulerian_random').count()['path']
num_5_cycles_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('num_5_cycles').count()['path']
num_5_cycles_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('num_5_cycles').count()['path']
num_5_cycles_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('num_5_cycles_random').count()['path']
num_5_cycles_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('num_5_cycles_random').count()['path']
num_edges_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('num_edges').count()['path']
num_edges_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('num_edges').count()['path']
num_edges_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('num_edges_random').count()['path']
num_edges_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('num_edges_random').count()['path']
num_triangles_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('num_triangles').count()['path']
num_triangles_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('num_triangles').count()['path']
num_triangles_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('num_triangles_random').count()['path']
num_triangles_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('num_triangles_random').count()['path']
planar_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('planar').count()['path']
planar_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('planar').count()['path']
planar_random_tri_g = tri_qaoa_df[tri_qaoa_df['p_1 diff'] > 0].groupby('planar_random').count()['path']
planar_random_tri_l = tri_qaoa_df[tri_qaoa_df['p_1 diff'] < 0].groupby('planar_random').count()['path']

print('For Triangle Removal:\n')
print(f'max_degree greater:\n{max_degree_tri_g}\n')
print(f'max_degree less:\n{max_degree_tri_l}\n')
print(f'max_degree_random greater:\n{max_degree_random_tri_g}\n')
print(f'max_degree_random less:\n{max_degree_random_tri_l}\n')
print(f'bipartite greater:\n{bipartite_tri_g}\n')
print(f'bipartite less:\n{bipartite_tri_l}\n')
print(f'bipartite_random greater:\n{bipartite_random_tri_g}\n')
print(f'bipartite_random less:\n{bipartite_random_tri_l}\n')
print(f'connected greater:\n{connected_tri_g}\n')
print(f'connected less:\n{connected_tri_l}\n')
print(f'connected_random greater:\n{connected_random_tri_g}\n')
print(f'connected_random less:\n{connected_random_tri_l}\n')
print(f'density greater:\n{density_tri_g}\n')
print(f'density less:\n{density_tri_l}\n')
print(f'density_random greater:\n{density_random_tri_g}\n')
print(f'density_random less:\n{density_random_tri_l}\n')
print(f'diameter greater:\n{diameter_tri_g}\n')
print(f'diameter less:\n{diameter_tri_l}\n')
print(f'diameter_random greater:\n{diameter_random_tri_g}\n')


#|%%--%%| <ym8rbpuMoq|KmEq7ABSeE>
r"""°°°
Latex Tables
°°°"""
#|%%--%%| <KmEq7ABSeE|MgpUHBn0tr>


def write_percent_table(filename, percents):
    with open(filename, 'w') as fn:
        fn.write('\\begin{tabular}{|c|c|}\n')
        fn.write('\\hline\n')
        fn.write('Percentage of Results Better & Percentage of Graphs Better \\\\\n')
        fn.write('\\hline\n')
        fn.write(f'{np.round(percents[0], 6) * 100}\\% & {np.round(percents[1], 6) * 100}\\% \\\\\n')
        fn.write('\\hline\n')
        fn.write('\\end{tabular}')


def write_ar_table(filename, ars, ars_qaoa, diffs):
    with open(filename, 'w') as fn:
        fn.write('\\begin{tabular}{|l|c|c|c|}\n')
        fn.write('\\hline\n')
        fn.write('Circuit & Max & Min & Average \\\\\n')
        fn.write('\\hline\n')
        fn.write(f'AR & {np.round(ars[0], 4)} & {np.round(ars[1], 4)} & {np.round(ars[2], 4)} \\\\\n')
        fn.write('\\hline\n')
        fn.write(f'QAOA & {np.round(ars_qaoa[0], 4)} & {np.round(ars_qaoa[1], 4)} & {np.round(ars_qaoa[2], 4)} \\\\\n')
        fn.write('\\hline\n')
        fn.write(f'Difference & {np.round(diffs[0], 4)} & {np.round(diffs[1], 4)} & {np.round(diffs[2], 4)} \\\\\n')
        fn.write('\\hline\n')
        fn.write('\\end{tabular}')

#|%%--%%| <MgpUHBn0tr|4zvrdfJYw4>


table_tri_percent_filename = '/home/vilcius/Papers/random_circuit_maxcut/paper/triangle_percent_table.tex'
table_tri_ar_filename = '/home/vilcius/Papers/random_circuit_maxcut/paper/triangle_ar_table.tex'
percents_tri = [percent_better_tri, percent_graph_better_tri]
ars_tri = [max_p1_tri, min_p1_tri, average_p1_tri]
ars_qaoa = [max_p1_qaoa, min_p1_qaoa, average_p1_qaoa]
diffs_tri = [max_diff_tri, min_diff_tri, average_diff_tri]

write_percent_table(table_tri_percent_filename, percents_tri)
write_ar_table(table_tri_ar_filename, ars_tri, ars_qaoa, diffs_tri)

#|%%--%%| <4zvrdfJYw4|mluH58G2oI>

table_random_percent_filename = '/home/vilcius/Papers/random_circuit_maxcut/paper/random_percent_table.tex'
table_random_ar_filename = '/home/vilcius/Papers/random_circuit_maxcut/paper/random_ar_table.tex'
percents_random = [percent_better_random, percent_graph_better_random]
ars_random = [max_p1_random, min_p1_random, average_p1_random]
ars_qaoa = [max_p1_qaoa, min_p1_qaoa, average_p1_qaoa]
diffs_random = [max_diff_random, min_diff_random, average_diff_random]

write_percent_table(table_random_percent_filename, percents_random)
write_ar_table(table_random_ar_filename, ars_random, ars_qaoa, diffs_random)

#|%%--%%| <mluH58G2oI|97ct22RKLA>

table_pseudo_percent_filename = '/home/vilcius/Papers/random_circuit_maxcut/paper/pseudo_percent_table.tex'
table_pseudo_ar_filename = '/home/vilcius/Papers/random_circuit_maxcut/paper/pseudo_ar_table.tex'
percents_pseudo = [percent_better_pseudo, percent_graph_better_pseudo]
ars_pseudo = [max_p1_pseudo, min_p1_pseudo, average_p1_pseudo]
ars_qaoa = [max_p1_qaoa, min_p1_qaoa, average_p1_qaoa]
diffs_pseudo = [max_diff_pseudo, min_diff_pseudo, average_diff_pseudo]

write_percent_table(table_pseudo_percent_filename, percents_pseudo)
write_ar_table(table_pseudo_ar_filename, ars_pseudo, ars_qaoa, diffs_pseudo)
#|%%--%%| <97ct22RKLA|hU5b6pu1Ow>

pr_p1 = subgraph_qaoa_df[subgraph_qaoa_df['p_1'] == 1]
# prettify(pr_p1, col_limit=25, row_limit=25)
pr_p1[['max_degree', 'max_degree_random', 'bipartite', 'bipartite_random', 'connected', 'connected_random', 'density', 'density_random', 'diameter', 'diameter_random', 'eulerian', 'eulerian_random', 'num_5_cycles', 'num_5_cycles_random', 'num_edges', 'num_edges_random', 'num_triangles', 'num_triangles_random', 'planar', 'planar_random']].describe()

p1_graphs = pr_p1[['path', 'random_path', 'graph_num']]
prettify(pr_p1, col_limit=25, row_limit=25, first_cols=False)


#|%%--%%| <hU5b6pu1Ow|FDOwjpEAuv>
graph_row = 1
print(f'max_degree: {pr_p1["max_degree"].iloc[graph_row]}')
print(f'max_degree_random: {pr_p1["max_degree_random"].iloc[graph_row]}')
print(f'bipartite: {pr_p1["bipartite"].iloc[graph_row]}')
print(f'bipartite_random: {pr_p1["bipartite_random"].iloc[graph_row]}')
print(f'connected: {pr_p1["connected"].iloc[graph_row]}')
print(f'connected_random: {pr_p1["connected_random"].iloc[graph_row]}')
print(f'density: {pr_p1["density"].iloc[graph_row]}')
print(f'density_random: {pr_p1["density_random"].iloc[graph_row]}')
print(f'diameter: {pr_p1["diameter"].iloc[graph_row]}')
print(f'diameter_random: {pr_p1["diameter_random"].iloc[graph_row]}')
print(f'eulerian: {pr_p1["eulerian"].iloc[graph_row]}')
print(f'eulerian_random: {pr_p1["eulerian_random"].iloc[graph_row]}')
print(f'num_5_cycles: {pr_p1["num_5_cycles"].iloc[graph_row]}')
print(f'num_5_cycles_random: {pr_p1["num_5_cycles_random"].iloc[graph_row]}')
print(f'num_edges: {pr_p1["num_edges"].iloc[graph_row]}')
print(f'num_edges_random: {pr_p1["num_edges_random"].iloc[graph_row]}')
print(f'num_triangles: {pr_p1["num_triangles"].iloc[graph_row]}')
print(f'num_triangles_random: {pr_p1["num_triangles_random"].iloc[graph_row]}')
print(f'planar: {pr_p1["planar"].iloc[graph_row]}')
print(f'planar_random: {pr_p1["planar_random"].iloc[graph_row]}')

#|%%--%%| <FDOwjpEAuv|8n82GzUEOS>


def isomorphic_subgraph(random_path):
    graphs_path = '/home/vilcius/Papers/angle_analysis_ma_qaoa/code/MA-QAOA/'
    subgraph = nx.read_gml(graphs_path+random_path)
    iso_graphs = []
    for i in range(11117):
        iso_graph = nx.read_gml(graphs_path+f'graphs/main/all_8/graph_{i}/{i}.gml')
        if nx.is_isomorphic(subgraph, iso_graph):
            iso_graphs.append(i)
            return i

    # return iso_graphs


iso_graphs = p1_graphs['random_path'].apply(isomorphic_subgraph)

prettify(p1_graphs, col_limit=25, row_limit=25)
print(iso_graphs)

#|%%--%%| <8n82GzUEOS|CbO0pSyUEZ>

G = nx.read_gml(graph_path+p1_graphs.iloc[0]['path'])
G_random = nx.read_gml(graph_path+p1_graphs.iloc[0]['random_path'])

print(G.edges())
print(G_random.edges())

#|%%--%%| <CbO0pSyUEZ|JuApFe74tN>

ar_1_file = '/home/vilcius/Papers/random_circuit_maxcut/results/ar_1.txt'

with open(ar_1_file, 'w') as fn:
    for graph, random_graph, graph_num in zip(p1_graphs['path'], p1_graphs['random_path'], p1_graphs['graph_num']):
        G = nx.read_gml(graph_path+graph)
        G_random = nx.read_gml(graph_path+random_graph)
        maxcut = cmax_df['maxcut'][graph_num]
        maxcut_val, maxcut_sol = nx.algorithms.approximation.maxcut.one_exchange(G)
        qaoa_ar = subgraph_qaoa_df['qaoa p_1'][graph_num]
        fn.write(f'graph_num = {graph_num}\n')
        fn.write(f'graph = {G.edges()}\n')
        fn.write(f'random_graph = {G_random.edges()}\n')
        fn.write(f'maxcut = {maxcut}\n')
        fn.write(f'maxcut_sol = {maxcut_sol}\n')
        fn.write(f'qaoa_ar = {qaoa_ar}\n')

        fn.write('\n')

#|%%--%%| <JuApFe74tN|gxubqHwYhn>

for graph, random_graph, graph_num in zip(p1_graphs['path'], p1_graphs['random_path'], p1_graphs['graph_num']):
    G = nx.read_gml(graph_path+graph)
    G_random = nx.read_gml(graph_path+random_graph)
    maxcut = cmax_df['maxcut'][graph_num]
    qaoa_ar = subgraph_qaoa_df['qaoa p_1'][graph_num]
    maxcut_val, maxcut_sol = nx.algorithms.approximation.maxcut.one_exchange(G)
    print(G_random.edges())
    print(maxcut)
    print(maxcut_val)
    print(maxcut_sol)
    print(qaoa_ar)

#|%%--%%| <gxubqHwYhn|qZDzjiFCpN>
r"""°°°
Correlation Matrix
°°°"""
#|%%--%%| <qZDzjiFCpN|ntOUKcUlsi>

most_tri_df = tri_qaoa_df[tri_qaoa_df['case'] == 'most'][[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]
most_2_tri_df = tri_qaoa_df[tri_qaoa_df['case'] == '2_most'][[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]
all_tri_df = tri_qaoa_df[tri_qaoa_df['case'] == 'all'][[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]
random_tri_df = tri_qaoa_df[tri_qaoa_df['case'] == 'random'][[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]

most_tri_df.rename(columns={
    'p_1': 'TR-Driver AR', 'qaoa p_1': 'Cost Driver AR',
    'p_1 diff': 'AR diff',
    'num_triangles_random': 'num_triangles_driver',
    'max_degree_random': 'max_degree_driver',
    'num_edges_random': 'num_edges_driver',
    'diameter_random': 'diameter_driver'
}, inplace=True)
most_2_tri_df.rename(columns={
    'p_1': 'TR-Driver AR', 'qaoa p_1': 'Cost Driver AR',
    'p_1 diff': 'AR diff',
    'num_triangles_random': 'num_triangles_driver',
    'max_degree_random': 'max_degree_driver',
    'num_edges_random': 'num_edges_driver',
    'diameter_random': 'diameter_driver'
}, inplace=True)
all_tri_df.rename(columns={
    'p_1': 'TR-Driver AR', 'qaoa p_1': 'Cost Driver AR',
    'p_1 diff': 'AR diff',
    'num_triangles_random': 'num_triangles_driver',
    'max_degree_random': 'max_degree_driver',
    'num_edges_random': 'num_edges_driver',
    'diameter_random': 'diameter_driver'
}, inplace=True)
random_tri_df.rename(columns={
    'p_1': 'TR-Driver AR', 'qaoa p_1': 'Cost Driver AR',
    'p_1 diff': 'AR diff',
    'num_triangles_random': 'num_triangles_driver',
    'max_degree_random': 'max_degree_driver',
    'num_edges_random': 'num_edges_driver',
    'diameter_random': 'diameter_driver'
}, inplace=True)

# prettify(most_tri_df.sort_values('p_1 diff', ascending = True), col_limit=25, row_limit=50)

# plot correlation matrix for most_tri_df

code_path = '/home/vilcius/Papers/random_circuit_maxcut/code/'
paper_path = '/home/vilcius/Papers/random_circuit_maxcut/paper/'


def plot_corr_matrix(df, title):
    fig, ax = plt.subplots(figsize=(15, 15))
    corr = df.corr()
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr.iloc[0:3], vmax=1, vmin=-1, square=True, annot=True, ax=ax, cmap='coolwarm', center=0, cbar_kws={'location': 'bottom'}, annot_kws={"fontsize": 14})
    # ax.yaxis.set_tick_params(rotation=0)
    # ax.xaxis.set_tick_params(rotation=45, left=True)
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.savefig(f'{paper_path}corr_matrix_{title}.png')
    plt.show()


plot_corr_matrix(most_tri_df, 'tr_most')
plot_corr_matrix(most_2_tri_df, 'tr_2_most')
plot_corr_matrix(all_tri_df, 'tr_all')
plot_corr_matrix(random_tri_df, 'tr_random')

#|%%--%%| <ntOUKcUlsi|W8dneLLmR0>

# sub_p1 = subgraph_qaoa_df[subgraph_qaoa_df['p_1'] == 1]
# sub_p1.drop(columns=['path', 'random_path', 'graph_num', 'maxcut', 'p_1_nfev', 'qaoa p_1 nfev', 'random_num'], inplace=True)
# sub_p1.rename(columns={'p_1': 'subgraph AR', 'qaoa p_1': 'QAOA AR', 'p_1 diff': 'AR diff'}, inplace=True)
rand_qaoa_df = random_qaoa_df[[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]
rand_qaoa_df.rename(columns={
    'p_1': 'Random Driver AR', 'qaoa p_1': 'Cost Driver AR',
    'p_1 diff': 'AR diff',
    'num_triangles_random': 'num_triangles_driver',
    'max_degree_random': 'max_degree_driver',
    'num_edges_random': 'num_edges_driver',
    'diameter_random': 'diameter_driver'
}, inplace=True)
sub_qaoa_df = subgraph_qaoa_df[[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]
sub_qaoa_df.rename(columns={
    'p_1': 'Subgraph Driver AR', 'qaoa p_1': 'Cost Driver AR',
    'p_1 diff': 'AR diff',
    'num_triangles_random': 'num_triangles_driver',
    'max_degree_random': 'max_degree_driver',
    'num_edges_random': 'num_edges_driver',
    'diameter_random': 'diameter_driver'
}, inplace=True)


code_path = '/home/vilcius/Papers/random_circuit_maxcut/code/'
paper_path = '/home/vilcius/Papers/random_circuit_maxcut/paper/'


def plot_corr_matrix_sub(df, title):
    fig, ax = plt.subplots(figsize=(15, 15))
    corr = df.corr()
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr.iloc[0:3], vmax=1, vmin=-1, square=True, annot=True, ax=ax, cmap='coolwarm', center=0, cbar_kws={'location': 'bottom'}, annot_kws={"fontsize": 14})
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.savefig(f'{paper_path}corr_matrix_{title}.png')
    plt.show()


# plot_corr_matrix_sub(sub_p1, 'subgraph')
corr_sub = plot_corr_matrix_sub(sub_qaoa_df, 'subgraph')
corr_rand = plot_corr_matrix_sub(rand_qaoa_df, 'random')


#|%%--%%| <W8dneLLmR0|6sLyUyZvGM>
r"""°°°
Correlation Matrix for Random Max Degree Dataframes
°°°"""
#|%%--%%| <6sLyUyZvGM|41nKZP5lxO>

deg_most_qaoa_df = degree_most_qaoa_df[[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]
deg_most_qaoa_df.rename(columns={
                        'p_1': 'MDER-1 Driver AR', 'qaoa p_1': 'Cost Driver AR',
                        'p_1 diff': 'AR diff',
                        'num_triangles_random': 'num_triangles_driver',
                        'max_degree_random': 'max_degree_driver',
                        'num_edges_random': 'num_edges_driver',
                        'diameter_random': 'diameter_driver'
                        }, inplace=True)

deg_2_most_qaoa_df = degree_2_most_qaoa_df[[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]
deg_2_most_qaoa_df.rename(columns={
    'p_1': 'Max Degree 2 Driver AR', 'qaoa p_1': 'Cost Driver AR',
    'p_1 diff': 'AR diff',
    'num_triangles_random': 'num_triangles_driver',
    'max_degree_random': 'max_degree_driver',
    'num_edges_random': 'num_edges_driver',
    'diameter_random': 'diameter_driver'
}, inplace=True)

deg_2_most_other_qaoa_df = degree_2_most_other_qaoa_df[[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]
deg_2_most_other_qaoa_df.rename(columns={
    'p_1': 'MDER-2 Driver AR', 'qaoa p_1': 'Cost Driver AR',
    'p_1 diff': 'AR diff',
    'num_triangles_random': 'num_triangles_driver',
    'max_degree_random': 'max_degree_driver',
    'num_edges_random': 'num_edges_driver',
    'diameter_random': 'diameter_driver'
}, inplace=True)

deg_all_qaoa_df = degree_all_qaoa_df[[
    'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
]]
deg_all_qaoa_df.rename(columns={
    'p_1': 'MDER-All Driver AR', 'qaoa p_1': 'Cost Driver AR',
    'p_1 diff': 'AR diff',
    'num_triangles_random': 'num_triangles_driver',
    'max_degree_random': 'max_degree_driver',
    'num_edges_random': 'num_edges_driver',
    'diameter_random': 'diameter_driver'
}, inplace=True)


code_path = '/home/vilcius/Papers/random_circuit_maxcut/code/'
paper_path = '/home/vilcius/Papers/random_circuit_maxcut/paper/'


def plot_corr_matrix_sub(df, title):
    fig, ax = plt.subplots(figsize=(15, 15))
    corr = df.corr()
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr.iloc[0:3], vmax=1, vmin=-1, square=True, annot=True, ax=ax, cmap='coolwarm', center=0, cbar_kws={'location': 'bottom'}, annot_kws={"fontsize": 14})
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.savefig(f'{paper_path}corr_matrix_{title}.png')
    plt.show()


deg_most_corr = plot_corr_matrix_sub(deg_most_qaoa_df, 'deg_most')
deg_2_most_other_corr = plot_corr_matrix_sub(deg_2_most_other_qaoa_df, 'deg_2_most_other')
deg_all_corr = plot_corr_matrix_sub(deg_all_qaoa_df, 'deg_all')

#|%%--%%| <41nKZP5lxO|4ItkTZwgDk>
r"""°°°
All Correlation Matrices
°°°"""
#|%%--%%| <4ItkTZwgDk|1MAi4UtFHv>

print('Most Triangle Removal')
plot_corr_matrix(most_tri_df, 'tr_most')
print('2 Most Triangle Removal')
plot_corr_matrix(most_2_tri_df, 'tr_2_most')
print('All Triangle Removal')
plot_corr_matrix(all_tri_df, 'tr_all')
print('Random Triangle Removal')
plot_corr_matrix(random_tri_df, 'tr_random')

print('Subgraph')
plot_corr_matrix_sub(sub_qaoa_df, 'subgraph')
print('Random')
plot_corr_matrix_sub(rand_qaoa_df, 'random')

print('Most Degree')
plot_corr_matrix_sub(deg_most_qaoa_df, 'deg_most')
print('2 Most Other Degree')
plot_corr_matrix_sub(deg_2_most_other_qaoa_df, 'deg_2_most_other')
print('All Degree')
plot_corr_matrix_sub(deg_all_qaoa_df, 'deg_all')

#|%%--%%| <1MAi4UtFHv|QIeucxz9vT>

# df_filename = f'/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_2_most_other_qaoa_df.csv'


def read_in_df(df_filename):
    if os.path.exists(df_filename):
        df = pd.read_csv(df_filename)
    else:
        df['max_degree'] = df.apply(apply_max_degree, axis=1)
        df['max_degree_random'] = df.apply(apply_max_degree_random, axis=1)
        df['num_edges'] = df.apply(apply_num_edges, axis=1)
        df['num_edges_random'] = df.apply(apply_num_edges_random, axis=1)
        df['num_triangles'] = df.apply(apply_num_triangles, axis=1)
        df['num_triangles_random'] = df.apply(apply_num_triangles_random, axis=1)
        df['diameter'] = df.apply(apply_diameter, axis=1)
        df['diameter_random'] = df.apply(apply_diameter_random, axis=1)
        df.to_csv(df_filename, index=False)

    return df


random_qaoa_df = read_in_df('/home/vilcius/Papers/random_circuit_maxcut/results/random_qaoa_df.csv')
subgraph_qaoa_df = read_in_df('/home/vilcius/Papers/random_circuit_maxcut/results/subgraph_qaoa_df.csv')
tri_qaoa_df = read_in_df('/home/vilcius/Papers/random_circuit_maxcut/results/triangle_removed_qaoa_df.csv')
degree_most_qaoa_df = read_in_df('/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_most_qaoa_df.csv')
degree_2_most_other_qaoa_df = read_in_df('/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_2_most_other_qaoa_df.csv')
degree_all_qaoa_df = read_in_df('/home/vilcius/Papers/random_circuit_maxcut/results/max_degree_all_qaoa_df.csv')


#|%%--%%| <QIeucxz9vT|vNTcoJvcNo>

def make_new_df(df, title, driver_name):
    new_qaoa_df = qaoa_df.copy()
    df.drop(columns=['qaoa p_1', 'p_1 diff'])
    new_qaoa_df = new_qaoa_df.merge(df, how='left')
    new_qaoa_df['p_1 diff'] = new_qaoa_df['p_1'] - new_qaoa_df['qaoa p_1']
    new_qaoa_df = new_qaoa_df[[
        'p_1', 'qaoa p_1', 'p_1 diff', 'num_triangles', 'num_triangles_random', 'max_degree', 'max_degree_random', 'num_edges', 'num_edges_random', 'diameter', 'diameter_random'
    ]]
    qaoa_p1 = new_qaoa_df['qaoa p_1']
    diff = new_qaoa_df['p_1 diff']
    new_qaoa_df.drop(columns=['qaoa p_1'], inplace=True)
    new_qaoa_df.drop(columns=['p_1 diff'], inplace=True)
    new_qaoa_df.insert(1, 'qaoa p_1', qaoa_p1)
    new_qaoa_df.insert(2, 'p_1 diff', diff)
    new_qaoa_df.rename(columns={
        'p_1': f'{driver_name} Driver AR', 'qaoa p_1': 'Cost Driver AR',
        'p_1 diff': 'AR diff',
        'num_triangles_random': 'num_triangles_driver',
        'max_degree_random': 'max_degree_driver',
        'num_edges_random': 'num_edges_driver',
        'diameter_random': 'diameter_driver'
    }, inplace=True)
    plot_corr_matrix_sub(new_qaoa_df, title)


make_new_df(degree_most_qaoa_df, 'deg_most', 'MDER-1')
make_new_df(degree_2_most_other_qaoa_df, 'deg_2_most_other', 'MDER-2')
