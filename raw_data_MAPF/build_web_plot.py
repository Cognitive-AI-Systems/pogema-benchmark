import yaml
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import pandas as pd
import glob

from scipy.interpolate import PchipInterpolator


def load_and_normalize_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    return df


def add_algorithm_prefix(df):
    algorithm_name = df['algorithm'].iloc[0]
    if "tuned" in algorithm_name:
        algorithm_name = algorithm_name.replace("-tuned", "")
    metric_columns = [col for col in df.columns if col.startswith('metrics.')]
    new_column_names = {col: f"{algorithm_name}_{col.split('.')[-1]}" for col in metric_columns}
    df.rename(columns=new_column_names, inplace=True)
    return df


def get_combined_df(dataset, path):
    json_files = glob.glob(f'{path}/{dataset}/*.json')
    print(json_files)
    dataframes = [load_and_normalize_json(file) for file in json_files]
    dataframes = [add_algorithm_prefix(df) for df in dataframes]
    combined_df = dataframes[0]
    for df in dataframes[1:]:
        if dataset == '05-puzzles':
            combined_df = pd.merge(combined_df, df, on=['env_grid_search.num_agents', 'env_grid_search.seed',
                                                        'env_grid_search.map_name'], suffixes=('', '_dup'))
        elif dataset == '03-warehouse':
            combined_df = pd.merge(combined_df, df, on=['env_grid_search.num_agents', 'env_grid_search.seed'],
                                   suffixes=('', '_dup'))
        elif dataset in ['01-random', '02-mazes', '04-movingai']:
            combined_df = pd.merge(combined_df, df, on=['env_grid_search.num_agents', 'env_grid_search.map_name'],
                                   suffixes=('', '_dup'))
        elif dataset in ['06-pathfinding']:
            combined_df = pd.merge(combined_df, df, on=['env_grid_search.seed', 'env_grid_search.map_name'],
                                   suffixes=('', '_dup'))

    # Drop duplicate columns resulting from the merge
    combined_df = combined_df.loc[:, ~combined_df.columns.str.endswith('_dup')]
    if 'algorithm' in combined_df.columns:
        combined_df.drop(columns=['algorithm'], inplace=True)
    return combined_df


def add_coopeartion(data_dict, algos, combined_df):
    ratios = {algo: [] for algo in algos}
    for index, row in combined_df.iterrows():
        values = []
        for algo in algos:
            if row[f'{algo}_CSR'] > 0:
                values.append(row[f'{algo}_SoC'])
        if len(values) > 0:
            best_value = min(values)
            for algo in algos:
                if row[f'{algo}_CSR'] > 0:
                    ratios[algo].append(best_value / row[f'{algo}_SoC'])
                else:
                    ratios[algo].append(0)
    for algo in algos:
        data_dict[algo]['Cooperation'] = np.array(ratios[algo]).mean()


def add_scalability(data_dict, algos, combined_df):
    for algo in algos:
        runtimes = []
        for n in combined_df['env_grid_search.num_agents'].unique().tolist():
            filtered_df = combined_df[combined_df['env_grid_search.num_agents'] == n]
            runtime = (filtered_df[f'{algo}_runtime'] / filtered_df[f'{algo}_makespan']).mean()
            runtimes.append((n, runtime))
        scaled_runtimes = [(agents, runtime / agents) for agents, runtime in runtimes]

        ratios = []
        for (agents1, scaled_runtime1), (agents2, scaled_runtime2) in itertools.combinations(scaled_runtimes, 2):
            ratio = scaled_runtime1 / scaled_runtime2
            ratios.append(ratio)

        data_dict[algo]['Scalability'] = min(1.0, np.array(ratios).mean())


def add_congestion(data_dict, algos, combined_df, path_to_maps):
    with open(f'{path_to_maps}/maps.yaml', 'r') as f:
        maps = yaml.safe_load(f)
    traversable_cells = {}
    for m in maps:
        cells = 0
        for i in maps[m]:
            if i in ['.', '@', '&', '$', '!']:
                cells += 1
        traversable_cells[m] = cells
    num_agents = combined_df['env_grid_search.num_agents'].max()
    filtered_df = combined_df[
        combined_df['env_grid_search.num_agents'] == combined_df['env_grid_search.num_agents'].max()]
    for algo in algos:
        density = []
        if len(traversable_cells) == 1:
            for index, row in filtered_df.iterrows():
                density.append((float(num_agents) / cells) / row[f'{algo}_avg_agents_density'])
        else:
            for index, row in filtered_df.iterrows():
                density.append((float(num_agents) / traversable_cells[row['env_grid_search.map_name']]) / row[
                    f'{algo}_avg_agents_density'])
        data_dict[algo]['Congestion'] = np.array(density).mean()


def add_out_of_distribution(data_dict, algos, combined_df):
    ratios = {algo: [] for algo in algos}
    for index, row in combined_df.iterrows():
        values = []
        for algo in algos:
            if row[f'{algo}_CSR'] > 0:
                values.append(row[f'{algo}_SoC'])
        if len(values) > 0:
            best_value = min(values)
            for algo in algos:
                if row[f'{algo}_CSR'] > 0:
                    ratios[algo].append(best_value / row[f'{algo}_SoC'])
                else:
                    ratios[algo].append(0)
    for algo in algos:
        data_dict[algo]['Out-of-Distribution'] = np.array(ratios[algo]).mean()


def add_performance(data_dict, algos, combined_df):
    ratios = {algo: [] for algo in algos}
    for index, row in combined_df.iterrows():
        values = []
        for algo in algos:
            if row[f'{algo}_CSR'] > 0:
                values.append(row[f'{algo}_SoC'])
        if len(values) > 0:
            best_value = min(values)
            for algo in algos:
                if row[f'{algo}_CSR'] > 0:
                    ratios[algo].append(best_value / row[f'{algo}_SoC'])
                else:
                    ratios[algo].append(0)
    for algo in algos:
        data_dict[algo]['Performance'] = np.array(ratios[algo]).mean()


def add_pathfinding(data_dict, algos, combined_df):
    results = {algo: [] for algo in algos}
    for index, row in combined_df.iterrows():
        values = []
        for algo in algos:
            values.append(row[f'{algo}_makespan'])
        best_value = min(values)
        for algo in algos:
            if row[f'{algo}_makespan'] > 0:
                results[algo].append(best_value/row[f'{algo}_makespan'])
            else:
                results[algo].append(0)
    for algo in algos:
        data_dict[algo]['Pathfinding'] = np.array(results[algo]).mean()

def add_coordination(data_dict, algos, combined_df):
    results = {algo: 0 for algo in algos}
    
    for algo in algos:
        values = []
        if f'{algo}_a_collisions' in combined_df.columns:   
            for index, row in combined_df.iterrows():
                values.append((row[f'{algo}_a_collisions'] + row[f'{algo}_o_collisions']) / (row[f'{algo}_ep_length'] * row['env_grid_search.num_agents']))
            results[algo] = np.array(values).mean()
        else:
            print(f'{algo} does not have collision data')

    for algo in algos:
        data_dict[algo]['Coordination'] = 1 - results[algo]

def smooth_between_pairs(scores, angles):
    scores = np.array(scores)
    angles = np.array(angles)

    smooth_angles = np.linspace(angles[0], angles[-1], 200)

    interp_func = PchipInterpolator(angles, scores)
    smooth_scores = interp_func(smooth_angles)

    return smooth_angles, smooth_scores


def draw_web(data_dict, labels, draw_dashed=(), filename='web_plot.pdf'):
    data = {algo: [data_dict[algo][l] * 100 for l in labels] for algo in data_dict}
    for key in data:
        data[key] += data[key][:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [2 * np.pi]
    algorithm_colors = {
        'LaCAM': '#1f77b4',    # blue
        'DCC': '#2ca02c',      # green
        'VDN': '#9467bd',      # purple
        'QPLEX': '#e377c2',    # pink
        'SCRIMP': '#ff7f0e',   # orange
        'IQL': '#d62728',      # red
        'QMIX': '#8c564b',     # brown
        'MAMBA': '#7f7f7f'     # gray
    }
    colormap = cm.get_cmap('tab20', len(data))
    for idx, (label, values) in enumerate(data.items()):
        smooth_angles, smooth_values = smooth_between_pairs(values, angles)
        color = algorithm_colors[label]
        linestyle = '--' if label in draw_dashed else 'solid'
        ax.plot(smooth_angles, smooth_values, linewidth=6, alpha=0.9, linestyle=linestyle, label=label, color=color,
                zorder=2)

        ax.plot(angles[:-1], values[:-1], 'o', color=color, zorder=2, markersize=12)
        # if label not in draw_dashed:
        #     ax.fill(smooth_angles, smooth_values, alpha=0.1, color=color, zorder=1)

    ax.set_ylim(0, 101)
    ax.set_yticklabels([''] * 5)
    y_labels = [20, 40, 60, 80, 100]
    y_ticks = [25, 44, 64, 84.5, 108]
    for y_tick, y_label in zip(y_ticks, y_labels):
        ax.text(np.pi / 6, y_tick, str(y_label), horizontalalignment='center', size=26, zorder=3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([''] * len(labels))
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 102, label, horizontalalignment='center', size=32, zorder=3)
    ax.grid(zorder=0)
    ax.spines['polar'].set_visible(False)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=20, ncol=4)

    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()


def main():
    path_to_results = '.'
    algos = ['LaCAM', 'SCRIMP', 'DCC', 'IQL', 'VDN', 'QMIX', 'QPLEX', 'MAMBA']
    #labels = ['Scalability', 'Pathfinding', 'Cooperation', 'Out-of-Distribution', 'Performance', 'Coordination' ]
    labels = ['Cooperation', 'Coordination', 'Out-of-Distribution', 'Pathfinding', 'Performance', 'Scalability']
    centralized = ['LaCAM']

    data_dict = {algo: {} for algo in algos}
    add_coopeartion(data_dict, algos, get_combined_df('05-puzzles', path_to_results))
    add_scalability(data_dict, algos, get_combined_df('03-warehouse', path_to_results))
    #add_congestion(data_dict, algos, get_combined_df('03-warehouse', path_to_results), f'{path_to_results}/03-warehouse')
    add_out_of_distribution(data_dict, algos, get_combined_df('04-movingai', path_to_results))
    add_performance(data_dict, algos, pd.concat(
        [get_combined_df('01-random', path_to_results), get_combined_df('02-mazes', path_to_results)],
        ignore_index=True))
    add_pathfinding(data_dict, algos, get_combined_df('06-pathfinding', path_to_results))
    add_coordination(data_dict, algos, pd.concat(
        [get_combined_df('01-random', path_to_results), get_combined_df('02-mazes', path_to_results)],
        ignore_index=True))

    for algo in algos:
        print(algo, data_dict[algo])
    draw_web(data_dict, labels, filename='MAPF_web.pdf', draw_dashed=centralized)


if __name__ == '__main__':
    main()
