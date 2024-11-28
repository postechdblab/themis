import sys, os
import pandas as pd
from pygnuplot import gnuplot
import numpy as np
exp_id, mode = sys.argv[1], sys.argv[2]

exp_dir = f'./exp_result/{exp_id}'

os.makedirs(f'./plots', exist_ok=True)
os.makedirs(f'./plots/{exp_id}', exist_ok=True)


def init_info(queries, indices):
    df_intra_warp_idle_ratio = {
        'Pyper':  [[0] * len(queries) for sf in indices],
        'Pyper-w/o-ibr':  [[0] * len(queries) for sf in indices],
        'DogQC++':  [[0] * len(queries) for sf in indices],
        'Themis-w/o-aws':  [[0] * len(queries) for sf in indices],
        'Themis': [[0] * len(queries) for sf in indices],
    }
    
    df_inter_warp_load_imbalance = {
        'Pyper': [[0] * len(queries) for sf in indices],
        'Pyper-w/o-ibr': [[0] * len(queries) for sf in indices],
        'DogQC++': [[0] * len(queries) for sf in indices],
        'Themis-w/o-aws': [[0] * len(queries) for sf in indices],
        'Themis': [[0] * len(queries) for sf in indices],
    }
    
    df_elapsed_time = {
        'Pyper': [[0] * len(queries) for sf in indices],
        'Pyper-w/o-ibr': [[0] * len(queries) for sf in indices],
        'DogQC++': [[0] * len(queries) for sf in indices],
        'Themis-w/o-aws': [[0] * len(queries) for sf in indices],
        'Themis': [[0] * len(queries) for sf in indices],
    }
    
    return df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance

def extract_kernel_time(log_path):
    lines = open(log_path).readlines()
    elapsedTime = None
    lines = list(filter(lambda x: 'totalKernelTime' in x, lines))                    
    s = 0
    for line in lines:
        l = line.strip()
        l = l[l.find(':') + 1:]
        l = l.replace("ms","").strip()
        elapsedTime = float(l)
        s += elapsedTime

    if len(lines) == 0:
        elapsedTime = 0
    else:
        elapsedTime = s / len(lines)
    return elapsedTime

def extract_lif_and_idle_ratio(log_path):
    lines = open(log_path).readlines()
    num_oracle_lanes, num_active_lanes = 0, 0
    imbalances = []
    sum_active_clocks, max_active_clocks = [], []
    kernelTimes = []
    real_lane_activities, oracle_lane_activities = [], []
    num_active_warps, num_warps = [], []
    #print(log_path)
    if len(list(filter(lambda x: 'totalKernelTime' in x, lines))) == 0:
        return 0, 0
    for line in lines:
        l = line.strip()
        if 'active_lanes_nums' in line[:200]:
            if True: #ㅂq and system != 'lb_themis_glvl_twolvlbitmaps':
                if '_tail' not in line[:200]:
                    start = l.find('active_lanes_nums:') + len('active_lanes_nums:')
                    es = l[start:].strip().split(' ')
                    es = list(map(lambda x: int(x), es))
                    sum_lanes = sum(es)
                    if 'oracle' in l[:start]: 
                        num_oracle_lanes += sum_lanes
                        oracle_lane_activities.append(sum_lanes)                                    
                    else: 
                        num_active_lanes += sum_lanes
                        real_lane_activities.append(sum_lanes)
                else:
                    start = l.find('active_lanes_nums:') + len('active_lanes_nums:')
                    es = l[start:].strip().split(' ')
                    es = list(map(lambda x: int(x), es))
                    sum_lanes = sum(es)
                    if 'oracle' in l[:start]: 
                        num_oracle_lanes += sum_lanes
                        oracle_lane_activities[-1] += sum_lanes
                    else: 
                        num_active_lanes += sum_lanes
                        real_lane_activities[-1] += sum_lanes
        elif 'active_clocks' in line[:200]: # and system != 'themis':
            #if 'themis' in system: print(system, line)
            if '_tail' not in line[:200]:
                start = l.find('active_clocks:') + len('active_clocks:')
                es = list(map(lambda x: int(x), l[start:].strip().split(' ')))
                num_warps.append(len(es))
                es = list(filter(lambda y : y > 0, es))
                s = sum(es)                            
                sum_active_clocks.append(s)
                m = max(es)                                
                max_warp_id = es.index(m)
                #print('max warp id', line[:30], max_warp_id)
                max_active_clocks.append(m)
                num_active_warps.append(len(es))
            else:
                start = l.find('active_clocks:') + len('active_clocks:')
                es = list(map(lambda x: int(x), l[start:].strip().split(' ')))
                num_warps[-1] += len(es)
                es = list(filter(lambda y : y > 0, es))
                s = sum(es)
                sum_active_clocks[-1] += s
                m = max(es)
                max_active_clocks[-1] = max(max_active_clocks[-1], m)                            
                num_active_warps[-1] += len(es)
        elif len(line) < 200 and '_tail' not in line[:200] and 'ms' in line:
            if l[-2:] != 'ms': continue       
            if 'krnl' in line or 'KernelTime' in line:
                l = line.strip()
                l = l[l.find(':') + 1:]
                l = l.replace("ms","").strip()
                elapsedTime = float(l)
                kernelTimes.append(elapsedTime)

    totalKernelTime = sum(kernelTimes[:len(num_active_warps)])                    
    totalSumClocks = sum(sum_active_clocks[:len(num_active_warps)])
    lif = sum(map(lambda x: sum_active_clocks[x] * (max_active_clocks[x] / sum_active_clocks[x] * num_active_warps[x]) / totalSumClocks, filter(lambda y: num_active_warps[y] > 0,  [i for i in range(len(max_active_clocks))])))                
    
    # if 'themis' in log_path:     
    #     print(log_path, lif, max_active_clocks, sum_active_clocks, num_active_warps)
    #     print(list(map(lambda x: sum_active_clocks[x] / num_active_warps[x], list(range(len(num_active_warps))))))
    #     print(list(map(lambda x: max_active_clocks[x] / sum_active_clocks[x] * num_active_warps[x], list(range(len(num_active_warps))))))
    
    idle_ratio = -1
    if num_oracle_lanes > 0:
        idle_ratio = 1 - num_active_lanes / num_oracle_lanes
    return lif, idle_ratio


def aggregate_for_relative_stat(df, base_system, systems, indices, queries, agg_type):

    df_relative_data = {
        'Pyper': [0] * len(indices),
        'Pyper-w/o-ibr': [0] * len(indices),
        'DogQC++': [0] * len(indices),
        'Themis-w/o-aws': [0] * len(indices),
        'Themis': [0] * len(indices),
    }
    
    #systems = list(df_relative_data.keys())                    
    for i, index in enumerate(indices):
        
        if base_system == None:
            bases = [1] * len(queries)
        else:
            bases = [0] * len(queries)
            for sys_id, system in enumerate(systems):
                system_label = system_labels[sys_id]
                if system_label != base_system: continue
                for q_id, q in enumerate(queries):
                    bases[q_id] = df[system_label][i][q_id]
                    if bases[q_id] == 0:
                        print("Error", system_label, q)

        for sys_id, system in enumerate(systems):
            system_label = system_labels[sys_id]
            if system_label not in df: continue
            if base_system != None and system_label == base_system: 
                df_relative_data[system_label][i] = 1
                continue
            execution_time = 0
            num_valid = 0
            for q_id, q in enumerate(queries):
                if bases[q_id] == 0: continue
                if df[system_label][i][q_id] == 0:
                    #print("Error", system_label, i, index, q)
                    continue
                num_valid += 1
                val = df[system_label][i][q_id] / bases[q_id]
                if agg_type == 'max':
                    execution_time = max(execution_time, val)
                elif agg_type == 'mean':
                    execution_time += val
            if num_valid == len(queries):
                if agg_type == 'mean':
                    df_relative_data[system_label][i] = execution_time / num_valid
                    #if system_label == 'Themis': print(system_label, execution_time, num_valid,  df[system_label][i], bases)
                else:
                    df_relative_data[system_label][i] = execution_time
    return df_relative_data


def draw_chart(name, df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance, systems, indices, labels, queries):

    df_data = aggregate_for_relative_stat(df_elapsed_time, None, systems, indices, queries, 'mean')
    df = pd.DataFrame(df_data, index=list(map(lambda x: str(x), labels)))
    print("===================================")
    print(f"{name}, query execution time")
    print(df)
    print("===================================")
    width = 0.35  # the width of the bars
    df.index.name = 'label'
    gnuplot.plot_data(df,
        'using 2:xticlabels(1) lc "web-blue"',
        'using 3:xticlabels(1) lc "skyblue"',
        'using 4:xticlabels(1) lc "orange"',
        'using 5:xticlabels(1) lc "pink"',
        'using 6:xticlabels(1) lc "red"',
        offset = 'graph -0.05, 0, 0, 0',
        yrange = '[0.8:200000]',
        key = 'noautotitle',
        #key = 'above samplen 1 width 5',
        logscale = 'y 10',
        style = ['data histogram',
                'histogram cluster gap 2',
                'fill solid border -1',
                'textbox transparent'],
        format="y '10^{%L}'" ,
        boxwidth = '0.9',
        output = f'"./plots/{exp_id}/plot-{name}-elapsed-time.png"',
        terminal = 'pngcairo size 600, 280',)
                    
    df_data = aggregate_for_relative_stat(df_elapsed_time, 'Themis', systems, indices, queries, 'mean')
    df = pd.DataFrame(df_data, index=list(map(lambda x: str(x), labels)))
    print("===================================")
    print(f"{name}, relative query execution time / Themis")
    print(df)
    print("===================================")
    width = 0.35  # the width of the bars
    df.index.name = 'label'
    gnuplot.plot_data(df,
        'using 2:xticlabels(1) lc "web-blue"',
        'using 3:xticlabels(1) lc "skyblue"',
        'using 4:xticlabels(1) lc "orange"',
        'using 5:xticlabels(1) lc "pink"',
        'using 6:xticlabels(1) lc "red"',
        offset = 'graph -0.1, 0, 0, 0',
        yrange = '[0.9:20000]',
        key = 'noautotitle',
        logscale = 'y 10',
        style = ['data histogram',
                'histogram cluster gap 1',
                'fill solid border -1',
                'textbox transparent'],
        format="y '10^{%L}'" ,
        boxwidth = '0.9',
        output = f'"./plots/{exp_id}/plot-{name}-relative-elapsed-time.png"',
        terminal = 'pngcairo size 600, 180',)
    
    df_data = aggregate_for_relative_stat(df_elapsed_time, 'Themis-w/o-aws', systems, indices, queries, 'mean')
    df = pd.DataFrame(df_data, index=list(map(lambda x: str(x), labels)))
    print("===================================")
    print(f"{name}, relative query execution time / Themis-w/o-aws")
    print(df)
    print("===================================")
    
    print("===================================")
    print(f"{name}, intra warp idle ratio")
    df_data = aggregate_for_relative_stat(df_intra_warp_idle_ratio, None, systems, indices, queries, 'mean')
    df = pd.DataFrame(df_data, index=list(map(lambda x: str(x), labels)))
    print(df)
    print("-----------------------------------")
    print("mean")
    df_relative_mean_data = aggregate_for_relative_stat(df_intra_warp_idle_ratio, 'Themis-w/o-aws', systems, indices, queries, 'mean')
    df_relative_mean = pd.DataFrame(df_relative_mean_data, index=list(map(lambda x: str(x), labels)))
    print(df_relative_mean)
    print("-----------------------------------")
    print("max")
    df_data = aggregate_for_relative_stat(df_intra_warp_idle_ratio, 'Themis-w/o-aws', systems, indices, queries, 'max')
    print(pd.DataFrame(df_data, index=list(map(lambda x: str(x), labels))))
    print("===================================")
    width = 0.35  # the width of the bars
    df.index.name = 'label'
    gnuplot.plot_data(df,
        'using 2:xticlabels(1) title columnheader(2) lc "web-blue"',
        'using 3:xticlabels(1) title columnheader(3) lc "skyblue"',
        'using 4:xticlabels(1) title columnheader(4) lc "orange"',
        'using 5:xticlabels(1) title columnheader(5) lc "pink"',
        'using 6:xticlabels(1) title columnheader(6) lc "red"',
        yrange = '[0:1]',
        key = 'above samplen 1 width 5',
        style = ['data histogram',
                'histogram cluster gap 1',
                'fill solid border -1',
                'textbox transparent'],
        #logscale = 'y 10',
        #format="y '10^{%L}'" ,
        boxwidth = '0.9',
        output = f'"./plots/{exp_id}/plot-{name}-iir.png"',
        terminal = 'pngcairo size 600, 230',)
    
    
    print("===================================")
    print(f"{name}, ILIF")
    #print(df_inter_warp_load_imbalance['Themis'])
    df_data = aggregate_for_relative_stat(df_inter_warp_load_imbalance, None, systems, indices, queries, 'mean')
    df = pd.DataFrame(df_data, index=list(map(lambda x: str(x), labels)))
    print(df)
    print("-----------------------------------")
    print("mean")
    df_relative_mean_data = aggregate_for_relative_stat(df_inter_warp_load_imbalance, 'Themis', systems, indices, queries, 'mean')
    df_relative_mean = pd.DataFrame(df_relative_mean_data, index=list(map(lambda x: str(x), labels)))
    print(df_relative_mean)
    print(pd.DataFrame(df_data,index=list(map(lambda x: str(x), labels))))
    print("-----------------------------------")
    print("max")
    df_data = aggregate_for_relative_stat(df_inter_warp_load_imbalance, 'Themis', systems, indices, queries, 'max')
    print(pd.DataFrame(df_data, index=list(map(lambda x: str(x), labels))))
    print("===================================")
    print(df)
    width = 0.35  # the width of the bars
    df.index.name = 'label'
    gnuplot.plot_data(df,
        'using 2:xticlabels(1) lc "web-blue"',
        'using 3:xticlabels(1) lc "skyblue"',
        'using 4:xticlabels(1) lc "orange"',
        'using 5:xticlabels(1) lc "pink"',
        'using 6:xticlabels(1) lc "red"',
        offset = 'graph -0.1, 0, 0, 0',
        yrange = '[0.9:3000]',
        key = 'noautotitle',
        style = ['data histogram',
                'histogram cluster gap 1',
                'fill solid border -1',
                'textbox transparent'],
        logscale = 'y 10',
        format="y '10^{%L}'" ,
        boxwidth = '0.9',
        output = f'"./plots/{exp_id}/plot-{name}-ilif.png"',
        terminal = 'pngcairo size 600, 180',)



if mode == 'block':
    grid_size = 13120
    block_sizes = [1, 2, 4, 8, 16, 32] #1, 2, 4, 8, 16, 32]
    
    queries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    systems = ['dogqc', 'themis', 'lb_themis_glvl_twolvlbitmaps', 'pyper', 'pyper_b']
    system_labels = ['DogQC++', 'Themis-w/o-aws', 'Themis', 'Pyper',  "Pyper-w/o-ibr"]
    
    indices = block_sizes
    df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance = init_info(queries, indices)
    
    for b_idx, bsize in enumerate(block_sizes):
        gsize = int(grid_size / bsize)
        bsize = 32 * bsize
        for q_id, q in enumerate(queries[:22]):
            for sys_id, system in enumerate(systems):
                try:
                    #print('process', q, system)
                    log_path = f'{exp_dir}/{system}_{q}_timecheck_block_{bsize}_{gsize}.log'
                    elapsedTime = extract_kernel_time(log_path)
                    log_path = f'{exp_dir}/{system}_{q}_profile_block_{bsize}_{gsize}.log'
                    lif, idle_ratio = extract_lif_and_idle_ratio(log_path)
                    system_label = system_labels[sys_id]
                    df_elapsed_time[system_label][b_idx][q_id] = elapsedTime
                    df_inter_warp_load_imbalance[system_label][b_idx][q_id] = lif   
                    if system != 'lb_themis_glvl_twolvlbitmaps':
                        df_intra_warp_idle_ratio[system_label][b_idx][q_id] = idle_ratio
                except:
                    print(q, system, 'wrong')
                    import traceback
                    print(traceback.format_exc())
                    
    
    labels = indices
    draw_chart("block", df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance, systems, indices, labels, queries)


if mode == 'grid':
    
    grid_sizes = [ 1, 8, 64, 512, 4096] #, 64, 512, 4096] #[1, 8, 64, 512, 4096]
    block_size = 4
    
    queries = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]
    systems = ['dogqc', 'themis', 'lb_themis_glvl_twolvlbitmaps', 'pyper', 'pyper_b']
    system_labels = ['DogQC++', 'Themis-w/o-aws', 'Themis', 'Pyper',  "Pyper-w/o-ibr"]
    
    indices = grid_sizes
    df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance = init_info(queries, indices)
    
    
    for idx, gsize in enumerate(grid_sizes):
        gsize = gsize
        bsize = 32 * block_size
        print(gsize, bsize)
        for q_id, q in enumerate(queries[:22]):
            for sys_id, system in enumerate(systems):
                try:
                    log_path = f'{exp_dir}/{system}_{q}_timecheck_grid_{bsize}_{gsize}.log'
                    elapsedTime = extract_kernel_time(log_path)
                    
                    log_path = f'{exp_dir}/{system}_{q}_profile_grid_{bsize}_{gsize}.log'
                    lif, idle_ratio = extract_lif_and_idle_ratio(log_path)
                    
                    system_label = system_labels[sys_id]
                    df_elapsed_time[system_label][idx][q_id] = elapsedTime
                    df_inter_warp_load_imbalance[system_label][idx][q_id] = lif   
                    if system != 'lb_themis_glvl_twolvlbitmaps':
                        df_intra_warp_idle_ratio[system_label][idx][q_id] = idle_ratio
                    
                except:
                    print(q, system, 'wrong')
                    import traceback
                    print(traceback.format_exc())
                    #pass

    
    labels = list(map(lambda x: 4 * x, grid_sizes))
    draw_chart("grid", df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance, systems, indices, labels, queries)


if mode == 'sf':
    
    scale_factors = [1, 5, 10, 15, 20, 25, 30]
    block_size = 4
    
    queries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    systems = ['dogqc', 'themis', 'lb_themis_glvl_twolvlbitmaps', 'pyper', 'pyper_b']
    system_labels = ['DogQC++', 'Themis-w/o-aws', 'Themis', 'Pyper',  "Pyper-w/o-ibr"]
    
    
    indices = scale_factors
    df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance = init_info(queries, indices)
    
    for idx, sf in enumerate(scale_factors):
        for q_id, q in enumerate(queries[:22]):
            for sys_id, system in enumerate(systems):
                try:
                    log_path = f'{exp_dir}/{system}_{q}_timecheck_sf_{sf}.log'
                    elapsedTime = extract_kernel_time(log_path)
                    log_path = f'{exp_dir}/{system}_{q}_profile_sf_{sf}.log'
                    lif, idle_ratio = extract_lif_and_idle_ratio(log_path)
                    system_label = system_labels[sys_id]
                    df_elapsed_time[system_label][idx][q_id] = elapsedTime
                    df_inter_warp_load_imbalance[system_label][idx][q_id] = lif   
                    if system != 'lb_themis_glvl_twolvlbitmaps':
                        df_intra_warp_idle_ratio[system_label][idx][q_id] = idle_ratio
                except:
                    print(q, system, 'wrong')
                    import traceback
                    print(traceback.format_exc())
                    #pass
    
    labels = indices
    draw_chart("sf", df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance, systems, indices, labels, queries)
    
if mode == 'profile':
    #Q1, Q6, Q12, Q14-15, Q19, Q21
    #queries = [1, 6, 12, 14, 15, 19, 2, 7, 11, 17, 18, 20, 21, 5, 8, 9, 3, 4, 10, 16, 13, 22]
    queries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    systems = ['dogqc', 'themis', 'lb_themis_glvl_twolvlbitmaps', 'pyper', 'pyper_b']
    system_labels = ['DogQC++', 'Themis-w/o-aws', 'Themis', 'Pyper',  "Pyper-w/o-ibr"]
    
    df_intra_warp_idle_ratio = {
        'Pyper': [0] * len(queries),
        'Pyper-w/o-ibr': [0] * len(queries),
        'DogQC++': [0] * len(queries),
        'Themis-w/o-aws': [0] * len(queries),
    }
    
    df_inter_warp_load_imbalance = {
        'Pyper': [0] * len(queries),
        'Pyper-w/o-ibr': [0] * len(queries),
        'DogQC++': [0] * len(queries),
        'Themis-w/o-aws': [0] * len(queries),
        'Themis': [0] * len(queries),
    }
    
    df_elapsed_time = {
        'Pyper': [0] * len(queries),
        'Pyper-w/o-ibr': [0] * len(queries),
        'DogQC++': [0] * len(queries),
        'Themis-w/o-aws': [0] * len(queries),
        'Themis': [0] * len(queries),
    }
    
    for q_id, q in enumerate(queries[:22]):
        for sys_id, system in enumerate(systems):
            try:    
                print('process', q, system)
                log_path = f'{exp_dir}/{system}_{q}_{mode}.log'
                lines = open(log_path).readlines()
                num_oracle_lanes = 0
                num_active_lanes = 0
                elapsedTime = None
                lines = list(filter(lambda x: 'totalKernelTime' in x, lines))                    
                s = 0
                for line in lines:
                    l = line.strip()
                    l = l[l.find(':') + 1:]
                    l = l.replace("ms","").strip()
                    elapsedTime = float(l)
                    s += elapsedTime
                    
                elapsedTime = s / len(lines)
                df[system_labels[sys_id]][q_id] = elapsedTime
                
                log_path = f'{exp_dir}/{system}_{q}_{mode}.log'
                lines = open(log_path).readlines()
                num_oracle_lanes = 0
                num_active_lanes = 0
                imbalances = []
                sum_active_clocks = []
                max_active_clocks = []
                kernelTimes = []
                real_lane_activities = []
                oracle_lane_activities = []
                num_active_warps = []
                num_warps = []
                for line in lines:
                    l = line.strip()
                    if len(line) < 300 and '_tail' not in line[:200]: 
                        if 'ms' not in line: continue
                        if l[-2:] != 'ms': continue       
                        if 'krnl' in line or 'KernelTime' in line:
                            l = line.strip()
                            l = l[l.find(':') + 1:]
                            l = l.replace("ms","").strip()
                            elapsedTime = float(l)
                            kernelTimes.append(elapsedTime)
                    else:
                        if 'active_lanes_nums' in line[:200] and system != 'lb_themis_glvl_twolvlbitmaps':
                            if '_tail' not in line[:200]:
                                start = l.find('active_lanes_nums:') + len('active_lanes_nums:')
                                es = l[start:].strip().split(' ')
                                es = list(map(lambda x: int(x), es))
                                sum_lanes = sum(es)
                                if 'oracle' in l[:start]: 
                                    num_oracle_lanes += sum_lanes
                                    oracle_lane_activities.append(sum_lanes)                                    
                                else: 
                                    num_active_lanes += sum_lanes
                                    real_lane_activities.append(sum_lanes)
                            else:
                                start = l.find('active_lanes_nums:') + len('active_lanes_nums:')
                                es = l[start:].strip().split(' ')
                                es = list(map(lambda x: int(x), es))
                                sum_lanes = sum(es)
                                if 'oracle' in l[:start]: 
                                    num_oracle_lanes += sum_lanes
                                    oracle_lane_activities[-1] += sum_lanes
                                else: 
                                    num_active_lanes += sum_lanes
                                    real_lane_activities[-1] += sum_lanes
                                
                        if 'active_clocks' in line[:200]: # and system != 'themis':
                            if '_tail' not in line[:200]:
                                start = l.find('active_clocks:') + len('active_clocks:')
                                es = list(map(lambda x: int(x), l[start:].strip().split(' ')))
                                num_warps.append(len(es))
                                es = list(filter(lambda y : y > 0, es))
                                s = sum(es)                            
                                sum_active_clocks.append(s)
                                m = max(es)                                
                                max_warp_id = es.index(m)
                                print('max warp id', line[:30], max_warp_id)
                                max_active_clocks.append(m)
                                num_active_warps.append(len(es))
                            else:
                                start = l.find('active_clocks:') + len('active_clocks:')
                                es = list(map(lambda x: int(x), l[start:].strip().split(' ')))
                                num_warps[-1] += len(es)
                                es = list(filter(lambda y : y > 0, es))
                                s = sum(es)
                                sum_active_clocks[-1] += s
                                m = max(es)
                                max_active_clocks[-1] = max(max_active_clocks[-1], m)                            
                                num_active_warps[-1] += len(es)

                totalKernelTime = sum(kernelTimes[:len(num_active_warps)])
                
                totalSumClocks = sum(sum_active_clocks[:len(num_active_warps)])
                
                #print(q, system, num_active_warps, sum_active_clocks, max_active_clocks, kernelTimes, real_lane_activities, oracle_lane_activities)
                
                lif = sum(map(lambda x: sum_active_clocks[x] * (max_active_clocks[x] / sum_active_clocks[x] * num_active_warps[x]) / totalSumClocks, filter(lambda y: num_active_warps[y] > 0,  [i for i in range(len(max_active_clocks))])))                
                df_inter_warp_load_imbalance[system_labels[sys_id]][q_id] = lif   
                if system != 'lb_themis_glvl_twolvlbitmaps':
                    df_intra_warp_idle_ratio[system_labels[sys_id]][q_id]  = 1 - num_active_lanes / num_oracle_lanes
            except:
                print(q, system, 'wrong')
                import traceback
                print(traceback.format_exc())
                #pass
                
        
    df = pd.DataFrame(df_intra_warp_idle_ratio, index=queries)
    width = 0.35  # the width of the bars
    df.index.name = 'label'

    print("=======================================================")
    print('Intra warp idle ratio, all queries')
    print(df)
    print(df.max())
    print(df.mean())
    
    df_stat = df.loc[[1, 6, 12, 14, 15, 19]]
    print("=======================================================")
    print('Intra warp idle ratio, QG-1')
    print(df_stat.max())
    print(df_stat.mean())
    
    df_stat = df.loc[[2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 16, 17, 18, 20, 21, 22]]
    print("=======================================================")
    print('Intra warp idle ratio, QG-M')
    print(df_stat.mean())
        
    df_stat = df.loc[[5, 8, 9, 2, 7, 11, 17, 18, 20, 21]]
    print("=======================================================")
    print('Intra warp idle ratio, QG-M, tiny scanned table')
    print(df_stat.mean())
    
    df_stat = df.loc[[5, 8, 9, 3, 4, 10, 16]]
    print("=======================================================")
    print('Intra warp idle ratio, QG-M, small last join output')
    print(df_stat.mean())

    df_stat = df.loc[[2, 7, 11, 17, 18, 20, 21, 13, 22]]
    print("=======================================================")
    print('Intra warp idle ratio, QG-M, large last join output')
    print(df_stat.mean())

    gnuplot.plot_data(df,
            'using 2:xticlabels(1) lc "web-blue"',
            'using 3:xticlabels(1) lc "skyblue"',
            'using 4:xticlabels(1) lc "orange"',
            'using 5:xticlabels(1) lc "pink"',
            'using 6:xticlabels(1) lc "red"',
            yrange = '[0:1]',
            key = 'noautotitle',
            style = ['data histogram',
                    'histogram cluster gap 1',
                    'fill solid border -1',
                    'textbox transparent'],
            boxwidth = '0.9',
            output = f'"./plots/{exp_id}/plot-lane-activity.png"',
            terminal = 'pngcairo size 1200, 150',)
    
    df = pd.DataFrame(df_inter_warp_load_imbalance, index=list(map(lambda x: str(x), queries)))
    
    width = 0.35  # the width of the bars
    df.index.name = 'label'
    
    print("=======================================================")
    print('ILIF, all queries')
    print("ILIF")
    print(df)
    print("-------------------------------------------------------")
    print(df.max())
    print("-------------------------------------------------------")
    print(df.mean())
    
    print("=======================================================")
    print('ILIF, queries with inter-warp load imbalances')
    df_stat = df.loc[[4, 16, 10, 13, 22, 8, 9, 2, 11, 21, 7, 17, 18, 20]]
    print(df_stat)
    print("-------------------------------------------------------")
    print(df_stat.max())
    print("-------------------------------------------------------")
    print(df_stat.mean())

    print("=======================================================")
    print('ILIF, queries without inter-warp load imbalances')
    df_stat = df.loc[[1, 6, 12, 14, 15, 19]]
    print(df_stat)
    print("-------------------------------------------------------")
    print(df_stat.max())
    print("-------------------------------------------------------")
    print(df_stat.mean())

        
    gnuplot.plot_data(df,
            'using 2:xticlabels(1) lc "web-blue"',
            'using 3:xticlabels(1) lc "skyblue"',
            'using 4:xticlabels(1) lc "orange"',
            'using 5:xticlabels(1) lc "pink"',
            'using 6:xticlabels(1) lc "red"',
            yrange = '[0.99:200000]',
            key = 'noautotitle',
            logscale = 'y 10',
            style = ['data histogram',
                    'histogram cluster gap 1',
                    'fill solid border -1',
                    'textbox transparent'],
            format="y '10^{%L}'" ,
            boxwidth = '0.9',
            output = f'"./plots/{exp_id}/plot-load-imbalance-factor.png"',
            terminal = 'pngcairo size 1200, 150',)
                    


if mode == 'timecheck':
    
    queries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    #queries = [1, 6, 12, 14, 15, 19, 2, 7, 11, 17, 18, 20, 21, 5, 8, 9, 3, 4, 10, 16, 13, 22]
    systems = ['dogqc', 'themis', 'lb_themis_glvl_twolvlbitmaps', 'pyper', 'pyper_b']
    system_labels = ['DogQC++', 'Themis-w/o-aws', 'Themis', 'Pyper', "Pyper-w/o-ibr"]
    
    
    df = {
        'Pyper': [0] * len(queries),
        'Pyper-w/o-ibr': [0] * len(queries),
        'DogQC++': [0] * len(queries),
        'Themis-w/o-aws': [0] * len(queries),
        'Themis': [0] * len(queries),
        
    }
    for q_id, q in enumerate(queries[:22]):
        for sys_id, system in enumerate(systems):
            try:
                log_path = f'{exp_dir}/{system}_{q}_{mode}.log'
                lines = open(log_path).readlines()
                num_oracle_lanes = 0
                num_active_lanes = 0
                elapsedTime = None
                lines = list(filter(lambda x: 'totalKernelTime' in x, lines))                    
                s = 0
                for line in lines:
                    l = line.strip()
                    l = l[l.find(':') + 1:]
                    l = l.replace("ms","").strip()
                    elapsedTime = float(l)
                    s += elapsedTime
                    
                elapsedTime = s / len(lines)
                df[system_labels[sys_id]][q_id] = elapsedTime
            except:
                pass
        
    df = pd.DataFrame(df, index=queries)

    def show_ratio(df_origin, dividers=['Themis-w/o-aws', 'Themis']):
        print('--------------------------')
        print(df_origin)
        for divider in dividers:
            df_stat = df_origin.copy()
            print('--------------------------')
            print(f'- system / {divider}')
            df_stat[f'/{divider}'] = df_stat[divider]
            for sys in system_labels:
                df_stat[sys] = df_stat[sys] / df_stat[f'/{divider}']
            print(df_stat.max())
            print(df_stat.mean())


    print('==========================')
    print('All queries')
    show_ratio(df, dividers=['Themis-w/o-aws', 'Themis', 'Pyper-w/o-ibr'])
    print('==========================')
    print('Queries without inter-warp load imbalances')        
    show_ratio(df.loc[[1, 6, 12, 14, 15, 19, 3]])
    print('==========================')
    print('QG-1')
    show_ratio(df.loc[[1, 6, 12, 14, 15, 19]])
    print('==========================')
    print('QG-M')
    show_ratio(df.loc[[2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 16, 17, 18, 20, 21, 22]])
    print('==========================')
    print('QG-M small last join outputs')
    show_ratio(df.loc[[5, 8, 9, 3, 4, 10, 16]])
    print('==========================')
    print('QG-M large last join outputs')
    show_ratio(df.loc[[2, 7, 11, 17, 18, 20, 21, 13, 22]])
    width = 0.35  # the width of the bars
    df.index.name = 'label'
    gnuplot.plot_data(df,
            #'using 2:xticlabels(1) title columnheader(2) lc "black"',
            'using 2:xticlabels(1) title columnheader(2) lc "web-blue"',
            'using 3:xticlabels(1) title columnheader(3) lc "skyblue"',
            'using 4:xticlabels(1) title columnheader(4) lc "orange"',
            'using 5:xticlabels(1) title columnheader(5) lc "pink"',
            'using 6:xticlabels(1) title columnheader(6) lc "red"',
            key = 'above samplen 2 width 5',
            #logscale = 'y',
            logscale = 'y 2',
            #ylabel = '"Elapsed\nTime (ms)"',
            style = ['data histogram',
                    'histogram cluster gap 1',
                    'fill solid border -1',
                    'textbox transparent'],
            #xtics= 'offset 0,-1,0',
            format="y '2^{%L}'" ,
            boxwidth = '0.9',
            output = f'"./plots/{exp_id}/plot-elapsed-time.png"',
            terminal = 'pngcairo size 1300, 250',)
                        

def parse_clocks(sys_label, log_path, data):
    #print(log_path)
    if os.path.isfile(log_path) == False:
        return  
    #print(sys_label, log_path)
    try:
    #if True:
        lines = open(log_path).readlines()
        processing_clocks = 0
        pushing_clocks = 0
        waiting_clocks = 0
        kernelTime = 0
        num_idle = 0       
        num_pushed = 0
        for line in lines:
            l = line.strip()
            if len(line) < 300 and '_tail' not in line[:200]: 
                    if 'ms' not in line: continue
                    if l[-2:] != 'ms': continue       
                    if 'total' in line: continue
                    if 'krnl' in line or 'KernelTime' in line:
                        l = line.strip()
                        l = l[l.find(':') + 1:]
                        l = l.replace("ms","").strip()
                        elapsedTime = float(l)
                        kernelTime += elapsedTime
            elif 'krnl_' in line[:100] and 'clocks:' in line[:200]:
                start = l.find('clocks:') + len('clocks:')
                es = list(filter(lambda y : y > 0, map(lambda x: int(x), l[start:].strip().split(' '))))
                s = sum(es)
                if 'processing_clocks' in line[:200]:
                    processing_clocks += s
                elif 'pushing_clocks' in line[:200]:
                    pushing_clocks += s
                elif 'waiting_clocks' in line[:200]:
                    waiting_clocks += s
            elif 'krnl_' in line[:100] and 'num_idle:' in line[:200]:
                start = l.find('num_idle:') + len('num_idle:')
                es = list(filter(lambda y : y > 0, map(lambda x: int(x), l[start:].strip().split(' '))))
                s = sum(es)
                num_idle += s
            elif 'krnl_' in line[:100] and 'num_pushed:' in line[:200]:
                start = l.find('num_pushed:') + len('num_pushed:')
                es = list(filter(lambda y : y > 0, map(lambda x: int(x), l[start:].strip().split(' '))))
                s = sum(es)
                num_pushed += s
                
        s = processing_clocks + pushing_clocks + waiting_clocks
        #print(sys_labels[sys_id], processing_clocks, pushing_clocks, waiting_clocks)
        data.append([sys_label, processing_clocks / s * kernelTime, pushing_clocks / s * kernelTime, waiting_clocks / s * kernelTime, num_idle, num_pushed])
        if sys_label == 'WS':
            print(data[-1])
    except:
        data.append([sys_label, 0, 0, 0, 0, 0])
    #    pass

def convert_num_to_str(num):
    num = int(num)
    if num >= 1000000:
        num = str(int(num/1000000)) + 'M'
    elif num >= 1000:
        num = str(int(num/1000)) + 'K'
    
    return str(num)    
            

if mode == 'cnts':
    
    queries = [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 16, 17, 18, 20, 21, 22]
    
        
    for q in queries:
        
        log_path = f'./exp_result/{exp_id}/themis_{q}_cnts1.log'
        
        lines = open(log_path).readlines()
        lines = list(filter(lambda x : 'attr' == x[:4], lines))
        
        data = []
        for line in lines:            
            es = line.split('attr')
            frequency = es[1].split(':')[1].strip()
            num = es[2].split(':')[1].strip()
            data.append([int(frequency), int(num)])
            
            
        #print(q, data)
        min_freq1 = min(map(lambda x: x[0], data))
        max_freq1 = max(map(lambda x: x[0], data))
        
        total_sum1 = sum(map(lambda x: x[0] * x[1], data))
        total_num1 = sum(map(lambda x : x[1], data))
        avg1 = total_sum1 / total_num1

        log_path = f'./exp_result/{exp_id}/themis_{q}_cnts2.log'
        
        lines = open(log_path).readlines()
        lines = list(filter(lambda x : 'attr' == x[:4], lines))
        
        data = []
        for line in lines:            
            es = line.split('attr')
            frequency = es[1].split(':')[1].strip()
            num = es[2].split(':')[1].strip()
            data.append([int(frequency), int(num)])
            
            
        #print(q, data)
        min_freq2 = min(map(lambda x: x[0], data))
        max_freq2 = max(map(lambda x: x[0], data))
        
        total_sum2 = sum(map(lambda x: x[0] * x[1], data))
        total_num2 = sum(map(lambda x : x[1], data))
        avg2 = total_sum2 / total_num2


        print(f'Q{q} & {convert_num_to_str(total_num2)} & {convert_num_to_str(avg1)} & {convert_num_to_str(min_freq1)} & {convert_num_to_str(max_freq1)} & {convert_num_to_str(avg2)} & {convert_num_to_str(min_freq2)} & {convert_num_to_str(max_freq2)} \\\\ \hline')        
    


if mode == 'stats':

    systems = ['themis', 'lb_themis_ws_1024', 'lb_themis_ws_2048', 'lb_themis_ws_4096', 'lb_themis_ws_8192', 'lb_themis_ws_16384', 'lb_themis_glvl_twolvlbitmaps'] #, 'lb_themis_llvl_twolvlbitmaps'] #, 'lb_themis_clvl_twolvlbitmaps']
    sys_labels = ['no-LB', 'fws (1K)', 'fws (2K)', 'fws (4K)', 'fws (8K)', 'fws (16K)', 'aws'] #WP-heaviest', 'WP-any'] #, 'WP-clvl']
    queries = [ 4, 5, 8, 9, 10, 11, 17, 20, 22]
    for q in queries:
        data = []
        for sys_id, sys in enumerate(systems):
            if '_ws_' in sys or sys == 'themis':
                log_path = f'{exp_dir}/{sys}_{q}_stats.log'
                sys_label = sys_labels[sys_id]
                parse_clocks(sys_label, log_path, data)
            else:
                for j in range(3):
                    log_path = f'{exp_dir}/{sys}_{q}_{j}_stats.log'
                    sys_label = sys_labels[sys_id] + ' ' + str(j)
                    parse_clocks(sys_label, log_path, data)

        pg = gnuplot.Gnuplot()
        
        max_y = 0
        with open('data.dat', 'w') as f:
            for row in data:
                total = sum(row[1:4])
                max_y = max(total, max_y)
                f.write(f"\"{row[0]}\" {row[1]} {row[2]} {row[3]} {row[4]}\n")

        max_y = max_y * 1.3
        # Start Gnuplot
        
        pg('set terminal pngcairo size 500,250')  # Set the output format and size
        pg(f'set output "./plots/{exp_id}/cost-breakdown-Q{q}.png"')  # Set the output file
        pg('set style data histograms')
        pg('set style histogram rowstacked')
        pg('set style fill solid 1.0 border -1')
        pg('set boxwidth 0.7')
        
        if q in [2, 8, 11, 20]: #5, 8, 10, 13, 17, 20]:
            pg('unset ylabel')
            #pg('set ylabel "Execution time (ms)" font "Arial,16"')
        else:
            pg('unset ylabel')
        #pg(f'set xlabel "Q{q}"')
        pg(f"set yrange [0:{max_y}]")
        pg(f'set xtics rotate by -45 font "Arial,14" offset 0,0')
        #pg('set key autotitle columnheader')
        pg('set nogrid')
        pg('unset key')
        #pg('set key horizontal outside top')

        # Plotting the data
        
        plot_command = 'plot "data.dat" using 2:xtic(1) title "Processing Time", "" using 3 title "Pushing Time", "" using 4 title "Waiting Time"'
        for i, row in enumerate(data):
            total = sum(row[1:4])
            num_idle = row[-2]
            label = ''
            if num_idle >= 1000000:
                num_idle = int(num_idle / 1000000) * 1000000
                label= str(int(float(num_idle) / 1000000))  + 'M'
                #print(label, num_idle)
            elif num_idle >= 1000:
                num_idle = int(num_idle / 1000) * 1000
                label= str(int(float(num_idle) / 1000))  + 'K'
                #print(label, num_idle)
            elif num_idle > 0:
                label = str(num_idle)
            plot_command += f', "" using ({i}):({total+(max_y/10)}):("{label}") with labels notitle rotate by 45 font "Arial,11" '

        #print(plot_command)
        pg(plot_command)

        # Reset output to display the plot on screen
        pg('set output')
        
if mode == 'lazy_materialization':
    systems = ['dogqc', 'themis', 'lb_themis_aws_twolvlbitmaps', 'pyper', 'pyper_b']
    system_labels = ['DogQC++', 'Themis-w/o-aws', 'Themis', 'Pyper',  "Pyper-w/o-ibr"]
    queries = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    
    indices = ['f','t']
    df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance = init_info(queries, indices)
    
    for idx, val in enumerate(indices):
        for q_id, q in enumerate(queries):
            for sys_id, system in enumerate(systems):
                try:
                    log_path = f'{exp_dir}/{system}_{q}_timecheck_{val}.log'
                    elapsedTime = extract_kernel_time(log_path)
                    #log_path = f'{exp_dir}/{system}_{q}_profile_grid_{bsize}_{gsize}.log'
                    #lif, idle_ratio = extract_lif_and_idle_ratio(log_path)                    
                    system_label = system_labels[sys_id]
                    df_elapsed_time[system_label][idx][q_id] = elapsedTime
                    # df_inter_warp_load_imbalance[system_label][idx][q_id] = lif   
                    # if system != 'lb_themis_aws_twolvlbitmaps':
                    #     df_intra_warp_idle_ratio[system_label][idx][q_id] = idle_ratio
                except:
                    print(q, system, 'wrong')
                    import traceback
                    print(traceback.format_exc())
                    #pass
    #labels = ['No lazy materiialization', 'Lazy materialization']
    #print(df_elapsed_time)
    speedups = {
        'DogQC++': [0] * len(queries),
        'Themis-w/o-aws': [0] * len(queries),
        'Themis': [0] * len(queries),
        'Pyper': [0] * len(queries),
        'Pyper-w/o-ibr': [0] * len(queries),
    }
    
    for q_id, q in enumerate(queries):
        for sys_id, system in enumerate(systems):
            system_label = system_labels[sys_id]
            print(system_label, q_id)
            speedup = df_elapsed_time[system_label][0][q_id] / df_elapsed_time[system_label][1][q_id]
            speedups[system_label][q_id] = speedup
    
    print('speedup')
    for system_label, speedup in speedups.items():
        print(f"\t{system_label}", sum(speedup) / len(speedup), list(map(lambda x: float(str(x)[:5]), speedup)))
        
    speedups = {
        'DogQC++': [0] * len(queries),
        'Themis-w/o-aws': [0] * len(queries),
        'Themis': [0] * len(queries),
        'Pyper': [0] * len(queries),
        'Pyper-w/o-ibr': [0] * len(queries),
    }
    print('speedup / Themis-w/o-aws')
    for q_id, q in enumerate(queries):
        for sys_id, system in enumerate(systems):
            system_label = system_labels[sys_id]
            speedup = df_elapsed_time[system_label][1][q_id] / df_elapsed_time['Themis-w/o-aws'][1][q_id]
            speedups[system_label][q_id] = speedup
    for system_label, speedup in speedups.items():
        print(f"\t{system_label}", sum(speedup) / len(speedup), list(map(lambda x: float(str(x)[:5]), speedup)))
    
    speedups = {
        'DogQC++': [0] * len(queries),
        'Themis-w/o-aws': [0] * len(queries),
        'Themis': [0] * len(queries),
        'Pyper': [0] * len(queries),
        'Pyper-w/o-ibr': [0] * len(queries),
    }
    print('speedup / Themis')
    for q_id, q in enumerate(queries):
        for sys_id, system in enumerate(systems):
            system_label = system_labels[sys_id]
            speedup = df_elapsed_time[system_label][1][q_id] / df_elapsed_time['Themis'][1][q_id]
            speedups[system_label][q_id] = speedup
    for system_label, speedup in speedups.items():
        print(f"\t{system_label}", sum(speedup) / len(speedup), list(map(lambda x: float(str(x)[:5]), speedup)))
    
    draw_chart("grid", df_elapsed_time, df_intra_warp_idle_ratio, df_inter_warp_load_imbalance, systems, indices, labels, queries)


if mode == 'themis_twolvlbitmaps':
    queries = [ 4, 5, 8, 9, 10, 11, 17, 20, 22 ]  # [ 4, 5, 8, 9, 10, 11, 17, 20, 22 ] #, 17, 18, 19, 20, 22]
    
    systems = ['lb_themis_aws_twolvlbitmaps', 'lb_themis_aws_idqueue']
    system_labels = ['twolvlbitmaps', 'queue']
    
    df_elapsed_time = {
        'twolvlbitmaps': [0] * len(queries),
        'queue': [0] * len(queries),
    }
    
    for q_id, q in enumerate(queries):
        for sys_id, system in enumerate(systems):
            try:
                log_path = f'{exp_dir}/{system}_{q}_timecheck.log'
                elapsedTime = extract_kernel_time(log_path)
                system_label = system_labels[sys_id]
                df_elapsed_time[system_label][q_id] = elapsedTime
            except:
                print(q, system, 'wrong')
                import traceback
                print(traceback.format_exc())
    
    print('themis')
    print('\ttwolvlbitmaps,', df_elapsed_time['twolvlbitmaps'])
    print('\tqueue,', df_elapsed_time['queue'])
    speedups = [0] * len(queries)    
    for q_id, q in enumerate(queries):
        speedup = df_elapsed_time['queue'][q_id] / df_elapsed_time['twolvlbitmaps'][q_id]
        speedups[q_id] = speedup
    print('\tspeedup', sum(speedups) / len(speedups), speedups)
    
    df_aws_elapsed_time = df_elapsed_time
    

    queries = [ 4, 5, 8, 9, 10, 11, 17, 20, 22 ] # [ 4, 5, 8, 9, 10, 11, 17, 20, 22 ]
    queries = queries
    systems = ['twolvlbitmaps', 'queue']
    thresholds = [ 1024, 2048, 4096, 8192, 16394 ]
    systems = ['lb_themis_ws_twolvlbitmaps', 'lb_themis_ws_idqueue']
    system_labels = ['twolvlbitmaps', 'queue']
    
    df_elapsed_time = {
        'twolvlbitmaps': [[0] * len(thresholds) for i in range(len(queries))],
        'queue': [[0] * len(thresholds) for i in range(len(queries))]
    }
    for q_id, q in enumerate(queries):
        for sys_id, system in enumerate(systems):
            for thr_id, thr in enumerate(thresholds):
                try:
                    log_path = f'{exp_dir}/{system}_{thr}_{q}_timecheck.log'
                    elapsedTime = extract_kernel_time(log_path)
                    system_label = system_labels[sys_id]
                    df_elapsed_time[system_label][q_id][thr_id] = elapsedTime
                except:
                    print(q, system, 'wrong')
                    import traceback
                    print(traceback.format_exc())

    print('work sharing')
    print('\ttwolvlbitmaps,', df_elapsed_time['twolvlbitmaps'])
    print('\tqueue,', df_elapsed_time['queue'])

    df_min_elapsed_time = {
        'twolvlbitmaps': [0] * len(queries),
        'queue': [0] * len(queries)
    }
    
    for q_id, q in enumerate(queries):
        for sys_id, system in enumerate(systems):
            system_label = system_labels[sys_id]
            df_min_elapsed_time[system_label][q_id] = min(df_elapsed_time[system_label][q_id])
    print('\ttwolvlbitmaps,', df_min_elapsed_time['twolvlbitmaps'])
    print('\tqueue,', df_min_elapsed_time['queue'])
    
    print('speedup between ws twolvlbitmaps ws queue')
    speedups = [0] * len(queries)
    for q_id, q in enumerate(queries):
        speedup = df_min_elapsed_time['queue'][q_id] / df_min_elapsed_time['twolvlbitmaps'][q_id]
        speedups[q_id] = speedup
    print('\tspeedup', sum(speedups) / len(speedups), min(speedups), speedups)
    
    print('speedup between aws and ws twolvlbitmaps')
    speedups = [0] * len(queries)
    for q_id, q in enumerate(queries):
        speedup = df_min_elapsed_time['twolvlbitmaps'][q_id] / df_aws_elapsed_time['twolvlbitmaps'][q_id]
        speedups[q_id] = speedup
    print('\tspeedup', sum(speedups) / len(speedups), min(speedups), speedups)
    
    print('speedup between aws twolvlbitamps and ws queue')
    speedups = [0] * len(queries)
    for q_id, q in enumerate(queries):
        speedup = df_min_elapsed_time['queue'][q_id] / df_aws_elapsed_time['twolvlbitmaps'][q_id]
        speedups[q_id] = speedup
    print('\tspeedup', sum(speedups) / len(speedups), min(speedups), speedups)