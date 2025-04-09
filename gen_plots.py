import pandas as pd
import numpy as np
import seaborn as sns
import simulator
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser("AAAI-2023...")
parser.add_argument("--plot", type=float, default=1)
parser.add_argument("--name", type=str, default="")
parser.add_argument("--use_cache", type=int, default=1)
parser.add_argument("--user_num", type=int, default=100)
parser.add_argument("--n", type=int, default=5000)
parser.add_argument("--k", type=int, default=500)
parser.add_argument("--c_num", type=int, default=5)
parser.add_argument("--c_delta", type=float, default=0.1)
parser.add_argument("--min_csr", type=float, default=0.9)

args = parser.parse_args()
PLOT = args.plot
N = args.n
K = args.k
USER_NUM = args.user_num
C_NUM = args.c_num
C_DELTA = args.c_delta
MIN_CSR = args.min_csr
FILE = args.name if args.name != '' else f"figure{PLOT}".replace('.','_')
print("filename: {}".format(FILE))
USE_CACHE = args.use_cache


if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):
    # 初始化数据, 拆分用户候选集
    # df = pd.read_pickle('data_for_plots.pkl')
    # data = []
    # for user_id in df['user_id'].unique():
    #     data.append(df[df['user_id']==user_id][['eco_score','pt_norm', 'like_norm', 'follow_norm', 'comment_norm', 'forward_norm']].values.tolist())

    df = pd.read_pickle('data_for_plots_v2.pkl')
    data = []
    for user_id in df['user_id'].unique():
        data.append(df[df['user_id']==user_id][['freshness', 'pt_norm', 'like_norm', 'follow_norm', 'comment_norm', 'forward_norm']].values.tolist())

"""
Table 1: Performance comparison
"""

if args.plot == 0:
    if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):
        print(N, K, C_DELTA, C_NUM)
        sim = simulator.Simulator(data, n=N, k=K, consume_lambdas=[0.2, 0.2, 0.2, 0.2, 0.2])
        results = {
            'Objective Value': [],
            'Constraints Satisfaction Rate': [],
            'Latency': [],
            'Method': [],
        }

        opt, success_rate, latency = sim.run_strategy(USER_NUM, C_DELTA, C_NUM, strategy="hyper_opt")
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results["Latency"].append(latency)
        results['Method'].append("HyperOpt")

        opt, success_rate, latency = sim.run_strategy(USER_NUM, C_DELTA, C_NUM, strategy="ip")
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results["Latency"].append(latency)
        results['Method'].append("IP(CBC)")

        opt, success_rate, latency = sim.run_strategy(USER_NUM, C_DELTA, C_NUM, strategy="sub_grad")
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results["Latency"].append(latency)
        results['Method'].append("SubGrad")

        opt, success_rate, latency = sim.run_strategy(USER_NUM, C_DELTA, C_NUM, strategy="bisect_cpp")
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results["Latency"].append(latency)
        results['Method'].append("IR")

        opt, success_rate, latency = sim.find_boost(USER_NUM, C_DELTA, C_NUM, strategy="add_boost", min_success_rate=0.9)
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results["Latency"].append(latency)
        results['Method'].append("WS(0.9)")

        opt, success_rate, latency = sim.find_boost(USER_NUM, C_DELTA, C_NUM, strategy="add_boost", min_success_rate=0.8)
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results["Latency"].append(latency)
        results['Method'].append("WS(0.8)")

        opt, success_rate, latency = sim.find_boost(USER_NUM, C_DELTA, C_NUM, strategy="mul_boost", min_success_rate=0.9)
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results["Latency"].append(latency)
        results['Method'].append("WM(0.9)")

        opt, success_rate, latency = sim.find_boost(USER_NUM, C_DELTA, C_NUM, strategy="mul_boost", min_success_rate=0.8)
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results["Latency"].append(latency)
        results['Method'].append("WM(0.8)")

        tmp = pd.DataFrame(results)
        tmp.to_pickle(f"data/{FILE}_data.pkl")
    else:
        tmp = pd.read_pickle(f"data/{FILE}_data.pkl")
    print(tmp)

"""
Figure1: 固定delta和约束量, 不同约束满足率下各个方法的opt对比
"""
if args.plot == 1:
    if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):
        sim = simulator.Simulator(data, n=N, k=K, consume_lambdas=[0.2, 0.2, 0.2, 0.2, 0.2])

        c_delta = C_DELTA
        c_num = C_NUM

        # ip
        results = {
            'Objective Value': [],
            'Constraints Satisfaction Rate': [],
            'Method': []
        }
        opt, success_rate, _ = sim.run_strategy(USER_NUM, c_delta, c_num, strategy="ip")
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results['Method'].append("IP(CBC)")

        opt, success_rate, _ = sim.run_strategy(USER_NUM, c_delta, c_num, strategy="bisect_cpp")
        results['Objective Value'].append(opt)
        results['Constraints Satisfaction Rate'].append(success_rate)
        results['Method'].append("IR")

        opt = 0
        max_opt = max(results['Objective Value'])
        boost = 0
        while opt < max_opt:
            opt, success_rate, _ = sim.run_strategy(USER_NUM, c_delta, c_num, strategy="add_boost", boost=boost)
            results['Objective Value'].append(opt)
            results['Constraints Satisfaction Rate'].append(success_rate)
            results['Method'].append("WS")
            print(f"mul_boost: {boost}, {opt}")
            boost += 0.1

        opt = 0
        max_opt = max(results['Objective Value'])
        boost = 0
        while opt < max_opt:
            opt, success_rate, _ = sim.run_strategy(USER_NUM, c_delta, c_num, strategy="mul_boost", boost=boost)
            results['Objective Value'].append(opt)
            results['Constraints Satisfaction Rate'].append(success_rate)
            results['Method'].append("WM")
            print(f"mul_boost: {boost}, {opt}")
            boost += 0.1

        tmp = pd.DataFrame(results)
        tmp.to_pickle(f"data/{FILE}_data.pkl")
    else:
        tmp = pd.read_pickle(f"data/{FILE}_data.pkl")

    fig, ax = plt.subplots()
    tmp[tmp['Method']=='IP(CBC)'].plot.scatter('Constraints Satisfaction Rate', 'Objective Value', ax=ax, label='IP(CBC)', marker='s', s=50)
    tmp[tmp['Method']=='IR'].plot.scatter('Constraints Satisfaction Rate', 'Objective Value', ax=ax, label='IR', marker='o', color='red', s=50)
    tmp[tmp['Method']=='WS'][tmp['Constraints Satisfaction Rate']<1].plot('Constraints Satisfaction Rate', 'Objective Value', ax=ax, label='WS', color='blue')
    tmp[tmp['Method']=='WM'][tmp['Constraints Satisfaction Rate']<1].plot('Constraints Satisfaction Rate', 'Objective Value', ax=ax, label='WM', color='orange')

    plt.savefig(f'figures/{FILE}.png', dpi=300)

"""
Figure2: 不同约束量下各个方法的opt函数（对于非优化算法限定90%约束满足率）
"""
if args.plot == 2:
    if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):
        sim = simulator.Simulator(data, n=N, k=K, consume_lambdas=[0.2, 0.2, 0.2, 0.2, 0.2])
        # ip
        results = {
            'Objective Value': [],
            'Constraints Satisfaction Rate': [],
            'Number of Constraints': [],
            'Method': []
        }
        for c_num in range(1, 6):
            opt, success_rate, _ = sim.run_strategy(USER_NUM, C_DELTA, c_num, strategy="ip")
            results['Objective Value'].append(opt)
            results['Constraints Satisfaction Rate'].append(success_rate)
            results['Number of Constraints'].append(c_num)
            results['Method'].append("IP(CBC)")

            opt, success_rate, _ = sim.run_strategy(USER_NUM, C_DELTA, c_num, strategy="bisect_cpp")
            results['Objective Value'].append(opt)
            results['Constraints Satisfaction Rate'].append(success_rate)
            results['Number of Constraints'].append(c_num)
            results['Method'].append("IBR")

            opt, success_rate, _ = sim.run_strategy(USER_NUM, C_DELTA, c_num, strategy="sub_grad")
            results['Objective Value'].append(opt)
            results['Constraints Satisfaction Rate'].append(success_rate)
            results['Number of Constraints'].append(c_num)
            results['Method'].append("SubGrad")

            opt, success_rate, _ = sim.find_boost(USER_NUM, C_DELTA, c_num, strategy="add_boost", min_success_rate=0.99)
            results['Objective Value'].append(opt)
            results['Constraints Satisfaction Rate'].append(success_rate)
            results['Number of Constraints'].append(c_num)
            results['Method'].append("WS")

            opt, success_rate, _ = sim.find_boost(USER_NUM, C_DELTA, c_num, strategy="mul_boost", min_success_rate=0.99)
            results['Objective Value'].append(opt)
            results['Constraints Satisfaction Rate'].append(success_rate)
            results['Number of Constraints'].append(c_num)
            results['Method'].append("WM")

        tmp = pd.DataFrame(results)
        tmp.to_pickle(f"data/{FILE}_data.pkl")
    else:
        tmp = pd.read_pickle(f"data/{FILE}_data.pkl")

    fig, ax = plt.subplots()
    sns.pointplot(tmp, x='Number of Constraints', y='Objective Value', hue='Method', dodge=True, markers=['o', '*', 's', 'x'])
    plt.savefig(f'figures/{FILE}.png', dpi=300)

"""
Figure3: 不同delta下各个方法的opt函数（对于非优化算法限定90%约束满足率）
"""
if args.plot == 3:
    if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):

        sim = simulator.Simulator(data, n=N, k=K, consume_lambdas=[0.2, 0.2, 0.2, 0.2, 0.2])

        # ip
        results = {
            'Objective Value': [],
            'Tolerance of Constraints': [],
            'Method': []
        }
        for c_delta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            print(f"c_delta = {c_delta}")
            print("running IP...")
            opt, success_rate, _ = sim.run_strategy(USER_NUM, c_delta, C_NUM, strategy="ip")
            results['Objective Value'].append(opt)
            results['Tolerance of Constraints'].append(c_delta)
            results['Method'].append("IP(CBC)")

            print("running IR...")
            opt, success_rate, _ = sim.run_strategy(USER_NUM, c_delta, C_NUM, strategy="bisect_cpp")
            results['Objective Value'].append(opt)
            results['Tolerance of Constraints'].append(c_delta)
            results['Method'].append("IBR")

            print("running SubGrad...")
            opt, success_rate, _ = sim.run_strategy(USER_NUM, c_delta, C_NUM, strategy="sub_grad")
            results['Objective Value'].append(opt)
            results['Tolerance of Constraints'].append(c_delta)
            results['Method'].append("SubGrad")

            print("running MS...")
            opt, success_rate, _ = sim.find_boost(USER_NUM, c_delta, C_NUM, strategy="add_boost")
            results['Objective Value'].append(opt)
            results['Tolerance of Constraints'].append(c_delta)
            results['Method'].append("MS(0.9)")

            print("running MW...")
            opt, success_rate, _ = sim.find_boost(USER_NUM, c_delta, C_NUM, strategy="mul_boost")
            results['Objective Value'].append(opt)
            results['Tolerance of Constraints'].append(c_delta)
            results['Method'].append("MW(0.9)")
        
        tmp = pd.DataFrame(results)
        tmp.to_pickle(f"data/{FILE}_data.pkl")
    else:
        tmp = pd.read_pickle(f"data/{FILE}_data.pkl")

    fig, ax = plt.subplots()
    ax = sns.pointplot(tmp, x='Tolerance of Constraints', y='Objective Value', hue='Method', dodge=True, markers=['o', '*', 's', 'x'])
    ax.set_xlabel("Tolerance of Constraints ($\\delta_k$)")
    plt.savefig('figures/figure3.png', dpi=300)

"""
Figure4: IR算法在不同约束量下不同迭代次数的opt值
"""
if args.plot == 4:
    if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):
        sim = simulator.Simulator(data, n=N, k=K, consume_lambdas=[0.2, 0.2, 0.2, 0.2, 0.2])

        c_delta = C_DELTA
        user_num = USER_NUM

        # ip
        results = {
            'opt_ratio': [],
            'iteration': [],
            'c_num': []
        }

        for c_num in [1, 2, 3, 4, 5]:
            user_cnt = len(sim.data)
            success_cnt = 0
            opt_sum = []
            for user_id in range(min(user_cnt, user_num)):
                user_data = sim._get_user_data(user_id)
                constraints = sim._get_user_constraints(user_id, c_delta, c_num)
                ip_topk = sim._ip(user_data, constraints)
                ip_opt, success = sim._evaluate_topk(ip_topk, constraints)
                _, opt_history = sim._bisect(user_data, constraints, max_iteration=20, return_log=True)
                if not opt_sum:
                    opt_sum = [0 for _ in range(len(opt_history))]
                opt_sum = [opt_history[i]/ip_opt + opt_sum[i] for i in range(len(opt_history))]
            
            opt_ratio = [x/min(user_cnt, user_num) for x in opt_sum]
            iter_list = [i+1 for i in range(len(opt_ratio))]
            c_num_list = [c_num for _ in range(len(opt_ratio))]
            results['opt_ratio'] = results['opt_ratio'] + opt_ratio
            results['iteration'] = results['iteration'] + iter_list
            results['c_num'] = results['c_num'] + c_num_list

        tmp = pd.DataFrame(results)
        tmp.to_pickle(f"data/{FILE}_data.pkl")
    else:
        tmp = pd.read_pickle(f"data/{FILE}_data.pkl")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))  # 1行3列

    # tmp = tmp[tmp['iteration'] <= 10]
    ax2 = sns.pointplot(tmp[tmp['opt_ratio']>0.96], x='iteration', y='opt_ratio', hue='c_num', palette=['red', 'orange', 'blue', 'green', 'purple'], dodge=True, markers=['o', '*', 's', 'x', '^'], ax=ax2)
    ax2.set_xlabel("Number of Iterations", fontsize=16)
    ax2.set_ylabel("Percentage of Optimal Objective Value", fontsize=16)
    ax2.legend(title="K", title_fontsize=16, fontsize=16)
    # ax2.grid(True)

    sns.pointplot(tmp[tmp['iteration'] <= 10], x='iteration', y='opt_ratio', hue='c_num', palette=['red', 'orange', 'blue', 'green', 'purple'], dodge=True, markers=['o', '*', 's', 'x', '^'], ax=ax1)
    ax1.set_xlabel("Number of Iterations", fontsize=16)
    ax1.set_ylabel("Percentage of Optimal Objective Value", fontsize=16)
    ax1.legend(title="K", title_fontsize=16, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figures/{FILE}.png')


"""
Figure4.1: IR算法在不同约束量(K)下不同loop次数的opt值
"""
if args.plot == 4.1:
    if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):
        sim = simulator.Simulator(data, n=N, k=K, consume_lambdas=[0.2, 0.2, 0.2, 0.2, 0.2])

        c_delta = C_DELTA
        user_num = USER_NUM

        # ip
        results = {
            'opt_ratio': [],
            'loop': [],
            'c_num': []
        }

        for c_num in [1, 2, 3, 4, 5]:
            print(f"solving K={c_num}...")
            user_cnt = len(sim.data)
            success_cnt = 0
            opt_sum = []
            for user_id in range(min(user_cnt, user_num)):
                user_data = sim._get_user_data(user_id)
                constraints = sim._get_user_constraints(user_id, c_delta, c_num)
                ip_topk = sim._ip(user_data, constraints)
                ip_opt, success = sim._evaluate_topk(ip_topk, constraints)
                _, opt_history = sim._bisect(user_data, constraints, max_loop=5, return_loop_log=True)
                if not opt_sum:
                    opt_sum = [0 for _ in range(len(opt_history))]
                opt_sum = [opt_history[i]/ip_opt + opt_sum[i] for i in range(len(opt_history))]
            
            opt_ratio = [x/min(user_cnt, user_num) for x in opt_sum]
            iter_list = [i+1 for i in range(len(opt_ratio))]
            c_num_list = [c_num for _ in range(len(opt_ratio))]
            results['opt_ratio'] = results['opt_ratio'] + opt_ratio
            results['loop'] = results['loop'] + iter_list
            results['c_num'] = results['c_num'] + c_num_list

        tmp = pd.DataFrame(results)
        tmp.to_pickle(f"data/{FILE}_data.pkl")
    else:
        tmp = pd.read_pickle(f"data/{FILE}_data.pkl")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))  # 1行3列

    # tmp = tmp[tmp['iteration'] <= 10]
    ax2 = sns.pointplot(tmp[tmp['opt_ratio']>0.96], x='loop', y='opt_ratio', hue='c_num', palette=['red', 'orange', 'blue', 'green', 'purple'], dodge=True, markers=['o', '*', 's', 'x', '^'], ax=ax2)
    ax2.set_xlabel("Number of Loops", fontsize=16)
    ax2.set_ylabel("Percentage of Optimal Objective Value", fontsize=16)
    ax2.legend(title="K", title_fontsize=16, fontsize=16)
    # ax2.grid(True)

    sns.pointplot(tmp[tmp['iteration'] <= 10], x='loop', y='opt_ratio', hue='c_num', palette=['red', 'orange', 'blue', 'green', 'purple'], dodge=True, markers=['o', '*', 's', 'x', '^'], ax=ax1)
    ax1.set_xlabel("Number of Loops", fontsize=16)
    ax1.set_ylabel("Percentage of Optimal Objective Value", fontsize=16)
    ax1.legend(title="K", title_fontsize=16, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figures/{FILE}.png')

"""
Figure5: IR算法在不同约束量下不同delta的opt值
"""
if args.plot == 5:
    if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):
        sim = simulator.Simulator(data, n=N, k=K, consume_lambdas=[0.2, 0.2, 0.2, 0.2, 0.2])

        c_num = C_NUM
        user_num = USER_NUM

        # ip
        results = {
            'opt_ratio': [],
            'iteration': [],
            'c_delta': []
        }

        for c_delta in [0.1, 0.15, 0.2, 0.25, 0.3]:
            user_cnt = len(sim.data)
            success_cnt = 0
            opt_sum = []
            for user_id in range(min(user_cnt, user_num)):
                user_data = sim._get_user_data(user_id)
                constraints = sim._get_user_constraints(user_id, c_delta, c_num)
                ip_topk = sim._ip(user_data, constraints)
                ip_opt, success = sim._evaluate_topk(ip_topk, constraints)
                _, opt_history = sim._bisect(user_data, constraints, return_log=True)
                if not opt_sum:
                    opt_sum = [0 for _ in range(len(opt_history))]
                opt_sum = [opt_history[i]/ip_opt + opt_sum[i] for i in range(len(opt_history))]
            
            opt_ratio = [x/min(user_cnt, user_num) for x in opt_sum]
            iter_list = [i+1 for i in range(len(opt_ratio))]
            c_delta_list = [c_delta for _ in range(len(opt_ratio))]
            results['opt_ratio'] = results['opt_ratio'] + opt_ratio
            results['iteration'] = results['iteration'] + iter_list
            results['c_delta'] = results['c_delta'] + c_delta_list

        tmp = pd.DataFrame(results)
        tmp.to_pickle(f"data/{FILE}_data.pkl")
    else:
        tmp = pd.read_pickle(f"data/{FILE}_data.pkl")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))  # 1行3列
    # tmp = tmp[tmp['iteration'] <= 10]
    ax2 = sns.pointplot(tmp[tmp['opt_ratio']>0.97], x='iteration', y='opt_ratio', hue='c_delta', palette=['red', 'orange', 'blue', 'green', 'purple'], dodge=True, markers=['o', '*', 's', 'x', '^'], ax=ax2)
    ax2.legend(title="$\\delta_k$", title_fontsize=16, fontsize=16)
    ax2.set_xlabel("Number of Iterations", fontsize=16)
    ax2.set_ylabel("Percentage of Optimal Objective Value", fontsize=16)
    # ax2.grid(True)
    sns.pointplot(tmp[tmp['iteration'] <= 10], x='iteration', y='opt_ratio', hue='c_delta', palette=['red', 'orange', 'blue', 'green', 'purple'], dodge=True, markers=['o', '*', 's', 'x', '^'], ax=ax1)
    ax1.legend(title="$\\delta_k$", title_fontsize=16, fontsize=16)
    ax1.set_xlabel("Number of Iterations", fontsize=16)
    ax1.set_ylabel("Percentage of Optimal Objective Value", fontsize=16)
    # ax1.grid(True)
    # 调整布局
    plt.tight_layout()
    # 显示图形
    plt.savefig('figures/figure5_1.png')

"""
Figure6: IR算法在下不同N下的耗时
"""
if args.plot == 6:
    if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):

        results = {
            'N': [],
            'time_cost': [],
            'method': []
        }

        for n in range(1, 10):

            n = n * 1000
            sim = simulator.Simulator(data, n=n, k=K, consume_lambdas=[0.2, 0.2, 0.2, 0.2, 0.2])
            _, _, time_cost = sim.run_strategy(USER_NUM, C_DELTA, C_NUM, "ip")
            results['N'].append(n)
            results['time_cost'].append(time_cost)
            results['method'].append('ip')

            _, _, time_cost = sim.run_strategy(USER_NUM, C_DELTA, C_NUM, "bisect_cpp")
            results['N'].append(n)
            results['time_cost'].append(time_cost)
            results['method'].append('bisect')
        
        tmp = pd.DataFrame(results)
        tmp.to_pickle(f"data/{FILE}_data.pkl")
    else:
        tmp = pd.read_pickle(f"data/{FILE}_data.pkl")
    
    fig, ax = plt.subplots()
    sns.pointplot(tmp, x='N', y='time_cost', hue='method', dodge=True, markers=['o', '*'])
    plt.savefig('figures/figure6.png', dpi=300)

"""
Figure7: Latency of IBR by number of constraints (K)
"""
if args.plot == 7:
    if not USE_CACHE or not os.path.exists(f"data/{FILE}_data.pkl"):
        sim = simulator.Simulator(data, n=N, k=K, consume_lambdas=[0.2, 0.2, 0.2, 0.2, 0.2])

        results = {
            'c_num': [],
            'time_cost': [],
            'method': []
        }

        for c_num in range(1, 6):

            _, _, time_cost = sim.run_strategy(USER_NUM, C_DELTA, c_num, "ip")
            results['c_num'].append(c_num)
            results['time_cost'].append(time_cost)
            results['method'].append('ip')

            _, _, time_cost = sim.run_strategy(USER_NUM, C_DELTA, c_num, "bisect_cpp")
            results['c_num'].append(c_num)
            results['time_cost'].append(time_cost)
            results['method'].append('bisect')
        
        tmp = pd.DataFrame(results)
        tmp.to_pickle(f"data/{FILE}_data.pkl")
    else:
        tmp = pd.read_pickle(f"data/{FILE}_data.pkl")

    fig, ax = plt.subplots()
    sns.pointplot(tmp, x='c_num', y='time_cost', hue='method', dodge=True, markers=['o', '*'])
    plt.savefig('figures/figure7.png', dpi=300)
