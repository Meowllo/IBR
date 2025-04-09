import heapq
import numpy as np
from pulp import *
from functools import lru_cache
from multi_bisect_cpp import multi_bisect_cpp
import time
from tqdm import tqdm
import pandas as pd
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval

class Simulator:
    def __init__(self, data, n, k, consume_lambdas):
        df = pd.read_pickle('data_for_plots_v2.pkl')
        data = {}
        for user_id in df['user_id'].unique():
            data[user_id] = df[df['user_id']==user_id][['freshness', 'pt_norm', 'like_norm', 'follow_norm', 'comment_norm', 'forward_norm']].values.tolist()

        self.data = data
        self.user_list = list(self.data.keys())
        self.N = n
        self.K = k
        self.consume_lambdas = consume_lambdas
        self.user_data = {}
        self.user_constraints = {}
        self.strategies = {
            "add_boost": self._add_boost,
            "mul_boost": self._mul_boost,
            "ip": self._ip,
            "sub_grad": self._sub_gradient,
            "bisect": self._bisect,
            "bisect_cpp": self._bisect_cpp,
            "hyper_opt": self.hyper_opt,
        }
        self._user_states = None
        self._tscac_agent = None

    def find_boost(self, user_num, constraint_delta, constraint_num, strategy, min_success_rate=0.9, lb=0, ub=100, epsilon=0.01, verbose=False):
        assert strategy in ['add_boost', 'mul_boost']
        l, r = lb, ub
        best_opt = 0
        best_success_rate = 0
        best_latency = 0
        while r >= l + epsilon:
            m = (r + l) / 2
            opt, success_rate, latency = self.run_strategy(user_num, constraint_delta, constraint_num, strategy, boost=m)
            if verbose:
                print(l, r, m, opt, success_rate)
            if success_rate > min_success_rate:
                if opt > best_opt:
                    best_opt = opt
                    best_success_rate = success_rate
                    best_latency = latency
                l = m
            else:
                r = m
            
        return best_opt, best_success_rate, best_latency
        
    # @lru_cache(maxsize=1280)
    def run_strategy(self, user_num, constraint_delta, constraint_num, strategy, **kwargs):
        # 执行策略，输出约束满足率和opt函数值
        assert strategy in self.strategies
        topk_func = self.strategies[strategy]
        user_cnt = len(self.data)
        success_cnt = 0
        time_cost = 0 
        opt_sum = 0
        # for user_id in tqdm(range(min(user_cnt, user_num))):
        for user_id in tqdm(self.user_list[:user_num]):
            data = self._get_user_data(user_id)
            constraints = self._get_user_constraints(user_id, constraint_delta, constraint_num)
            t0 = time.time()
            topk = topk_func(data, constraints, **kwargs)
            t1 = time.time()
            opt, success = self._evaluate_topk(topk, constraints)
            opt_sum += opt
            if success:
                success_cnt += 1
            time_cost += t1 - t0
        opt_avg = opt_sum / min(user_cnt, user_num)
        success_rate = success_cnt / min(user_cnt, user_num)
        avg_time_cost = time_cost / min(user_cnt, user_num)
        return opt_avg, success_rate, avg_time_cost

    def _evaluate_topk(self, topk, constraints):
        opt = sum([topk[i][0] for i in range(self.K)])
        success = True
        for ci in range(len(constraints)):
            if sum([topk[i][ci+1] for i in range(self.K)]) < constraints[ci]:
                success = False
                break
        return opt, success

    def _get_user_data(self, user_id):
        # get user candidate set
        if user_id not in self.user_data:
            self.user_data[user_id] = self.data[user_id][:self.N]
        return self.user_data[user_id]

    def _get_user_constraints(self, user_id, constraint_delta, constraint_num=5):
        # get user constraints
        cache_key = f"{user_id}-{constraint_delta}-{constraint_num}"
        if cache_key not in self.user_constraints:
            data = self._get_user_data(user_id)
            consume_topk = heapq.nlargest(self.K, data, key=lambda x: sum([x[i] * self.consume_lambdas[i] for i in range(len(self.consume_lambdas))]))
            constraints = [
                sum([x[i+1] for x in consume_topk]) * (1 - constraint_delta)
                for i in range(constraint_num)
            ]
            self.user_constraints[cache_key] = constraints
        return self.user_constraints[cache_key]

    def _add_boost(self, data, constraints, boost):
        topk = heapq.nlargest(self.K, data, key=lambda x: boost * x[0] + sum([x[i+1] * self.consume_lambdas[i] for i in range(len(self.consume_lambdas))]))
        return topk

    def _mul_boost(self, data, constraints, boost):
        topk = heapq.nlargest(self.K, data, key=lambda x: ((1 + x[0]) ** boost) * sum([x[i+1] * self.consume_lambdas[i] for i in range(len(self.consume_lambdas))]))
        return topk 

    def _ip(self, data, constraints, verbose=False):
        # Integer Programming
        prob = LpProblem("Test", LpMaximize)
        x = [LpVariable(f'x{i}', 0, 1, LpInteger) for i in range(self.N)]
        prob += lpSum([x[i] * data[i][0] for i in range(self.N)])
        prob += lpSum(x) == self.K
        for ci in range(len(constraints)):
            prob += lpSum([x[i] * data[i][ci+1] for i in range(self.N)]) >= constraints[ci]
        if not verbose:
            pulp.PULP_CBC_CMD(msg=0, timeLimit=10).solve(prob)
        else:
            pulp.PULP_CBC_CMD(timeLimit=10).solve(prob)
        vars = {var.name: var.varValue for var in prob.variables()}
        topk = [data[i] for i in range(self.N) if vars[f'x{i}'] == 1]
        return topk
    
    def _bisect(self, data, constraints, lb=0, ub=1, epsilon=0.001, max_loop=4, max_iteration=0, verbose=False, return_log=False, return_loop_log=False):
        k = self.K
        lambdas = [0 for _ in range(len(constraints))]
        iter = 0
        loop_times = 0
        best_opt = 0
        best_lambdas = lambdas.copy()
        best_topk = None
        best_loop = 0
        best_iter = 0
        opt_history = []
        loop_opt_history = []
        while loop_times < max_loop or iter < max_iteration:
            loop_times += 1
            for i in range(len(lambdas)):
                iter += 1
                l, r = lb, ub
                while r >= l + epsilon:
                    lambdas[i] = (r + l) / 2
                    lambdas_0 = max(0, 1 - sum(lambdas))
                    topk = heapq.nlargest(k, data, key=lambda x: x[0] * lambdas_0 + sum([x[i+1] * lambdas[i] for i in range(len(lambdas))]))
                    constraints_satisfied = [sum([topk[i][ci+1] for i in range(k)]) >= constraints[ci] for ci in range(len(lambdas))]
                    if verbose:
                        constraints_values = [sum([topk[i][ci+1] for i in range(k)]) for ci in range(len(constraints))]
                        print(iter, lambdas, constraints, constraints_values)
                    if all(constraints_satisfied): 
                        opt = sum([topk[i][0] for i in range(k)])
                        if opt > best_opt:
                            best_lambdas = lambdas.copy()
                            best_topk = topk.copy()
                            best_opt = opt
                            best_loop = loop_times
                            best_iter = iter
                            # print(opt, iter, loop_times)
                    if constraints_satisfied[i]:
                        r = lambdas[i]
                    else:
                        l = lambdas[i]
                opt_history.append(best_opt)
            loop_opt_history.append(best_opt)
        if verbose:
            print(best_loop, best_iter,best_opt, best_lambdas)
        if return_log:
            return best_topk, opt_history
        elif return_loop_log:
            return best_topk, loop_opt_history
        else:
            return best_topk

    def _bisect_cpp(self, data, constraints, lb=0, ub=1, epsilon=0.001, max_loop=2, verbose=False):
        return multi_bisect_cpp(data, self.N, self.K, constraints, lb, ub, epsilon, max_loop, verbose)

    def _sub_gradient(self, data, constraints, max_steps=10000, early_stop=100, step_size=2, verbose=False):
        lambdas = [1] + self.consume_lambdas
        lambdas = lambdas[:len(constraints)+1]
        best_opt = 0
        best_lambdas = lambdas.copy()
        best_step = 0
        best_topk = None
        steps = 0
        stop = 0
        while steps < max_steps and stop < early_stop:
            steps += 1
            topk = heapq.nlargest(self.K, data, key=lambda x: sum([x[i] * lambdas[i] for i in range(len(lambdas))]))
            constraints_satisfied = True
            opt_updated = False
            sub_grads = []
            sub_grads_sum_square = 0
            for j in range(1, len(lambdas)):
                sub_grad = sum([topk[i][j] for i in range(self.K)]) - constraints[j-1]
                if sub_grad < 0:
                    constraints_satisfied = False
                sub_grads.append(sub_grad)
                sub_grads_sum_square += sub_grad * sub_grad
            if verbose:
                print("lambdas: ", steps, lambdas, sub_grads)
            if constraints_satisfied:
                opt = sum([topk[i][0] for i in range(self.K)])
                if opt > best_opt:
                    best_lambdas = lambdas.copy()
                    best_topk = topk.copy()
                    best_opt = opt
                    best_step = steps
                    opt_updated = True
                    stop = 0
                    if verbose:
                        print("new opt found: ", best_step, best_opt, best_lambdas)
            if not opt_updated:
                stop += 1
            for j in range(1, len(lambdas)):
                lambdas[j] -= step_size * sub_grads[j-1] / sub_grads_sum_square
                lambdas[j] = max(0, lambdas[j])

        if verbose:
            print(best_step, best_opt, best_lambdas)
        if not best_topk:
            return heapq.nlargest(self.K, data, key=lambda x: sum([x[i] * lambdas[i] for i in range(1, len(lambdas))]))
        return best_topk

    def hyper_opt(self, data, constraints, max_steps=100, penalty_coeff=20):
        # 
        def obj(lambdas):
            topk = heapq.nlargest(self.K, data, key=lambda x: sum([x[i] * lambdas[f'x{i}'] for i in range(len(lambdas))]))
            obj_value = sum([topk[i][0] for i in range(self.K)]) + sum([
                min(0, sum([topk[i][ci+1] for i in range(self.K)]) - constraints[ci]) * penalty_coeff 
                for ci in range(len(lambdas))
            ])
            return -obj_value

        search_space = {
            f'x{i}': hp.uniform(f'x{i}', 0, 1)
            for i in range(len(constraints))
        }

        best = fmin(
            fn=obj,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_steps,
            show_progressbar=False
        )
        best_topk = heapq.nlargest(self.K, data, key=lambda x: sum([x[i] * best[f'x{i}'] for i in range(len(best))]))
        return best_topk
