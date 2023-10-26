import time
import joblib
import os
import os.path as osp
import tensorflow as tf
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding

import random
import math
import numpy as np
import sys

# from HPCSimPickJobs import *
from HPCSimPickJobsHeterog import *

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.rcdefaults()
tf.enable_eager_execution()

from collections import defaultdict

def load_policy(model_path, itr='last'):
    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(model_path) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(model_path, 'simple_save'+itr))

    # get the correct op for executing actions
    pi = model['pi']
    v = model['v']
    out = model['out']
    get_out = lambda x ,y  : sess.run(out, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES), model['mask']:y.reshape(-1, MAX_QUEUE_SIZE)})
    # make function for producing an action given a single state
    get_probs = lambda x ,y  : sess.run(pi, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * env.NUM_NODES * TOTAL_FEATURES), model['mask']:y.reshape(-1, MAX_QUEUE_SIZE*env.NUM_NODES)})
    get_v = lambda x : sess.run(v, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
    return get_probs, get_out


#@profile
def run_policy(env, get_probs, get_out, nums, iters, score_type):

    rl_r = []
    results = defaultdict(list)

    for iter_num in range(0, iters):
        start = iter_num *args.len
        env.reset_for_test(nums,start)
        for k1, js in env.JOB_SCORES().items():
            for k2, ns in env.NODE_SCORES().items():
                results[(k1,k2)].append(sum(env.schedule_curr_sequence_reset_heterog(js, ns).values()))
                print(f'{k1}{k2}: {results[(k1,k2)][-1]}')
        [o, lst] = env.combine_observations(env.build_observation(), env.build_nodes_observation())
        print ("schedule: ", end="")
        rl = 0
        total_decisions = 0
        rl_decisions = 0
        while True:
            total_decisions += 1
            pi = get_probs(o, np.array(lst))
            a = pi[0]
            rl_decisions += 1

            # print (str(a)+"("+str(count)+")", end="|")
            [o, lst], r, d, _ = env.step_for_test(a)
            rl += r
            if d:
                print("Sequence Length:",total_decisions)
                break
        print(f'RL res: {rl}')
        rl_r.append(rl)
        print ("")

    # plot
    all_data = list(results.values())
    all_data.append(rl_r)

    all_medians = [np.median(p) for p in all_data]

    plt.rc("font", size=23)
    plt.figure(figsize=(9, 5))
    axes = plt.axes()

    xticks = [y + 1 for y in range(len(all_data))]
    for i in range(len(all_data)):
        plt.plot(xticks[i:i+1], all_data[i:i+1], 'o', color='darkorange')
        
    plt.boxplot(all_data, showfliers=False, meanline=True, showmeans=True, medianprops={"linewidth":0},meanprops={"color":"darkorange", "linewidth":4,"linestyle":"solid"})


    axes.yaxis.grid(True)
    axes.set_xticks([y + 1 for y in range(len(all_data))])
    xticklabels = [f'{k1}{k2}' for k1 in env.JOB_SCORES() for k2 in env.NODE_SCORES()] + ['SA']
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))], xticklabels=xticklabels)
    if score_type == BSLD:
        plt.ylabel("Average bounded slowdown", fontsize=15)
    elif score_type == AVGW:
        plt.ylabel("Average waiting time", fontsize=15)
    elif score_type == AVGT:
        plt.ylabel("Average turnaround time", fontsize=15)
    elif score_type == RESU:
        plt.ylabel("Resource utilization", fontsize=15)
    elif score_type == SLD:
        plt.ylabel("Slowdown", fontsize=15)
    else:
        raise NotImplementedError

    # plt.title('Comparison of results for the Squared Agent with Clustering (BSLD).', fontsize=14)
    # plt.title('Comparison of results for the Squared Agent (BSLD). Train: lublin_256 | Test: RICC-2010-2', fontsize=18)
    plt.title('Comparison of results for the Squared Agent (AVGW)', fontsize=18)
    # plt.title('Comparison of results for the Squared Agent with clustering (BSLD)', fontsize=18)
    # plt.ylabel("Average waiting time (s)")
    plt.xlabel("Scheduling Policies", fontsize=15)
    # plt.tick_params(axis='both', which='major', labelsize=40)
    # plt.tick_params(axis='both', which='minor', labelsize=40)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)

    # plt.savefig('data/NuevosExperimentos/DiferenteObjectivo/Figures/BoxplotExp_SLD_BSLD.png')
    plt.savefig('data/Pruebas/Test1epoch.png')
    # plt.savefig('data/ExperimentosTFG/Figures/BoxplotExp5_SqSLD.png')
    # plt.savefig('data/ExperimentosTFG/Figures/BoxplotExp6_SqAVGW.png')
    # plt.savefig('data/ExperimentosTFG/Figures/BoxplotExp7_SqBSLD_Clust_NEW.png')

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    # parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp4_SquaredSLD/Exp4_SquaredSLD_s2406")
    # parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp5_SquaredSLD/Exp5_SquaredSLD_s2406")
    parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp6_SquaredAVGW/Exp6_SquaredAVGW_s2406")
    # parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp6_SquaredAVGW/Exp6_SquaredAVGW_s2406")
    # parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp7_SquaredClustering/Exp7_SquaredClustering_s2406")
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')
    parser.add_argument('--platform', type=str, default='./data/cluster_x4_64procs.json')
    parser.add_argument('--len', '-l', type=int, default=1024)
    parser.add_argument('--seed', '-s', type=int, default=500)
    parser.add_argument('--iter', '-i', type=int, default=20)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    parser.add_argument('--enable_clustering', type=int, default=0)
    # parser.add_argument('--enable_clustering', type=int, default=1)

    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    platform_file = os.path.join(current_dir, args.platform)
    model_file = os.path.join(current_dir, args.rlmodel)

    get_probs, get_value = load_policy(model_file, 'last') 
    
    # initialize the environment from scratch
    env = HPCEnv(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, job_score_type=args.score_type,
                 batch_job_slice=args.batch_job_slice, build_sjf=False, enable_clustering=args.enable_clustering)
    env.my_init(workload_file=workload_file, platform_file=platform_file)
    env.seed(args.seed)
    random.seed(args.seed)

    start = time.time()
    run_policy(env, get_probs, get_value, args.len, args.iter, args.score_type)
    print("elapse: {}".format(time.time()-start))