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
        saves = [int(x[11:]) for x in os.listdir(model_path) if 'tf1_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(model_path, 'tf1_save'+itr))

    # get the correct op for executing actions
    pi_j = model['pi_j']
    pi_n = model['pi_n']
    v = model['v']
    out_j = model['out_j']
    out_n = model['out_n']
    # get_out = lambda x ,y  : sess.run(out, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES), model['mask']:y.reshape(-1, MAX_QUEUE_SIZE)})
    # make function for producing an action given a single state
    get_jobs = lambda x ,y  : sess.run(pi_j, feed_dict={model['xj']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES), model['maskj']:y.reshape(-1, MAX_QUEUE_SIZE)})
    get_nodes = lambda x ,y  : sess.run(pi_n, feed_dict={model['xn']: x.reshape(-1, NUM_NODES * TOTAL_FEATURES), model['maskn']:y.reshape(-1, NUM_NODES)})
    # get_v = lambda x : sess.run(v, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
    # return get_probs, get_out
    return get_jobs, get_nodes


#@profile
def run_policy(env, get_jobs, get_nodes, nums, iters, score_type):

    rl_r = []
    results = defaultdict(list)

    for iter_num in range(0, iters):
        start = iter_num *args.len
        env.reset_for_test(nums,start) # TODO
        for k1, js in env.JOB_SCORES().items():
            for k2, ns in env.NODE_SCORES().items():
                results[(k1,k2)].append(sum(env.schedule_curr_sequence_reset_heterog(js, ns).values()))

        jo, lstj = env.build_observation(with_mask=True)
        print ("schedule: ", end="")
        rl = 0
        total_decisions = 0
        rl_decisions = 0
        while True:
            total_decisions += 1
            
            pi_j = get_jobs(jo, lstj)
            a_j = pi_j[0]
            
            no, lstn = env.build_nodes_observation_for_job(a_j)
            
            pi_n = get_nodes(no, lstn)
            a_n = pi_n[0]

            rl_decisions += 1

            # print (str(a)+"("+str(count)+")", end="|")
            [o, lstj], r, d, _ = env.step_for_test_2nets(a_j, a_n)
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
    xticklabels = [f'{k1}{k2}' for k1 in env.JOB_SCORES() for k2 in env.NODE_SCORES()] + ['DA']
    # xticklabels = [f'{k1}_{k2}' for k1 in env.JOB_SCORES() for k2 in env.NODE_SCORES()] + ['rl']
    # xticklabels = [f'{i*len(env.JOB_SCORES())+j+1}' for i in range(len(env.JOB_SCORES())) for j in range(len(env.NODE_SCORES()))] + ['MEAN']
    plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
             xticklabels=xticklabels)
    if score_type == 0:
        plt.ylabel("Average bounded slowdown", fontsize=15)
    elif score_type == 1:
        plt.ylabel("Average waiting time", fontsize=15)
    elif score_type == 2:
        plt.ylabel("Average turnaround time", fontsize=15)
    elif score_type == 3:
        plt.ylabel("Resource utilization", fontsize=15)
    else:
        raise NotImplementedError

    plt.title('Comparison of results for the SEP Double Agent', fontsize=20)
    # plt.ylabel("Average waiting time (s)")
    plt.xlabel("Scheduling Policies", fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)

    # plt.tight_layout()
    # plt.savefig('data/ExperimentosTFG/Figures/BoxplotExp2_MEAN.png')
    plt.savefig('data/ExperimentosTFG/Figures/BoxplotExp3_2opt.png')

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    # parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp2_DoubleMEAN/Exp2_DoubleMEAN_s2406")
    parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp3_Double2opt/Exp3_Double2opt_s2406")
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')
    parser.add_argument('--platform', type=str, default='./data/cluster_x4_64procs.json')
    parser.add_argument('--len', '-l', type=int, default=1024)
    parser.add_argument('--seed', '-s', type=int, default=500)
    parser.add_argument('--iter', '-i', type=int, default=20)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=BSLD)
    parser.add_argument('--batch_job_slice', type=int, default=0)

    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    platform_file = os.path.join(current_dir, args.platform)
    model_file = os.path.join(current_dir, args.rlmodel)

    get_jobs, get_nodes = load_policy(model_file, 'last') 
    
    # initialize the environment from scratch
    env = HPCEnv(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, job_score_type=args.score_type,
                 batch_job_slice=args.batch_job_slice, build_sjf=False)
    env.my_init(workload_file=workload_file, platform_file=platform_file)
    env.seed(args.seed)

    start = time.time()
    run_policy(env, get_jobs, get_nodes, args.len, args.iter, args.score_type)
    print("elapse: {}".format(time.time()-start))