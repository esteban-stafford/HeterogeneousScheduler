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
    pi = model['pi']
    v = model['v']
    out = model['out']
    get_out = lambda x ,y  : sess.run(out, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES), model['mask']:y.reshape(-1, MAX_QUEUE_SIZE)})
    # make function for producing an action given a single state
    get_probs = lambda x ,y  : sess.run(pi, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * env.NUM_NODES * TOTAL_FEATURES), model['mask']:y.reshape(-1, MAX_QUEUE_SIZE*env.NUM_NODES)})
    get_v = lambda x : sess.run(v, feed_dict={model['x']: x.reshape(-1, MAX_QUEUE_SIZE * JOB_FEATURES)})
    return get_probs, get_out


def run_policy(env, models, nums, iters):
    for iter_num in range(0, iters):
        start = iter_num *args.len
        env.reset_for_test(nums,start)
        for k1, js in env.JOB_SCORES().items():
            for k2, ns in env.NODE_SCORES().items():
                averages = env.schedule_curr_sequence_reset_heterog(js, ns)
                print(f'{iter_num} {k1}{k2} ' + ' '.join([str(averages[k]) for k in sorted(averages.keys())]))
 
        for model in models:
            model_file = os.path.join(current_dir, model)
            get_probs, get_value = load_policy(model_file, 'last')
            total_decisions = 0
            rl_decisions = 0
            averages = {}
            
            [o, lst] = env.combine_observations(env.build_observation(), env.build_nodes_observation())
            while True:
                total_decisions += 1
                pi = get_probs(o, np.array(lst))
                action = pi[0]
                rl_decisions += 1

                [o, lst], metrics, d, _ = env.step_for_test(action)
                
                if d:
                    for metric, value in metrics.items():
                        averages[metric] = value
                    env.reset()
                    break
            print(f'{iter_num} {model} ' + ' '.join([str(averages[metric]) for metric in sorted(averages.keys())]))


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    # parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp4_SquaredSLD/Exp4_SquaredSLD_s2406")
    # parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp5_SquaredSLD/Exp5_SquaredSLD_s2406")
    parser.add_argument('--rlmodel', type=str, default="./data/logs/Exp6_SquaredAVGW/Exp6_SquaredAVGW_s2406", nargs="+")
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
    parser.add_argument('--batch_job_slice', type=int, default=0)
    parser.add_argument('--enable_clustering', type=int, default=0)
    # parser.add_argument('--enable_clustering', type=int, default=1)

    args = parser.parse_args()

    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    platform_file = os.path.join(current_dir, args.platform)

    # initialize the environment from scratch
    env = HPCEnv(shuffle=args.shuffle, backfil=args.backfil, skip=args.skip,
                 batch_job_slice=args.batch_job_slice, build_sjf=False, enable_clustering=args.enable_clustering)
    env.my_init(workload_file=workload_file, platform_file=platform_file)
    env.seed(args.seed)
    random.seed(args.seed)

    print("iteration","scheduler","BSLD","AVGW","AVGT","SLD")
    start = time.time()
    run_policy(env, args.rlmodel, args.len, args.iter)
    print("elapse: {}".format(time.time()-start), file=sys.stderr)
