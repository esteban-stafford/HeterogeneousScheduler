import numpy as np
import tensorflow as tf
import gym
import os
import sys
import time

import spinup

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.logx import restore_tf_graph
import os.path as osp
# from HPCSimPickJobs import *
from HPCSimPickJobsHeterog import *
from pprint import pprint

from tensorflow.python.util import deprecation
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Join Losses Types
MEAN = 0
SUM = 1
MIN = 2

def critic_mlp(x, act_dim):
    x = tf.reshape(x, shape=[-1,MAX_QUEUE_SIZE, JOB_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    return tf.layers.dense(x, units=act_dim)

def critic_combined(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE * JOB_FEATURES + NUM_NODES * NODE_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    # x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    # x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    # x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    # x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    return tf.layers.dense(x, units=act_dim)

def rl_kernel_jobs(x, act_dim):
    x = tf.reshape(x, shape=[-1, MAX_QUEUE_SIZE, JOB_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    return x

def rl_kernel_nodes(x, act_dim):
    x = tf.reshape(x, shape=[-1, NUM_NODES, TOTAL_FEATURES])
    x = tf.layers.dense(x, units=32, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=16, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=8, activation=tf.nn.relu)
    x = tf.squeeze(tf.layers.dense(x, units=1), axis=-1)
    return x

"""
Policies
"""
def categorical_policy(x, a, mask, action_space, jobs=False, nodes=False):
    act_dim = action_space.n
    if jobs:
        output_layer = rl_kernel_jobs(x, act_dim)
    elif nodes:
        output_layer = rl_kernel_nodes(x, act_dim)
    else:
        raise NotImplementedError
    output_layer = output_layer+(mask-1)*1000000
    logp_all = tf.nn.log_softmax(output_layer)

    pi = tf.squeeze(tf.multinomial(output_layer, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi, output_layer

"""
Actor-Critics
"""
def actor_critic_jobs(x, c, a, mask, jobs_action_space=None):
    with tf.variable_scope('pi_j'):
        pi_j, logp_j, logp_pi_j, out_j = categorical_policy(x, a, mask, jobs_action_space, jobs=True)
    with tf.variable_scope('v'):
        v = tf.squeeze(critic_combined(c, 1), axis=-1)
        # v = critic_combined(c, 1)
    return pi_j, logp_j, logp_pi_j, v, out_j

def actor_critic_nodes(x, a, mask, nodes_action_space=None):
    with tf.variable_scope('pi_n'):
        pi_n, logp_n, logp_pi_n, out_n = categorical_policy(x, a, mask, nodes_action_space, nodes=True)
    return pi_n, logp_n, logp_pi_n, out_n

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, nobs_dim, cobs_dim, actj_dim, actn_dim, size, gamma=0.99, lam=0.95):
        self.jobs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.nobs_buf = np.zeros(combined_shape(size, nobs_dim), dtype=np.float32)
        self.cobs_buf = np.zeros(combined_shape(size, cobs_dim), dtype=np.float32)
        size = size * 100 # assume the traj can be really long
        self.actj_buf = np.zeros(combined_shape(size, actj_dim), dtype=np.float32)
        self.actn_buf = np.zeros(combined_shape(size, actn_dim), dtype=np.float32)
        self.maskj_buf = np.zeros(combined_shape(size, MAX_QUEUE_SIZE), dtype=np.float32)
        self.maskn_buf = np.zeros(combined_shape(size, NUM_NODES), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logpj_buf = np.zeros(size, dtype=np.float32)
        self.logpn_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, jobs, nobs, cobs, actj, actn, maskj, maskn, rew, val, logpj, logpn):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.jobs_buf[self.ptr] = jobs
        self.nobs_buf[self.ptr] = nobs
        self.cobs_buf[self.ptr] = cobs
        self.actj_buf[self.ptr] = actj
        self.actn_buf[self.ptr] = actn
        self.maskj_buf[self.ptr] = maskj
        self.maskn_buf[self.ptr] = maskn
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logpj_buf[self.ptr] = logpj
        self.logpn_buf[self.ptr] = logpn
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr < self.max_size
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0

        actual_adv_buf = np.array(self.adv_buf, dtype = np.float32)
        actual_adv_buf = actual_adv_buf[:actual_size]
        # print ("-----------------------> actual_adv_buf: ", actual_adv_buf)
        adv_sum = np.sum(actual_adv_buf)
        adv_n = len(actual_adv_buf)
        adv_mean = adv_sum / adv_n
        adv_sum_sq = np.sum((actual_adv_buf - adv_mean) ** 2)
        adv_std = np.sqrt(adv_sum_sq / adv_n) + 1e-5
        # print ("-----------------------> adv_std:", adv_std)
        actual_adv_buf = (actual_adv_buf - adv_mean) / adv_std
        # print (actual_adv_buf)

        return [
            self.jobs_buf[:actual_size], self.nobs_buf[:actual_size], self.cobs_buf[:actual_size],
            self.actj_buf[:actual_size], self.actn_buf[:actual_size], 
            self.maskj_buf[:actual_size], self.maskn_buf[:actual_size], 
            actual_adv_buf, self.ret_buf[:actual_size], 
            self.logpj_buf[:actual_size], self.logpn_buf[:actual_size]
        ]

"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
def ppo(workload_file, platform_file, model_path, ac_kwargs=dict(), seed=0, 
        traj_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,pre_trained=0,trained_model=None,attn=False,shuffle=False,
        backfil=False, skip=False, score_type=0, batch_job_slice=0, join_loss_type=MEAN):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = HPCEnv(shuffle=shuffle, backfil=backfil, skip=skip, job_score_type=score_type, batch_job_slice=batch_job_slice, build_sjf=False)
    env.seed(seed)
    env.my_init(workload_file=workload_file, platform_file=platform_file, sched_file=model_path)
    
    # obs_dim = env.observation_space.shape
    jobs_dim = (MAX_QUEUE_SIZE * JOB_FEATURES,)
    nobs_dim = (NUM_NODES * TOTAL_FEATURES,)
    cobs_dim = (MAX_QUEUE_SIZE * JOB_FEATURES + NUM_NODES * NODE_FEATURES,)
    # act_dim = env.action_space.shape
    actj_dim = env.jobs_action_space.shape
    actn_dim = env.nodes_action_space.shape
    
    # Share information about action space with policy architecture
    jobs_action_space = env.jobs_action_space
    nodes_action_space = env.nodes_action_space

    # Inputs to computation graph

    buf = PPOBuffer(jobs_dim, nobs_dim, cobs_dim, actj_dim, actn_dim, traj_per_epoch * JOB_SEQUENCE_SIZE, gamma, lam)

    if pre_trained:
        sess = tf.Session()
        model = restore_tf_graph(sess, trained_model)
        logger.log('load pre-trained model')
        # Count variables
        var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
        logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

        xj_ph = model['xj']
        xn_ph = model['xn']
        a_ph = model['a']
        mask_ph = model['mask']
        adv_ph = model['adv']
        ret_ph = model['ret']
        logp_old_ph = model['logp_old_ph']

        pi_j = model['pi_j']
        pi_n = model['pi_n']
        v = model['v']
        # logits = model['logits']
        out_j = model['out_j']
        out_n = model['out_n']
        logp_j = model['logp_j']
        logp_n = model['logp_n']
        logp_pi_j = model['logp_pi_j']
        logp_pi_n = model['logp_pi_n']
        pi_loss_j = model['pi_loss_j']
        pi_loss_n = model['pi_loss_n']
        v_loss = model['v_loss']
        approx_ent = model['approx_ent']
        approx_kl = model['approx_kl']
        clipfrac = model['clipfrac']
        clipped = model['clipped']

        # Optimizers
        train_pi = tf.get_collection("train_pi")[0]
        train_v = tf.get_collection("train_v")[0]

        # Need all placeholders in *this* order later (to zip with data from buffer)
        all_phs = [xj_ph, xn_ph, aj_ph, an_ph, maskj_ph, maskn_ph, adv_ph, ret_ph, logpj_old_ph, logpn_old_ph]
        # Every step, get: action, value, and logprob
        get_job_action_ops = [pi_j, v, logp_pi_j, out_j]

    else:
        xj_ph = placeholder(MAX_QUEUE_SIZE * JOB_FEATURES)
        xn_ph = placeholder(NUM_NODES * TOTAL_FEATURES)
        co_ph = placeholder(MAX_QUEUE_SIZE * JOB_FEATURES + NUM_NODES * NODE_FEATURES)
        aj_ph, an_ph = placeholders_from_spaces(env.jobs_action_space, env.nodes_action_space)

        # y_ph = placeholder(JOB_SEQUENCE_SIZE*3) # 3 is the number of sequence features
        maskj_ph = placeholder(MAX_QUEUE_SIZE)
        maskn_ph = placeholder(NUM_NODES)
        adv_ph, ret_ph, logpj_old_ph, logpn_old_ph = placeholders(None, None, None, None)

        # Main outputs from computation graph
        pi_j, logp_j, logp_pi_j, v, out_j = actor_critic_jobs(xj_ph, co_ph, aj_ph, maskj_ph, jobs_action_space)
        pi_n, logp_n, logp_pi_n, out_n = actor_critic_nodes(xn_ph, an_ph, maskn_ph, nodes_action_space)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        all_phs = [xj_ph, xn_ph, co_ph, aj_ph, an_ph, maskj_ph, maskn_ph, adv_ph, ret_ph, logpj_old_ph, logpn_old_ph]

        # Every step, get: action, value, and logprob
        get_job_action_ops = [pi_j, v, logp_pi_j, out_j]
        get_node_action_ops = [pi_n, logp_pi_n, out_n]

        # Experience buffer

        # Count variables
        var_counts = tuple(count_vars(scope) for scope in ['pi_j', 'pi_n', 'v'])
        logger.log('\nNumber of parameters: \t pi_j: %d, \t pi_n: %d, \t v: %d\n' % var_counts)

        # PPO objectives
        ratio_j = tf.exp(logp_j - logpj_old_ph)  # pi(aj|s) / pi_old(aj|s)
        ratio_n = tf.exp(logp_n - logpn_old_ph)
        min_adv = tf.where(adv_ph > 0, (1 + clip_ratio) * adv_ph, (1 - clip_ratio) * adv_ph)
        # Aqui hay que hacer algo para unir los ratios #####################
        if join_loss_type == MEAN:
            join_ratios = tf.reduce_mean(tf.stack([ratio_j, ratio_n]), axis=0)
        if join_loss_type == SUM:
            join_ratios = ratio_j + ratio_n
        if join_loss_type == MIN:
            join_ratios = tf.minimum(ratio_j, ratio_n)
        pi_loss = -tf.reduce_mean(tf.minimum(join_ratios * adv_ph, min_adv))
        #########################################################################
        v_loss = tf.reduce_mean((ret_ph - v) ** 2)

        # Info (useful to watch during learning) - JOBS
        approx_kl_j = tf.reduce_mean(logpj_old_ph - logp_j)  # a sample estimate for KL-divergence, easy to compute
        approx_ent_j = tf.reduce_mean(-logp_j)  # a sample estimate for entropy, also easy to compute
        clipped_j = tf.logical_or(ratio_j > (1 + clip_ratio), ratio_j < (1 - clip_ratio))
        clipfrac_j = tf.reduce_mean(tf.cast(clipped_j, tf.float32))

        # Info (useful to watch during learning) - NODES
        approx_kl_n = tf.reduce_mean(logpn_old_ph - logp_n)
        approx_ent_n = tf.reduce_mean(-logp_n)
        clipped_n = tf.logical_or(ratio_n > (1 + clip_ratio), ratio_n < (1 - clip_ratio))
        clipfrac_n = tf.reduce_mean(tf.cast(clipped_n, tf.float32))

        # Optimizers
        train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
        train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.add_to_collection("train_pi", train_pi)
        tf.add_to_collection("train_v", train_v)


    # Setup model saving
    # logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'action_probs': action_probs, 'log_picked_action_prob': log_picked_action_prob, 'v': v})
    logger.setup_tf_saver(sess, 
        inputs = {
            'xj': xj_ph, 'xn': xn_ph, 'co': co_ph, 'aj':aj_ph, 'an': an_ph, 
            'adv':adv_ph, 'maskj':maskj_ph, 'maskn': maskn_ph, 'ret':ret_ph, 
            'logpj_old_ph':logpj_old_ph, 'logpn_old_ph': logpn_old_ph
        }, 
        outputs = {
            'pi_j': pi_j, 'pi_n': pi_n, 'v': v, 'out_j': out_j, 'out_n': out_n, 
            'pi_loss': pi_loss, 'logpj': logp_j, 'logpn': logp_n, 'logp_pi_j':logp_pi_j, 'logp_pi_n':logp_pi_n,
            'v_loss': v_loss, 'approx_ent_j':approx_ent_j, 'approx_ent_n':approx_ent_n, 
            'approx_kl_j': approx_kl_j, 'approx_kl_n': approx_kl_n, 
            'clipped_j': clipped_j, 'clipped_n': clipped_n, 
            'clipfrac_j': clipfrac_j, 'clipfrac_n': clipfrac_n
        })

    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        pi_l_old, n_l_old, v_l_old, entj, entn = sess.run([pi_loss, v_loss, approx_ent_j, approx_ent_n], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, _, klj, kln = sess.run([train_pi, approx_kl_j, approx_kl_n], feed_dict=inputs)
            klj = mpi_avg(klj)
            if klj > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, n_l_new, v_l_new, klj, kln, cfj, cfn = sess.run([pi_loss, v_loss, approx_kl_j, approx_kl_n, clipfrac_j, clipfrac_n], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossN=n_l_old, LossV=v_l_old,
                     KLj=klj, KLn=kln, EntropyJ=entj, EntropyN=entn, 
                     ClipFracJ=cfj, ClipFracN=cfn,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    [jo, lstj], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset_2nets(), 0, False, 0, 0,0,0,0
    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    num_total = 0
    for epoch in range(epochs):
        t = 0
        while True:
            
            co = env.get_critic_obs()
            a_j, v_t, logp_t_j, output_j = sess.run(get_job_action_ops, feed_dict={xj_ph: jo.reshape(1,-1), co_ph: co.reshape(1,-1), maskj_ph: lstj.reshape(1,-1)})

            no, lstn = env.build_nodes_observation_for_job(a_j[0])

            a_n, logp_t_n, output_n = sess.run(get_node_action_ops, feed_dict={xn_ph: no.reshape(1,-1), maskn_ph: lstn.reshape(1,-1)})

            num_total += 1
            '''
            action = np.random.choice(np.arange(MAX_QUEUE_SIZE), p=action_probs)
            log_action_prob = np.log(action_probs[action])
            '''

            # save and log
            buf.store(jo, no, co, a_j, a_n, lstj, lstn, r, v_t, logp_t_j, logp_t_n)
            logger.store(VVals=v_t)

            # print('ACTION:', a)

            [jo, lstj], r, d, r2, sjf_t, f1_t = env.step_2nets(a_j[0], a_n[0])
            ep_ret += r
            ep_len += 1
            show_ret += r2
            sjf += sjf_t
            f1 += f1_t

            if d:
                t += 1
                buf.finish_path(r)
                logger.store(EpRet=ep_ret, EpLen=ep_len, ShowRet=show_ret, SJF=sjf, F1=f1)
                [jo, lstj], r, d, ep_ret, ep_len, show_ret, sjf, f1 = env.reset_2nets(), 0, False, 0, 0, 0, 0, 0
                if t >= traj_per_epoch:
                    # print ("state:", state, "\nlast action in a traj: action_probs:\n", action_probs, "\naction:", action)
                    break
        # print("Sample time:", (time.time()-start_time)/num_total, num_total)
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            # logger.save_state({'env': env}, None)
            logger.save_state({}, None)

        # Perform PPO update!
        # start_time = time.time()
        update()
        # print("Train time:", time.time()-start_time)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)* traj_per_epoch * JOB_SEQUENCE_SIZE)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossN', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('EntropyJ', average_only=True)
        logger.log_tabular('EntropyN', average_only=True)
        logger.log_tabular('KLj', average_only=True)
        logger.log_tabular('KLn', average_only=True)
        logger.log_tabular('ClipFracJ', average_only=True)
        logger.log_tabular('ClipFracN', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('ShowRet', average_only=True)
        logger.log_tabular('SJF', average_only=True)
        logger.log_tabular('F1', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2 lublin_256.swf SDSC-SP2-1998-4.2-cln.swf
    parser.add_argument('--platform', type=str, default='./data/cluster_x4_64procs.json')
    # parser.add_argument('--model', type=str, default='./data/lublin_256.schd')
    parser.add_argument('--model', type=str, default='./data/cluster_x1248.json')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trajs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--pre_trained', type=int, default=0)
    parser.add_argument('--trained_model', type=str, default='./data/logs/ppo_temp/ppo_temp_s0')
    parser.add_argument('--attn', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--backfil', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--score_type', type=int, default=0)
    parser.add_argument('--batch_job_slice', type=int, default=0)
    parser.add_argument('--join_loss_type', type=int, default=0)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    
    # build absolute path for using in hpc_env.
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)
    platform_file = os.path.join(current_dir, args.platform)
    log_data_dir = os.path.join(current_dir, './data/logs/')
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)
    if args.pre_trained:
        model_file = os.path.join(current_dir, args.trained_model)

        ppo(workload_file, platform_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pre_trained=1,trained_model=os.path.join(model_file,"simple_save"),attn=args.attn,
            shuffle=args.shuffle, backfil=args.backfil, skip=args.skip, score_type=args.score_type,
            batch_job_slice=args.batch_job_slice)
    else:
        ppo(workload_file, platform_file, args.model, gamma=args.gamma, seed=args.seed, traj_per_epoch=args.trajs, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pre_trained=0, attn=args.attn,shuffle=args.shuffle, backfil=args.backfil,
            skip=args.skip, score_type=args.score_type, batch_job_slice=args.batch_job_slice, join_loss_type=args.join_loss_type)
