from job import Job, Workloads
from cluster import Cluster, HeterogeneousNode as Node

import os
import math
import json
import time
import sys
import random

import numpy as np
import tensorflow as tf
import scipy.signal

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding


MAX_QUEUE_SIZE = 128

MAX_WAIT_TIME = 12 * 60 * 60 # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60 # assume maximal runtime is 12 hours

# each job has three features: wait_time, requested_node, runtime, machine states,
# JOB_FEATURES = 8
JOB_FEATURES = 4
DEBUG = False

# NUM_NODES = 1
# NODE_FEATURES = 0
NUM_NODES = 20 
NODE_FEATURES = 3

TOTAL_FEATURES = JOB_FEATURES + NODE_FEATURES

CRITIC_SIZE = 3

JOB_SEQUENCE_SIZE = 256

SKIP_TIME = 360 # skip 60 seconds


# OBJECTIVES
BSLD = 0 # Average bounded slowdown
AVGW = 1 # Average waiting time
AVGT = 2 # Average turnaround time
RESU = 3 # Resource utilization
SLD = 4 # Slowdown

MIN_OBS_VALUE = 1e-5
MAX_OBS_VALUE = 1.0 - MIN_OBS_VALUE



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class HPCEnv(gym.Env):

    def __init__(self,shuffle=False, backfil=False, skip=False, job_score_type=0, batch_job_slice=0, build_sjf=False):
        super(HPCEnv, self).__init__()
        print("Initialize Heterog HPC Env")

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE * NUM_NODES)
        self.jobs_action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.nodes_action_space = spaces.Discrete(NUM_NODES)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(MAX_QUEUE_SIZE *  NUM_NODES * TOTAL_FEATURES,),
                                            dtype=np.float32)

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.jobs = []
        self.nodes = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.start_idx_last_reset = 0

        self.loads = None
        self.cluster = None

        self.bsld_algo_dict = {}
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []
        self.enable_preworkloads = False
        self.pre_workloads = []

        self.shuffle = shuffle
        self.backfil = backfil
        self.skip = skip
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization
        self.job_score_type = job_score_type
        self.batch_job_slice = batch_job_slice

        self.build_sjf = build_sjf
        self.sjf_scores = []

    def my_init(self, workload_file='', platform_file='', sched_file=''):
        print ("Loading workloads from dataset:", workload_file)
        self.loads = Workloads(workload_file)
        self.cluster = Cluster(platform_file)

        # Continue for trajectory filtering
        if not self.build_sjf:
            return

        # TODO Implement trajectory filtering if necessary
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def f1_score(self, job: Job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        return (np.log10(request_time if request_time>0 else 0.1) * request_processors + 870 * np.log10(submit_time if submit_time>0 else 0.1))

    def f2_score(self, job: Job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # f2: r^(1/2)*n + 25600 * log10(s)
        return (np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time))

    def f3_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # f3: r * n + 6860000 * log10(s)
        return (request_time * request_processors + 6860000 * np.log10(submit_time))

    def f4_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # f4: r * sqrt(n) + 530000 * log10(s)
        return (request_time * np.sqrt(request_processors) + 530000 * np.log10(submit_time))

    def sjf_score(self, job):
        request_time = job.request_time
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier 
        return (request_time, submit_time)
    
    def smallest_score(self, job):
        request_processors = job.request_number_of_processors
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier 
        return (request_processors, submit_time)

    def wfp_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time
        return -np.power(float(waiting_time)/request_time, 3)*request_processors

    def uni_score(self,job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time-job.submit_time

        return -(waiting_time+1e-15)/(np.log2(request_processors+1e-15)*request_time)

    def fcfs_score(self, job):
        submit_time = job.submit_time
        return submit_time

    def job_score(self, job_for_scheduling: Job):
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown
        if self.job_score_type == BSLD:
            return max(1.0, (float(job_for_scheduling.finish_time - job_for_scheduling.submit_time)
                            / max(job_for_scheduling.request_time, 10)))
        if self.job_score_type == AVGW:
            return float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time)
        if self.job_score_type == AVGT:
            return float(job_for_scheduling.finish_time - job_for_scheduling.submit_time)
        if self.job_score_type == RESU: # TODO Esto igual se puede redefinir algo mejor
            return -float(job_for_scheduling.request_time*job_for_scheduling.request_number_of_processors)
        if self.job_score_type == SLD:
            return float(job_for_scheduling.finish_time - job_for_scheduling.submit_time)\
                /job_for_scheduling.request_time
        raise NotImplementedError

    def gen_preworkloads(self, size):
        raise NotImplementedError

    def refill_preworkloads(self):
        raise NotImplementedError

    def reset(self) -> tuple:
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.jobs = []
        self.nodes = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        assert self.batch_job_slice == 0 or self.batch_job_slice >= JOB_SEQUENCE_SIZE

        if self.build_sjf:
            # Trajectory filtering
            raise NotImplementedError
        else:
            size = self.loads.size() \
                if self.batch_job_slice == 0 \
                else self.batch_job_slice
            self.start = self.np_random.randint(JOB_SEQUENCE_SIZE, (size - JOB_SEQUENCE_SIZE - 1))

        for job in self.loads[self.start:self.start+JOB_SEQUENCE_SIZE]:
            self.cluster.events_queue.put(job.submit_time)

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = JOB_SEQUENCE_SIZE
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            raise NotImplementedError

        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.sjf_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f1_score).values()))     

        # return self.build_observation(), self.build_nodes_observation(), self.build_critic_observation()
        return self.combine_observations(self.build_observation(), self.build_nodes_observation())

    def reset_2nets(self) -> tuple:
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.jobs = []
        self.nodes = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        assert self.batch_job_slice == 0 or self.batch_job_slice >= JOB_SEQUENCE_SIZE

        if self.build_sjf:
            # Trajectory filtering
            raise NotImplementedError
        else:
            size = self.loads.size() \
                if self.batch_job_slice == 0 \
                else self.batch_job_slice
            self.start = self.np_random.randint(JOB_SEQUENCE_SIZE, (size - JOB_SEQUENCE_SIZE - 1))

        for job in self.loads[self.start:self.start+JOB_SEQUENCE_SIZE]:
            self.cluster.events_queue.put(job.submit_time)

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = JOB_SEQUENCE_SIZE
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            raise NotImplementedError

        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.sjf_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.f1_score).values()))

        return self.build_observation(with_mask=True)

    def reset_for_test(self, num,start):
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        job_sequence_size = num
        assert self.batch_job_slice == 0 or self.batch_job_slice>=job_sequence_size
        if self.batch_job_slice == 0:
            self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
        else:
            self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))
        #self.start = start
        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

    def build_observation(self, with_mask=False):
        vector = np.zeros(MAX_QUEUE_SIZE * JOB_FEATURES, dtype=float)
        mask = np.zeros(MAX_QUEUE_SIZE, dtype=float)
        self.job_queue.sort(key=lambda job: self.fcfs_score(job))
        self.visible_jobs = self.job_queue[:MAX_QUEUE_SIZE]
        if self.shuffle:
            random.shuffle(self.visible_jobs)
        
        ###
        # TODO Optimize observable jobs en vez de coger directamente los primeros segun fcfs
        ###

        self.jobs = []
        for i, job in enumerate(self.visible_jobs):
            submit_time = job.submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time
            wait_time = self.current_timestamp - submit_time # TODO Este timestamp esta updated?

            normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), MAX_OBS_VALUE)
            normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), MAX_OBS_VALUE)
            normalized_request_procs = min(float(request_processors) / float(self.loads.max_procs),  MAX_OBS_VALUE)

            can_schedule_now = MAX_OBS_VALUE if self.cluster.can_allocate(job) else MIN_OBS_VALUE
            mask[i] = 1
            self.jobs.append([ job, normalized_wait_time, normalized_run_time, normalized_request_procs, can_schedule_now ])

        needed_jobs = MAX_QUEUE_SIZE - len(self.jobs)
        self.jobs += [ [ None,0,1,1,0 ] ] * needed_jobs

        for i in range(0, MAX_QUEUE_SIZE):
            vector[i*JOB_FEATURES:(i+1)*JOB_FEATURES] = self.jobs[i][1:]
        self.build_critic_observation_heterog(vector)
        
        if with_mask:
            return vector, mask
        return vector

    def build_nodes_observation(self) -> np.ndarray:
        vector = np.zeros(NUM_NODES * NODE_FEATURES, dtype=float)
        self.nodes = []
        for i, node in enumerate(self.cluster.all_nodes):
            normalized_proc_number = min(float(node.total_procs)/float(self.loads.max_procs), MAX_OBS_VALUE)
            normalized_free_procs = min(float(node.free_procs)/float(node.total_procs), MAX_OBS_VALUE)
            normalized_frec = min(float(node.frec)/float(self.cluster.max_frec), MAX_OBS_VALUE)
            self.nodes.append([node, normalized_proc_number, normalized_free_procs, normalized_frec])
            vector[i*NODE_FEATURES:(i+1)*NODE_FEATURES] = self.nodes[i][1:]
        return vector

    def build_nodes_observation_for_job(self, job_idx) -> tuple:
        vector = np.zeros(NUM_NODES * TOTAL_FEATURES, dtype=float)
        mask = []
        job = self.jobs[job_idx][0]
        jo = self.jobs[job_idx][1:]
        self.nodes = []
        for i, node in enumerate(self.cluster.all_nodes):
            normalized_proc_number = min(float(node.total_procs)/float(self.loads.max_procs), MAX_OBS_VALUE)
            normalized_free_procs = min(float(node.free_procs)/float(node.total_procs), MAX_OBS_VALUE)
            normalized_frec = min(float(node.frec)/float(self.cluster.max_frec), MAX_OBS_VALUE)
            self.nodes.append([node, normalized_proc_number, normalized_free_procs, normalized_frec])
            vector[i*TOTAL_FEATURES:(i+1)*TOTAL_FEATURES] = jo + self.nodes[i][1:]
            mask.append(job is not None and job.request_number_of_processors <= node.free_procs)
        return vector, np.array(mask)

    def combine_observations(self, o: np.ndarray, no: np.ndarray) -> tuple:
        o_ = o.reshape(MAX_QUEUE_SIZE, JOB_FEATURES)
        no_ = no.reshape(NUM_NODES, NODE_FEATURES)
        vector = np.zeros(MAX_QUEUE_SIZE *  NUM_NODES * TOTAL_FEATURES)
        mask = []
        for i, [job, j_] in enumerate(zip(o_,self.jobs)):
            for j, [node, n_] in enumerate(zip(no_,self.nodes)):
                if j_[0] is None or j_[0].request_number_of_processors > n_[0].free_procs:
                    mask.append(0)
                    continue
                p = i * TOTAL_FEATURES * NUM_NODES + j * TOTAL_FEATURES
                vector[p:p+TOTAL_FEATURES] = np.concatenate((job, node))
                mask.append(1)
        return vector, np.array(mask)

    def build_critic_observation(self) -> np.ndarray:
        vector = np.zeros(JOB_SEQUENCE_SIZE * CRITIC_SIZE,dtype=float)
        earlist_job = self.loads[self.start_idx_last_reset]
        earlist_submit_time = earlist_job.submit_time
        jobs = []
        for i in range(self.start_idx_last_reset, self.last_job_in_batch+1):
            job = self.loads[i]
            submit_time = job.submit_time - earlist_submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time

            normalized_submit_time = min(float(submit_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
            normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
            normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1.0 - 1e-5)

            jobs.append([normalized_submit_time, normalized_run_time, normalized_request_nodes])

        for i in range(JOB_SEQUENCE_SIZE):
            vector[i*CRITIC_SIZE:(i+1)*CRITIC_SIZE] = jobs[i]
        return vector

    def build_critic_observation_heterog(self, jobs_obs: np.ndarray) -> None:
        nodes = np.zeros((NUM_NODES * NODE_FEATURES))
        for i, node in enumerate(self.cluster.all_nodes):
            normalized_proc_number = min(float(node.total_procs)/float(self.loads.max_procs), MAX_OBS_VALUE)
            normalized_free_procs = min(float(node.free_procs)/float(node.total_procs), MAX_OBS_VALUE)
            normalized_frec = min(float(node.frec)/float(self.cluster.max_frec), MAX_OBS_VALUE)
            nodes[i*NODE_FEATURES:(i+1)*NODE_FEATURES] = [normalized_proc_number, normalized_free_procs, normalized_frec]
        self.critic_obs = np.concatenate((jobs_obs.flatten(), nodes.flatten()))

    def get_critic_obs(self) -> np.ndarray:
        return self.critic_obs

    def shortest_job_req_procs(self):
        return max([j.request_number_of_processors for j in self.job_queue])

    def advance_time(self):
        self.current_timestamp = self.cluster.advance_to_next_time_event()
        self.receive_jobs()

    def step(self, a: int) -> list:
        job_for_scheduling = self.jobs[a//NUM_NODES][0]
        node_for_scheduling = self.nodes[a%NUM_NODES][0]

        if not job_for_scheduling or not node_for_scheduling:
            done = self.skip_schedule()
            # TODO Mirar si hay un sitio mejor para hjacer esto
            self.cluster.free_resources(self.current_timestamp)
        else:
            assert job_for_scheduling.request_number_of_processors <= node_for_scheduling.free_procs
            done = self.schedule(job_for_scheduling, node_for_scheduling)

        # TODO Igual por aqui habria que manejar lo de avanzar el tiempo

        if not done:
            while self.shortest_job_req_procs() > self.cluster.most_free_procs():
                self.advance_time()
            obs = self.combine_observations(self.build_observation(), self.build_nodes_observation())
            return [obs, 0, False, 0, 0, 0]

        self.post_process_score(self.scheduled_rl)
        rl_total = sum(self.scheduled_rl.values())
        best_total = min(self.scheduled_scores)
        sjf = self.scheduled_scores[0]
        f1 = self.scheduled_scores[1]
        rwd2 = (best_total - rl_total)
        rwd = -rl_total
        return [(None, None), rwd, True, rwd2, sjf, f1]

    def step_2nets(self, aj: int, an: int) -> list:
        job_for_scheduling = self.jobs[aj][0]
        node_for_scheduling = self.nodes[an][0]

        if not job_for_scheduling or not node_for_scheduling:
            done = self.skip_schedule()
            self.cluster.free_resources(self.current_timestamp)
        else:
            assert job_for_scheduling.request_number_of_processors <= node_for_scheduling.free_procs
            done = self.schedule(job_for_scheduling, node_for_scheduling)
        
        if not done:
            while self.shortest_job_req_procs() > self.cluster.most_free_procs():
                self.advance_time()
            obs = self.build_observation(with_mask=True)
            return [obs, 0, False, 0, 0, 0]
        
        self.post_process_score(self.scheduled_rl)
        rl_total = sum(self.scheduled_rl.values())
        best_total = min(self.scheduled_scores)
        sjf = self.scheduled_scores[0]
        f1 = self.scheduled_scores[1]
        rwd2 = (best_total - rl_total)
        rwd = -rl_total
        return [(None, None), rwd, True, rwd2, sjf, f1]

    def step_for_test(self, a):
        job_for_scheduling = self.jobs[a//NUM_NODES][0]
        node_for_scheduling = self.nodes[a%NUM_NODES][0]

        if not job_for_scheduling or not node_for_scheduling:
            done = self.skip_schedule()
            self.cluster.free_resources(self.current_timestamp)
        else:
            done = self.schedule(job_for_scheduling, node_for_scheduling)

        if not done:
            obs = self.combine_observations(self.build_observation(), self.build_nodes_observation())
            return [obs, 0, False, None]

        self.post_process_score(self.scheduled_rl)
        rl_total = sum(self.scheduled_rl.values())
        return [(None, None), rl_total, True, None]

    def post_process_score(self, scheduled_logs: dict):
        if self.job_score_type in (BSLD, AVGW, AVGT):
            for k in scheduled_logs:
                scheduled_logs[k] /= self.num_job_in_batch
            return
        if self.job_score_type in (RESU):
            total_cpu_hour = (self.current_timestamp - self.loads[self.start].submit_time)*self.loads.max_procs
            for i in scheduled_logs:
                scheduled_logs[i] /= total_cpu_hour
            return
        raise NotImplementedError

    def schedule(self, job:Job, node:Node = None) -> bool:
        # Job cannot be scheduled until node finishes executing the current job
        if not self.cluster.can_allocate(job, node):
            if self.backfil:
                raise NotImplementedError
            else:
                # Skips until current job finishes
                self.skip_for_resources(job, node)

        assert job.scheduled_time == -1
        assert job.request_number_of_processors <= node.free_procs
        self.cluster.allocate(job, self.current_timestamp, node)
        self.running_jobs.append(job)
        score = self.job_score(job)   # calculated reward
        self.scheduled_rl[job.job_id] = score
        self.job_queue.remove(job)  # remove the job from job queue
        return not self.moveforward_for_job()        

    def skip_schedule(self):
        raise NotImplementedError

    def schedule_curr_sequence_reset_heterog(self, job_fn, node_fn) -> dict:
        scheduled_logs = {}
        self.job_queue.sort(key=lambda j:job_fn(j))
        self.cluster.all_nodes.sort(key=lambda n:node_fn(n))
        while self.job_queue:
            job = self.job_queue.pop(0)
            node = self.cluster.next_resource(job)
            while not node or job.request_number_of_processors > node.free_procs:
                # TODO Mirar esta funcion
                node = self.skip_for_resources_greedy_heterog(job)
                # self.skip_for_resources_greedy(job, scheduled_logs)
            assert job.scheduled_time == -1
            assert job.request_number_of_processors <= node.free_procs
            self.cluster.allocate(job, self.current_timestamp, node)
            self.running_jobs.append(job)
            score = self.job_score(job)
            scheduled_logs[job.job_id] = score
            self.moveforward_for_job() # TODO Esto igual hay que revisarlo
        
        self.post_process_score(scheduled_logs)
        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        for job in self.loads[self.start:self.start+JOB_SEQUENCE_SIZE]:
            self.cluster.events_queue.put(job.submit_time)
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            raise NotImplementedError

        return scheduled_logs

    def schedule_curr_sequence_reset(self, score_fn) -> dict:
        scheduled_logs = {}
        self.job_queue.sort(key=lambda j:score_fn(j))
        while self.job_queue:
            job_for_scheduling = self.job_queue.pop(0)
            if not self.cluster.can_allocate(job_for_scheduling):
                #TODO Implement backfill y meter if
                self.skip_for_resources_greedy(job_for_scheduling, scheduled_logs)
            assert job_for_scheduling.scheduled_time == -1
            self.cluster.allocate(job_for_scheduling, self.current_timestamp)
            self.running_jobs.append(job_for_scheduling)
            score = self.job_score(job_for_scheduling)
            scheduled_logs[job_for_scheduling.job_id] = score
            self.moveforward_for_job() # TODO Esto igual hay que revisarlo

        self.post_process_score(scheduled_logs)
        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        for job in self.loads[self.start:self.start+JOB_SEQUENCE_SIZE]:
            self.cluster.events_queue.put(job.submit_time)
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        if self.enable_preworkloads:
            raise NotImplementedError

        return scheduled_logs

    def skip_for_resources(self, job:Job, node:Node = None):

        assert not self.cluster.can_allocate(job, node)
        
        while job.request_number_of_processors > node.free_procs:
            self.current_timestamp = self.cluster.advance_to_next_time_event()      
            self.receive_jobs()
        # TODO Por aqui no se si habra que hacer mas cosas
    
    def receive_jobs(self):
        while self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= self.current_timestamp:
            self.job_queue.append(self.loads[self.next_arriving_job_idx])
            self.next_arriving_job_idx += 1

    def skip_for_resources_greedy(self, job:Job, scheduled_logs:dict):
        # TODO Revisar esto que está solo copiado
        assert not self.cluster.can_allocate(job)

        while not self.cluster.can_allocate(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: running_job.finish_time)
            next_resource_release_time = (self.running_jobs[0].finish_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def skip_for_resources_greedy_heterog(self, job):
        self.current_timestamp = self.cluster.advance_to_next_time_event()
        return self.cluster.next_resource(job)


    def moveforward_for_job(self) -> bool:
        # TODO Revisar esto que está solo copiado

        if self.job_queue:
            return True

        # Queue is empty or there are no more jobs to arrive
        if self.next_arriving_job_idx >= self.last_job_in_batch:
            return False

        while not self.job_queue:
            if self.running_jobs:
                self.running_jobs.sort(key=lambda rj: rj.finish_time)
                next_job = self.running_jobs[0]
                next_resource_release_time = next_job.finish_time
                next_resource_release_machines = next_job.allocated_machines
            else:
                next_resource_release_time = sys.maxsize
                next_resource_release_machines = []
            
            if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True

            self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
            self.cluster.release(next_resource_release_machines)
            self.running_jobs.pop(0)


        raise NotImplementedError

    def moveforward_for_resources_backfill(self, job):
        raise NotImplementedError

    def moveforward_for_resources_backfill_greedy(self, job, scheduled_logs):
        raise NotImplementedError
    


    # SORT JOBS
    def JOB_SCORES(self):
        return {
            'smallest': self.smallest_score, 
            'shortest': self.sjf_score, 
            'first': self.fcfs_score
        }

    # SORT NODES
    def smallest_node(self, n):
        return n.total_procs

    def biggest_node(self, n):
        return -n.total_procs

    def lowest_frec(self, n):
        return n.frec

    def highest_frec(self, n):
        return -n.frec

    def NODE_SCORES(self):
        return {
            'smallest': self.smallest_node,
            'biggest': self.biggest_node,
            'lowest': self.lowest_frec,
            'highest': self.highest_frec
        }
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = HPCEnv(batch_job_slice=100, build_sjf=True)
    env.seed(0)
    env.my_init(workload_file=workload_file, sched_file=workload_file)