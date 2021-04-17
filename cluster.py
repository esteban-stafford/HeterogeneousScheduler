import math
import json


            

class Processor:

    def __init__(self, id, node_id):
        self.id = f'proc_{id}'
        self.node_id = node_id
        self.running_job_id = -1
        self.running_job_finishtime = -1
        self.is_free = True
        self.job_history = []

    def taken_by_job(self, job_id, finish_time):
        if not self.is_free:
            return False
        self.running_job_id = job_id
        self.running_job_finishtime = finish_time
        self.is_free = False
        self.job_history.append(job_id)
        return True
    
    def release(self) -> bool :
        if self.is_free:
            return False
        self.running_job_id = -1
        self.running_job_finishtime = -1
        self.is_free = True
        return True

    def free_resource(self, current_time) -> bool:
        if self.is_free or current_time < self.running_job_finishtime:
            return False
        return self.release()

    def reset(self):
        self.running_job_id = -1
        self.running_job_finishtime = -1
        self.is_free = True
        self.job_history = []

    def __eq__(self, other):
        
        return other != None and self.id == other.id
    
    def __str__(self):
        return f'Proc[ID:{self.id}]-[N:{self.node_id}]-[Ends:{self.running_job_finishtime}]'

class HeterogeneousNode:
    
    def __init__(self, id, num_procs, frec, base_frec):
        self.id = f'{id}'
        self.frec = frec
        self.rel_frec = base_frec / frec 
        self.is_free = True
        self.total_procs = num_procs
        self.free_procs = num_procs
        self.used_procs = 0
        self.all_procs = [Processor(f'{id}_{i}', self.id) for i in range(num_procs)]

    def allocate(self, job):
        if not self.is_free or self.free_procs < job.request_number_of_processors:
            return False
        self.used_procs += job.request_number_of_processors
        self.free_procs -= job.request_number_of_processors
        allocated = []
        req_procs = job.request_number_of_processors
        job.finish_time = job.scheduled_time + int(job.request_time * self.rel_frec)
        for proc in self.all_procs:
            if proc.taken_by_job(job.job_id, job.finish_time):
                req_procs -= 1
                allocated.append(proc)
            if not req_procs:
                break
        return allocated
        
    def free_resources(self, current_time) -> bool:
        if not self.used_procs:
            return False
        freed = sum([p.free_resource(current_time) for p in self.all_procs])
        self.free_procs += freed
        self.used_procs -= freed
        return self.used_procs == 0

    def release(self, releases):
        num_released = sum([p.release() for p in releases])
        self.used_procs -= num_released
        self.free_procs += num_released
        return num_released > 0 and self.free_procs == num_released

    def reset(self):
        self.used_procs = 0
        self.free_procs = self.total_procs
        for p in self.all_procs:
            p.reset()

    def __str__(self):
        return f'Node[ID:{self.id}]-[Procs:{self.free_procs}/{self.total_procs}]'

    def __contains__(self, proc):
        return self.id == proc.node_id

class HeterogeneousCluster:

    def __init__(self, path):
        cluster = load_cluster(path)
        self.name = cluster['id']
        self.all_nodes = []
        
        self.min_frec = min([n['frec'] for n in cluster['nodes']])
        self.max_frec = max([n['frec'] for n in cluster['nodes']])

        for n in cluster['nodes']:
            for i in range(n['number']):
                name, num_procs, frec = n['id'], n['num_procs'], n['frec']
                self.max_frec = max(self.max_frec, frec)
                self.all_nodes.append(HeterogeneousNode(f'{name}_{i}', num_procs, frec, self.min_frec))
        # print('CLUSTER: ', [str(n) for n in self.all_nodes])

    def can_allocate(self, job, node=None):
        if node:
            return job.request_number_of_processors <= node.free_procs
        
        # if job.request_number_of_nodes != -1:
        #     return job.request_number_of_nodes <= self.free_node
        return any(n.free_procs >= job.request_number_of_processors for n in self.all_nodes)
        

    def allocate(self, job, node=None):
        if not node:
            node = next((n for n in self.all_nodes if n.free_procs >= job.request_number_of_processors), None)
        if not node or not self.can_allocate(job, node):
            return []

        allocated_procs = node.allocate(job)
        return allocated_procs
        
    def free_resources(self, current_time):
        for n in self.all_nodes:
            freed = n.free_resources(current_time)

    def release(self, releases):
        node = next((n for n in self.all_nodes if releases[0] in n), None)
        if not node:
            return
        released = node.release(releases)            
        
    def reset(self):
        for n in self.all_nodes:
            n.reset()

    def next_resource(self, job) -> HeterogeneousNode:
        for node in self.all_nodes:
            if node.free_procs >= job.request_number_of_processors:
                return node
        return None

# Cluster = SimpleCluster
Cluster = HeterogeneousCluster



def load_cluster(path):
    with open(path) as in_f:
        data = json.load(in_f)
    return data['clusters'][0]


if __name__ == '__main__':
    print('Loading resources...')
    load = HeterogeneousCluster('data/cluster_x4_64procs.json')
    print(load)
    print('Finished loading the resources...')