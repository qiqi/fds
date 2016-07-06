import os
import sys
import subprocess

def get_hostname():
    p = subprocess.Popen('hostname', stdout=subprocess.PIPE)
    output, _ = p.communicate()
    return output.decode().strip()

def get_slurm_nodes():
    available_nodes = os.environ['SLURM_NODELIST']
    p = subprocess.Popen(['scontrol', 'show', 'hostname', available_nodes],
                         stdout=subprocess.PIPE)
    available_nodes, _ = p.communicate()
    return available_nodes.decode().strip().split('\n')

class grab_from_SLURM_NODELIST:
    '''
    Grab num_nodes processes to launch MPI job on a subset of allocation.
    Needs a lock and a shared dictionary to prevent concurrent IO
    Remember to call release when MPI job finishes.
    '''
    def __init__(self, num_nodes, lock_and_dict, exclude_this_node=False):
        self.lock, self.dict = lock_and_dict
        self.grab(num_nodes, exclude_this_node)

    def grab(self, num_nodes, exclude_this_node=False):
        with self.lock:
            if 'available_nodes' not in self.dict:
                available_nodes = get_slurm_nodes()
                if exclude_this_node:
                    hostname = get_hostname()
                    print('Excluding {0} from {1} nodes'.format(
                        hostname, len(available_nodes)))
                    available_nodes = list(filter(
                        lambda n: n.strip() not in hostname, available_nodes))
                    print('Now {0} nodes left'.format(len(available_nodes)))
                    sys.stdout.flush()
            else:
                available_nodes = self.dict['available_nodes']
            if len(available_nodes) < num_nodes:
                msg = 'Trying to grab {0} nodes from {1}'
                raise ValueError(msg.format(num_nodes, len(available_nodes)))
            self.grabbed_nodes = available_nodes[:num_nodes]
            self.dict['available_nodes'] = available_nodes[num_nodes:]

    def release(self):
        with self.lock:
            self.dict['available_nodes'] += self.grabbed_nodes

    def write_to_sub_nodefile(self, filename):
        with open(filename, 'wt') as f:
            f.writelines(self.grabbed_nodes)
