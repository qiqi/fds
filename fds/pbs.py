import os
import sys
import subprocess

def get_hostname():
    p = subprocess.Popen('hostname', stdout=subprocess.PIPE)
    output, _ = p.communicate()
    return output.decode().strip()

class grab_from_PBS_NODEFILE:
    '''
    Grab num_procs processes to launch MPI job on a subset of allocation.
    Needs a lock and a shared dictionary to prevent concurrent IO
    Remember to call release when MPI job finishes.
    '''
    def __init__(self, num_procs, lock_and_dict, exclude_this_node=False):
        self.lock, self.dict = lock_and_dict
        self.grab(num_procs, exclude_this_node)

    def grab(self, num_procs, exclude_this_node=False):
        with self.lock:
            if 'available_nodes' not in self.dict:
                available_nodes = open(os.environ['PBS_NODEFILE']).readlines()
                if exclude_this_node:
                    hostname = get_hostname()
                    print('Excluding {0} from {1} ranks'.format(
                        hostname, len(available_nodes)))
                    available_nodes = list(filter(
                        lambda n: n.strip() not in hostname, available_nodes))
                    print('Now {0} ranks left'.format(len(available_nodes)))
                    sys.stdout.flush()
            else:
                available_nodes = self.dict['available_nodes']
            if len(available_nodes) < num_procs:
                msg = 'Trying to grab {0} processes from {1}'
                raise ValueError(msg.format(num_procs, len(available_nodes)))
            self.grabbed_nodes = available_nodes[:num_procs]
            self.dict['available_nodes'] = available_nodes[num_procs:]

    def release(self):
        with self.lock:
            self.dict['available_nodes'] += self.grabbed_nodes

    def write_to_sub_nodefile(self, filename):
        with open(filename, 'wt') as f:
            f.writelines(self.grabbed_nodes)
