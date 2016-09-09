import os
import sys
import subprocess

class CobaltManager:
    def __init__(self, shape, runs):
        self.partition = os.environ['COBALT_PARTNAME']
        eval_shap = eval(shape.replace('x','*'))
        nodes = eval_shape*runs
        self.blocks = self.get_available_blocks(nodes)
        self.shape = shape
        self.interprocess = None

    def get_available_blocks(self, nodesPerBlock):
        p = subprocess.Popen(['get-bootable-blocks','--size', str(nodesPerBlock), self.partition],
                              stdout=subprocess.PIPE)
        available_blocks = p.communicate()[0]
        return available_blocks.decode().strip().split('\n')

    def list_corners(self, blockName):
        p = subprocess.Popen(['/soft/cobalt/bgq_hardware_mapper/get-corners.py', blockName, self.shape],

                              stdout=subprocess.PIPE)
        corners = p.communicate()[0]
        return corners

    def get_corner(self):
        blockName = self.partition
        if self.interprocess:
            with self.interprocess[0]:
                if 'available_corners' not in self.interprocess[1]:
                    available_corners_tmp = self.list_corners(blockName)
                    self.interprocess[1]['available_corners'] = available_corners_tmp.decode().strip().split('\n')
                available_corners = self.interprocess[1]['available_corners']

                if len(available_corners) > 0:
                    grabbed_corner = available_corners[0]
                    self.interprocess[1]['available_corners'] = available_corners[1:]
                    return grabbed_corner
                else:
                    return -1
        else:
            return self.list_corners(blockName)[0]

    def free_corner(self, corner):
        if self.interprocessor:
            with self.interprocess[0]:
                self.interprocess[1]['available_corners'].append(corner)

    def boot_blocks(self):
        for block in self.blocks:
            subprocess.call(['boot-block','--block',block])

    def free_blocks(self):
        for block in self.blocks:
            subprocess.call(['boot-block','--block',block,'--free'])

