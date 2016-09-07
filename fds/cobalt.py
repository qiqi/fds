import os
import sys
import subprocess

class CobaltManager:
    def __init__(self, nodesPerBlock):
        self.partition = os.environ['COBALT_PARTNAME']
        self.blocks = self.get_available_blocks(nodesPerBlock)

    def get_available_blocks(self, nodesPerBlock):
        p = subprocess.Popen(['get-bootable-blocks','--size', str(nodesPerBlock), self.partition],
                              stdout=subprocess.PIPE)
        available_blocks = p.communicate()[0]
        return available_blocks.decode().strip().split('\n')

    def get_corner(self, subBlockShape, interprocess):
        blockName = self.partition
        with interprocess[0]:
            if 'available_corners' not in interprocess[1]:
                p = subprocess.Popen(['/soft/cobalt/bgq_hardware_mapper/get-corners.py', blockName, subBlockShape],
                                  stdout=subprocess.PIPE)
                available_corners_tmp = p.communicate()[0]
                interprocess[1]['available_corners'] = available_corners_tmp.decode().strip().split('\n')
            available_corners = interprocess[1]['available_corners']

            if len(available_corners) > 0:
                grabbed_corner = available_corners[0]
                interprocess[1]['available_corners'] = available_corners[1:]
                return grabbed_corner
            else:
                return -1

    def free_corner(self, corner, interprocess):
        with interprocess[0]:
            interprocess[1]['available_corners'].append(corner)

    def boot_blocks(self):
        for block in self.blocks:
            subprocess.call(['boot-block','--block',block])

    def free_blocks(self, block):
        for block in self.blocks:
            subprocess.call(['boot-block','--block',block,'--free'])

