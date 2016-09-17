import os
import sys
import subprocess
import copy

class CobaltManager:
    def __init__(self, shape, nodesPerBlock=512):
        self.partition = os.environ['COBALT_PARTNAME']
        self.shape = shape
        self.blocks = self.get_blocks(nodesPerBlock)
        self.corners = {}
        for block in self.blocks:
            self.corners[block] = self.get_corners(block)
        self.interprocess = None

    def get_blocks(self, nodesPerBlock):
        p = subprocess.Popen(['get-bootable-blocks','--size', str(nodesPerBlock), self.partition],
                              stdout=subprocess.PIPE)
        available_blocks = p.communicate()[0]
        if p.returncode:
            raise Exception('get-bootable-blocks failed')
        available_blocks = available_blocks.decode().strip().split('\n')
        if len(available_blocks[0]) == 0:
            raise Exception('get-bootable-blocks returned nothing')
        return available_blocks

    def get_corners(self, blockName):
        p = subprocess.Popen(['/soft/cobalt/bgq_hardware_mapper/get-corners.py', blockName, self.shape],

                              stdout=subprocess.PIPE)
        corners = p.communicate()[0]
        corners = corners.decode().strip().split('\n')
        if p.returncode:
            raise Exception('get-corners failed')
        return corners

    def get_alloc(self):
        blockName = self.partition
        if self.interprocess:
            while 1:
                with self.interprocess[0]:
                    if 'available_corners' not in self.interprocess[1]:
                        self.interprocess[1]['available_corners'] = copy.deepcopy(self.corners)
                    corners = self.interprocess[1]['available_corners']
                    for block in corners:
                        if len(corners[block]) > 0:
                            corner = corners[block][0]
                            corners[block] = corners[block][1:]
                            self.interprocess[1]['available_corners']= corners
                            return block, corner
        else:
            return self.blocks[0], self.corners[self.blocks[0]][0]

    def free_alloc(self, alloc):
        block, corner = alloc
        if self.interprocess:
            with self.interprocess[0]:
                corners = self.interprocess[1]['available_corners']
                corners[block].append(corner)
                self.interprocess[1]['available_corners'] = corners

    def boot_blocks(self):
        ps = []
        for block in self.blocks:
            ps.append(subprocess.Popen(['boot-block','--block',block]))
        for p in ps:
            p.wait()
        for p in ps:
            if p.returncode:
                raise Exception('booting failed')

    def free_blocks(self):
        ps = []
        for block in self.blocks:
            ps.append(subprocess.Popen(['boot-block','--block',block,'--free']))
        for p in ps:
            p.wait()
        for p in ps:
            if p.returncode:
                raise Exception('freeing failed')

