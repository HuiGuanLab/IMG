import math
class CosineUp(object):
    def __init__(self,num_loops):
        self._num_loops = num_loops

    def get_value(self, epoch):
        if epoch > self._num_loops:
            return 1
        value = math.cos(((epoch/ self._num_loops)-1)* math.pi )+1
        return 0.5*value

class LogUp(object):
    def __init__(self,num_loops):
        self._num_loops = num_loops

    def get_value(self, epoch):
        if epoch > self._num_loops:
            return 1
        value = math.cos((0.5*(epoch/ self._num_loops)-0.5)* math.pi )
        return value

class ExpUp(object):
    def __init__(self,num_loops):
        self._num_loops = num_loops

    def get_value(self, epoch):
        if epoch > self._num_loops:
            return 1
        value = math.cos((0.5*(epoch/ self._num_loops)-1)* math.pi )+1
        return value

class LinearUp(object):
    def __init__(self,num_loops):
        self._num_loops = num_loops

    def get_value(self, epoch):
        if epoch > self._num_loops:
            return 1
        value = epoch/ self._num_loops
        return value