import math
import numpy as np


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        '''
        :param schedule_timesteps: total timesteps
        :param final_p:
        :param initial_p:
        '''
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)           # 0 ~ 1
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ExpSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        '''
        :param schedule_timesteps: total timesteps
        :param final_p:
        :param initial_p:
        '''
        self.schedule_decay = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = math.exp( -1. * t / self.schedule_decay)            # e^(0 ~ -1000) ~= 1 ~ 0
        return self.final_p + fraction * (self.initial_p - self.final_p)