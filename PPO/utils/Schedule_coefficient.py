import numpy as np
import torch
import math

# ** The coefficient decreases gradually as the number of times increases.**


def LinearSche(initial_coeff, final_coeff, now_num_time, num_time):
    ep_ratio = 1 - (now_num_time/num_time)
    coeff_now = final_coeff + (initial_coeff - final_coeff) * max(0, ep_ratio)
    return coeff_now


def ExpSche(initial_coeff, final_coeff, now_num_time, num_time):
    ep_ratio = math.exp(- now_num_time / num_time)
    coeff_now = final_coeff + (initial_coeff - final_coeff) * ep_ratio
    return coeff_now

