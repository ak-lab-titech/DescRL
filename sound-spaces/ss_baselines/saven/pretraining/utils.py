
import subprocess

import numpy as np


def get_gpu_memory_map(max_memory_used_percent=0.8):
    """Get the current gpu usage if used memory is less than max_memory_used_percent

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    memory_used = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    memory_total = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.total',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    memory_used = np.array([int(x) for x in memory_used.strip().split('\n')])
    memory_total = np.array([int(x) for x in memory_total.strip().split('\n')])

    memory_used_percent = memory_used/memory_total

    gpu_memory_map = {}
    for gpu_id in range(len(memory_used)):
        if memory_used_percent[gpu_id] < max_memory_used_percent:
            gpu_memory_map[gpu_id] = memory_used[gpu_id]

    return gpu_memory_map

