'''
Created on Nov 09, 2020

@author: Euncheon Lim @ Chosun University
'''
import sys

def _get_available_GPUs(queue, including_gpus=None, excluding_gpus=None, memory_per_process=7000000000):
    import os
    import warnings
    warnings.filterwarnings(action="ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
#         tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=22000)])
        tf.config.experimental.set_memory_growth(gpu, True)
        
    from tensorflow.python.client import device_lib
    
    if None is including_gpus:
        including_gpus = []
        for a_device in device_lib.list_local_devices():
            if "GPU" != a_device.device_type:
                continue
            gpu_id = a_device.name.replace("/device:GPU:", "")
            including_gpus.append(gpu_id)
    else:
        including_gpus = including_gpus.split(",")
    
    if None is excluding_gpus:
        excluding_gpus = []
    else:
        excluding_gpus = excluding_gpus.split(",")
    
    gpu_dict = {}
    min_memory = sys.maxsize 
    for a_device in device_lib.list_local_devices():
        if "GPU" != a_device.device_type:
            continue
        gpu_id = a_device.name.replace("/device:GPU:", "")
        if gpu_id in excluding_gpus or gpu_id not in including_gpus:
            continue
        # print(a_device.name, a_device.device_type, a_device.memory_limit)
        if a_device.memory_limit > memory_per_process:
            gpu_dict[gpu_id] = a_device.memory_limit
            if min_memory > a_device.memory_limit:
                min_memory = a_device.memory_limit
    n_chunks = min_memory // memory_per_process
    gpus = list(gpu_dict.keys())
    
    queue.put((gpus, n_chunks))

def get_available_GPUs(including_gpus=None, excluding_gpus=None, memory_per_process=7000000000):
    import multiprocessing
    q = multiprocessing.Queue()
    arg_dict = {
        "including_gpus": including_gpus,
        "excluding_gpus": excluding_gpus,
        "memory_per_process": memory_per_process
        }
    
    # we fork a separate process encapsulating _get_available_GPUs function() because the function
    # consumes some amount of memory (about 300 Mb) for each GPU and does not release the memory.
    a_process_gpu = multiprocessing.Process(target=_get_available_GPUs, args=[q], kwargs=arg_dict)
    a_process_gpu.start()
    gpus, n_chunks = q.get()
    a_process_gpu.join()
    return gpus, n_chunks

def get_available_CPUs():
    import multiprocessing
    return multiprocessing.cpu_count()

if __name__ == '__main__':
    pass