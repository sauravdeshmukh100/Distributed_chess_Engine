import pickle
import os
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
log_file = open(f"rank_{rank}.txt", "a")

DEBUG = False  # Toggle this to enable/disable debug logs

def log(msg):
    print(msg, file=log_file, flush=True)

def serialize(obj):
    if DEBUG:
        log(f"Serializing object: {obj}")
    return pickle.dumps(obj)

def deserialize(obj_bytes):
    obj = pickle.loads(obj_bytes)
    if DEBUG:
        log(f"Deserialized object: {obj}")
    return obj
