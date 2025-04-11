

# --- run.py ---
from mpi4py import MPI
from utils import log
import sys
from chess_gui import main
# # Check if GUI mode is enabled
# gui_mode = "--gui" in sys.argv

# if gui_mode:
   
# else:
#     from master import master_process
#     from worker import worker_process

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

log("Run process started")

# if gui_mode:
    # GUI mode will handle master/worker division internally
main()
# else:
#     # Original console mode
#     if rank == 0:
#         log("Initializing master process")
#         master_process()
#     else:
#         log("Initializing worker process")
#         worker_process()