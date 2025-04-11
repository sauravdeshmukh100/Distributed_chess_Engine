# --- worker.py ---
from mpi4py import MPI
import chess
from chess_engine import minimax
from utils import deserialize, serialize, log

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def worker_process():
    log(f"Worker {rank} process started")
    while True:
        # Receive task 
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        task_tag = status.tag
        
        if task == "STOP":
            log(f"Worker {rank} received stop signal")
            break
            
        if task == "NO_MORE_TASKS":
            log(f"Worker {rank} has no more tasks, waiting for next round")
            continue
        
        try:
            log(f"Worker {rank} processing task")
            board_fen, depth = deserialize(task)
            board = chess.Board(board_fen)
            score, move = minimax(board, depth, -float('inf'), float('inf'), board.turn)
            
            # Send result back
            move_uci = move.uci() if move else None
            log(f"Worker {rank} sending result: {score}, {move_uci}")
            comm.send(serialize((score, move_uci)), dest=0, tag=2)
        except Exception as e:
            log(f"Worker {rank} error: {e}")
            # Send error result
            comm.send(serialize((-9999, None)), dest=0, tag=2)
