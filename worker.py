# --- worker.py ---
from mpi4py import MPI
import chess
from chess_engine import minimax
from utils import deserialize, serialize, log
import time
import os
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def worker_process():
    log(f"Worker {rank} started with PID {os.getpid()}")
    last_best_move = None
    
    while True:
        # Receive task with timeout
        status = MPI.Status()
        try:
            task = comm.recv(
                source=0, 
                tag=MPI.ANY_TAG, 
                status=status
            )
        except Exception as e:
            log(f"Worker {rank} receive error: {e}")
            continue
            
        task_tag = status.tag
        
        # Termination handling
        if task == "STOP":
            log(f"Worker {rank} received STOP")
            break
        if task == "NO_MORE_TASKS":
            last_best_move = None  # Reset between depth iterations
            continue
            
        try:
            # Parse task
            board_fen, depth, time_limit = deserialize(task)
            start_time = time.time()
            
            board = chess.Board(board_fen)
            log(f"Worker {rank} processing depth {depth} task")
            
            # Run search
            score, move = minimax(
                board, depth, 
                -float('inf'), float('inf'),
                board.turn,
                start_time,
                time_limit,
                last_best_move
            )
            
            # Update last best move for move ordering
            if move:
                last_best_move = move
                
            # Send result
            comm.send(
                serialize((score, move.uci() if move else None)),
                dest=0,
                tag=2
            )
            
        except TimeoutError:
            log(f"Worker {rank} timeout at depth {depth}")
            comm.send(serialize(("TIMEOUT", None)), dest=0, tag=2)
        except Exception as e:
            log(f"Worker {rank} error: {str(e)}")
            comm.send(serialize(("ERROR", str(e))), dest=0, tag=2)
