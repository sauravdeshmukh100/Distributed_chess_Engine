import chess
import time
from mpi4py import MPI
from collections import deque
import random
import traceback
from collections import deque

import time  
from utils import log, serialize, deserialize

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def minimax_wrapper(board, depth):
    from chess_engine import minimax
    
    # log("Starting minimax search")
    score, move = minimax(board, depth, -float('inf'), float('inf'), board.turn)
    log(f"Minimax search completed: score = {score}, move = {move}")
    return score, move


def distribute_and_collect(board, max_depth=4, time_limit=5):
    """
    Modified for iterative deepening with time management
    Preserves original MPI communication patterns
    """
    from chess_engine import evaluate_board
    
    comm = MPI.COMM_WORLD
    worker_count = comm.Get_size() - 1
    start_time = time.time()
    best_result = (float('-inf'), None) if board.turn else (float('inf'), None)
    
    # Iterative deepening loop
    for current_depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            # log(f"Time limit reached at depth {current_depth-1}", )
            break
            
        # log(f"Starting depth {current_depth}", )
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return (0, None)
        
        move_queue = deque(legal_moves)
        results = []
        move_mapping = {}
        active_workers = set()

        # Phase 1: Initial distribution for current depth
        for worker_rank in range(1, comm.Get_size()):
            if move_queue:
                move = move_queue.popleft()
                move_mapping[worker_rank] = move
                board_copy = board.copy()
                board_copy.push(move)
                task = serialize((
            board_copy.fen(), 
            current_depth - 1,
            time_limit - (time.time() - start_time)  # Remaining time
        ))  # Now 3 elements # Note: depth-1 for recursive search
                comm.send(task, dest=worker_rank, tag=1)
                active_workers.add(worker_rank)

        # Phase 2: Dynamic task handling with depth awareness
        while (move_queue or active_workers) and (time.time() - start_time < time_limit):
            if active_workers:
                status = MPI.Status()
                result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
                worker_rank = status.source
                # SAFETY CHECK: Only process if worker is still active
                if worker_rank in active_workers:
                    original_move = move_mapping[worker_rank]
                    score, _ = deserialize(result)
                    results.append((score, original_move))
                    active_workers.remove(worker_rank)  # Now safe

                # Assign new task if time remains
                if move_queue and (time.time() - start_time < time_limit):
                    move = move_queue.popleft()
                    move_mapping[worker_rank] = move
                    board_copy = board.copy()
                    board_copy.push(move)
                    task = serialize((
                        board_copy.fen(), 
                        current_depth - 1,
                        time_limit - (time.time() - start_time)  # Remaining time
                    ))  # Now 3 elements
                    comm.send(task, dest=worker_rank, tag=1)
                    active_workers.add(worker_rank)

        # Result validation and error handling
        valid_results = []
        error_messages = []
        
        for score, move in results:
            if isinstance(score, (int, float)):
                valid_results.append((score, move))
            else:
                error_messages.append(f"Invalid result: {score} - {move}")
        
        if error_messages:
            log("Worker errors:\n" + "\n".join(error_messages))
        
        # Handle case with no valid results
        if not valid_results:
            # log("No valid results! Using first legal move")
            return (0, next(iter(board.legal_moves), None))
        
        # Find best from valid results
        if board.turn:
            current_best = max(valid_results, key=lambda x: x[0])
        else:
            current_best = min(valid_results, key=lambda x: x[0])
        
        if (board.turn and current_best[0] > best_result[0]) or \
           (not board.turn and current_best[0] < best_result[0]):
            best_result = current_best
            # log(f"New best at depth {current_depth}: {best_result[1].uci()} ({best_result[0]})")

        # Cleanup workers between depth iterations
        for worker_rank in range(1, comm.Get_size()):
            comm.send("NO_MORE_TASKS", dest=worker_rank, tag=1)

    return best_result  



def ai_move_thread(board, config, callback):
    """Enhanced with proper time management and error handling"""
    try:
        start_time = time.time()
        best_move = None
        remaining_time = config['time_limit']

        def time_left():
            return max(0, config['time_limit'] - (time.time() - start_time))

        if config['use_distributed']:
            # Distributed search with MPI
            best_score, best_move = distribute_and_collect(
                board,
                max_depth=config['max_depth'],
                time_limit=remaining_time
            )
        else:
            # Local iterative deepening with time per depth
            for current_depth in range(1, config['max_depth'] + 1):
                if time_left() <= 0:
                    break
                
                try:
                    score, move = minimax_wrapper(
                        board, 
                        current_depth,
                        time_limit=time_left() * 0.8  # Reserve 20% for next depth
                    )
                    if move:
                        best_move = move
                        log(f"Depth {current_depth} best: {move.uci()} ({score})")
                except TimeoutError:
                    log(f"Depth {current_depth} timed out")
                    break

        callback(best_move if best_move else random.choice(list(board.legal_moves)) or None)
        
    except Exception as e:
        log(f"AI Error: {str(e)}\n{traceback.format_exc()}")
        callback(None)


def stop_workers():
    """Send stop signal to all worker processes"""
    for i in range(1, size):
        comm.send("STOP", dest=i)