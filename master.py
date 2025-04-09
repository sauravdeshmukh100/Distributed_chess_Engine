# --- master.py ---
from mpi4py import MPI
import chess
from utils import serialize, deserialize, log

comm = MPI.COMM_WORLD
size = comm.Get_size()


def master_process():
    board = chess.Board()
    depth = 3

    legal_moves = list(board.legal_moves)
    log(f"Distributing {len(legal_moves)} moves to {size-1} workers")

    for i, move in enumerate(legal_moves):
        board.push(move)
        task = serialize((board.fen(), depth - 1))
        dest = (i % (size - 1)) + 1
        comm.send(task, dest=dest)
        log(f"Sent task to worker {dest}: {move}")
        board.pop()

    results = []
    for i in range(len(legal_moves)):
        result = comm.recv(source=MPI.ANY_SOURCE)
        log(f"Received result: {result}")
        results.append(deserialize(result))

    best_score = max(results, key=lambda x: x[0])
    log(f"Best move found: {best_score}")

    for i in range(1, size):
        comm.send("STOP", dest=i)
        log(f"Sent STOP to worker {i}")