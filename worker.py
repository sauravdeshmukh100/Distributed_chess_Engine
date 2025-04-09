# --- worker.py ---
from mpi4py import MPI
import chess
from chess_engine import minimax
from utils import deserialize, serialize, log

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def worker_process():
    log("Worker process started")
    while True:
        data = comm.recv(source=0, tag=MPI.ANY_TAG)
        log(f"Received task from master: {data}")
        if data == "STOP":
            log("Received STOP signal. Exiting worker.")
            break

        board_fen, depth = deserialize(data)
        board = chess.Board(board_fen)
        score, move = minimax(board, depth, -float('inf'), float('inf'), board.turn)
        log(f"Processed board. Score: {score}, Move: {move}")
        comm.send(serialize((score, move.uci() if move else None)), dest=0)


