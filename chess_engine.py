import chess
import chess.engine
from mpi4py import MPI
import pickle
from utils import log  # Assuming utils.py has a log function implemented

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def evaluate_board(board):
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    score = 0
    for piece_type in piece_values:
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        score += white_count * piece_values[piece_type]
        score -= black_count * piece_values[piece_type]
    log(f"Evaluated board score: {score}")
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    log(f"Minimax call: depth={depth}, alpha={alpha}, beta={beta}, maximizing={maximizing_player}")
    if depth == 0 or board.is_game_over():
        score = evaluate_board(board)
        log(f"Reached leaf node or game over. Score: {score}")
        return score, None

    legal_moves = list(board.legal_moves)
    best_move = None

    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        log(f"Maximizing: Best eval = {max_eval}, Move = {best_move}")
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        log(f"Minimizing: Best eval = {min_eval}, Move = {best_move}")
        return min_eval, best_move

def distribute_moves_and_collect(board, depth):
    legal_moves = list(board.legal_moves)
    chunk_size = (len(legal_moves) + size - 2) // (size - 1)

    log(f"Distributing {len(legal_moves)} legal moves among {size - 1} workers")
    for i in range(1, size):
        start = (i - 1) * chunk_size
        end = start + chunk_size
        data = pickle.dumps((board, legal_moves[start:end], depth))
        comm.send(data, dest=i, tag=0)
        log(f"Sent {end-start} moves to worker {i}")

    best_eval = float('-inf')
    best_move = None
    for i in range(1, size):
        eval, move = comm.recv(source=i, tag=1)
        log(f"Received from worker {i}: eval = {eval}, move = {move}")
        if eval > best_eval:
            best_eval = eval
            best_move = move
    log(f"Best move after collection: {best_move}, eval: {best_eval}")
    return best_move

def worker_node():
    while True:
        data = comm.recv(source=0, tag=0)
        board, moves, depth = pickle.loads(data)

        log(f"Worker received {len(moves)} moves to evaluate at depth {depth}")
        best_eval = float('-inf')
        best_move = None
        for move in moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, float('-inf'), float('inf'), False)
            board.pop()
            if eval > best_eval:
                best_eval = eval
                best_move = move
        log(f"Worker sending back: eval = {best_eval}, move = {best_move}")
        comm.send((best_eval, best_move), dest=0, tag=1)

def play_game():
    board = chess.Board()

    while not board.is_game_over():
        print("\nCurrent board:\n")
        print(board)
        move_uci = input("\nYour move (e.g., 'e2e4', 'g1f3'): ").strip()

        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                print("Illegal move! Try again.")
                continue
            board.push(move)
        except:
            print("Invalid input! Use UCI format (e.g., 'e2e4').")
            continue

        if board.is_game_over():
            break

        log("Master distributing moves for AI response")
        best_move = distribute_moves_and_collect(board.copy(), depth=3)
        print(f"\nAI plays: {best_move.uci()}")
        board.push(best_move)

    print("\nFinal position:")
    print(board)
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    if rank == 0:
        log("Master node started")
        play_game()
    else:
        log("Worker node started")
        worker_node()
