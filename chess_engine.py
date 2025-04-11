import chess
import chess.engine
from mpi4py import MPI
import pickle
import random
from utils import log  # Assuming utils.py has a log function implemented

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Piece values (in centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-square tables (adapted from ChienKoNgu.py but formatted for python-chess)
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5,  5,  5,  5,  5,-10,
    -10,  0,  5,  0,  0,  5,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
    0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    5, 10, 10, 10, 10, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_MIDDLE_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

KING_END_TABLE = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
]




def evaluate_board(board):
    """
    Enhanced evaluation function that combines elements from both approaches.
    
    Features:
    1. Material counting
    2. Piece-square tables
    3. Mobility evaluation
    4. Pawn structure analysis
    5. King safety
    6. Game phase detection
    
    Returns a score from White's perspective.
    """
    # Handle game-over conditions
    if board.is_checkmate():
        return -10000 if board.turn else 10000
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0
    
    # --- 1. Material and Piece-Square Tables ---
    piece_tables = {
        chess.PAWN: PAWN_TABLE,
        chess.KNIGHT: KNIGHT_TABLE,
        chess.BISHOP: BISHOP_TABLE,
        chess.ROOK: ROOK_TABLE,
        chess.QUEEN: QUEEN_TABLE,
        chess.KING: KING_MIDDLE_TABLE  # Default to middle game
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Material score
            value = PIECE_VALUES[piece.piece_type] * (1 if piece.color else -1)
            score += value
            
            # Position score - needs to be mirrored for black pieces
            if piece.piece_type != chess.KING:  # King has special handling for game phase
                position_table = piece_tables[piece.piece_type]
                # For black pieces, mirror the square vertically
                position_idx = square if piece.color else chess.square_mirror(square)
                position_value = position_table[position_idx]
                score += position_value * (1 if piece.color else -1)
    
    # --- 2. Mobility ---
    # Count legal moves for both sides (this correlates with piece development and control)
    board_copy = board.copy()
    board_copy.turn = chess.WHITE
    white_mobility = len(list(board_copy.legal_moves))
    
    board_copy.turn = chess.BLACK
    black_mobility = len(list(board_copy.legal_moves))
    
    mobility_score = (white_mobility - black_mobility) * 8  # 8 points per extra move
    score += mobility_score
    
    # --- 3. Pawn Structure ---
    # Penalize doubled pawns
    files = [0] * 8
    for square in board.pieces(chess.PAWN, chess.WHITE):
        files[chess.square_file(square)] += 1
    white_doubled = sum(f - 1 for f in files if f > 1)
    
    files = [0] * 8
    for square in board.pieces(chess.PAWN, chess.BLACK):
        files[chess.square_file(square)] += 1
    black_doubled = sum(f - 1 for f in files if f > 1)
    
    score -= white_doubled * 15  # Penalty for doubled pawns
    score += black_doubled * 15
    
    # --- 4. King Safety ---
    # Simple approach: count pawns near the king
    def king_shield_count(king_square, pawns, color):
        """Count pawns that shield the king"""
        if king_square is None:
            return 0
            
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Look for pawns in front of the king (and diagonally in front)
        shield_count = 0
        direction = 1 if color else -1  # White pawns move up, black pawns move down
        
        # Check three files: king's file and adjacent files
        for f in range(max(0, king_file - 1), min(8, king_file + 2)):
            # Check up to two ranks in front of king
            for r in range(1, 3):
                check_rank = king_rank + r * direction
                if 0 <= check_rank < 8:
                    check_square = chess.square(f, check_rank)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        shield_count += 1
        
        return shield_count
    
    # Get king positions
    white_king_square = board.king(chess.WHITE) if any(board.pieces(chess.KING, chess.WHITE)) else None
    black_king_square = board.king(chess.BLACK) if any(board.pieces(chess.KING, chess.BLACK)) else None
    
    # Calculate king safety
    if white_king_square is not None:
        white_king_shield = king_shield_count(white_king_square, board.pieces(chess.PAWN, chess.WHITE), chess.WHITE)
        score += white_king_shield * 25  # Bonus for pawn shield
    
    if black_king_square is not None:
        black_king_shield = king_shield_count(black_king_square, board.pieces(chess.PAWN, chess.BLACK), chess.BLACK)
        score -= black_king_shield * 25
    
    # --- 5. Game Phase Detection and King Positioning ---
    # Count material to determine game phase
    white_material = sum(len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt] 
                      for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    black_material = sum(len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt] 
                      for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    
    total_material = white_material + black_material
    
    # Full material would be around 7800 (starting position)
    # Endgame transition starts around 3000
    endgame_factor = 1.0 - min(1.0, max(0.0, (total_material - 3000) / 4800))
    
    # Apply king position scores based on game phase
    if white_king_square is not None:
        # Remove the middle game king position score first
        mid_score = KING_MIDDLE_TABLE[white_king_square]
        score -= mid_score  # Remove middle game score
        
        # Apply weighted combination of middle and endgame tables
        mid_score = KING_MIDDLE_TABLE[white_king_square] * (1 - endgame_factor)
        end_score = KING_END_TABLE[white_king_square] * endgame_factor
        score += (mid_score + end_score)
    
    if black_king_square is not None:
        # For black pieces, mirror the square vertically
        mirror_square = chess.square_mirror(black_king_square)
        
        # Remove the middle game king position score first
        mid_score = KING_MIDDLE_TABLE[mirror_square]
        score += mid_score  # Remove middle game score (add because it's for black)
        
        # Apply weighted combination of middle and endgame tables
        mid_score = KING_MIDDLE_TABLE[mirror_square] * (1 - endgame_factor)
        end_score = KING_END_TABLE[mirror_square] * endgame_factor
        score -= (mid_score + end_score)
    
    # Add a small random factor to avoid repetition (1-2 points)
    score += random.randint(-2, 2)
    
    # Log the evaluation
    log(f"Evaluated board score: {score}")
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax search with alpha-beta pruning.
    Returns the best score and move from the current position.
    """
    log(f"Minimax call: depth={depth}, alpha={alpha}, beta={beta}, maximizing={maximizing_player}")
    
    # Base case: depth reached or game over
    if depth == 0 or board.is_game_over():
        score = evaluate_board(board)
        log(f"Reached leaf node or game over. Score: {score}")
        return score, None
    
    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)  # Randomize move order for better pruning
    best_move = None
    
    if maximizing_player:  # White's turn
        max_score = float('-inf')
        for move in legal_moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            
            if score > max_score:
                max_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if beta <= alpha:
                break  # Beta cutoff
        
        log(f"Maximizing: Best score = {max_score}, Move = {best_move}")
        return max_score, best_move
    
    else:  # Black's turn
        min_score = float('inf')
        for move in legal_moves:
            board.push(move)
            score, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            
            if score < min_score:
                min_score = score
                best_move = move
            
            beta = min(beta, score)
            if beta <= alpha:
                break  # Alpha cutoff
        
        log(f"Minimizing: Best score = {min_score}, Move = {best_move}")
        return min_score, best_move

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
        print("depth=", depth)
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
        best_move = distribute_moves_and_collect(board.copy(), depth=20)
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
