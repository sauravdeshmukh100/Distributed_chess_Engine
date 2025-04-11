# --- chess_gui.py ---
import pygame
import chess
import sys
import os
from mpi4py import MPI
import threading
from utils import log, serialize, deserialize

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# GUI Constants
WIDTH, HEIGHT = 640, 640
BOARD_SIZE = 480
SQUARE_SIZE = BOARD_SIZE // 8
BOARD_OFFSET_X = (WIDTH - BOARD_SIZE) // 2
BOARD_OFFSET_Y = (HEIGHT - BOARD_SIZE) // 2 + 20
INFO_HEIGHT = 80
PIECE_IMAGES = {}

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_SQUARE = (118, 150, 86)  # Green
LIGHT_SQUARE = (238, 238, 210)  # Cream
HIGHLIGHT = (255, 255, 0, 128)  # Yellow with transparency
MOVE_HIGHLIGHT = (100, 100, 255, 128)  # Blue with transparency
DIALOG_BG = (240, 240, 240)
DIALOG_BORDER = (100, 100, 100)
BUTTON_HOVER = (200, 200, 200)
BUTTON_ACTIVE = (180, 180, 180)

def load_piece_images():
    pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    for piece in pieces:
        # Load white pieces
        img_path = os.path.join('chess_pieces', f'w{piece}.png')
        if os.path.exists(img_path):
            PIECE_IMAGES[piece.upper()] = pygame.image.load(img_path)
        else:
            # Fallback to text rendering if images not found
            PIECE_IMAGES[piece.upper()] = None
        
        # Load black pieces
        img_path = os.path.join('chess_pieces', f'b{piece}.png')
        if os.path.exists(img_path):
            PIECE_IMAGES[piece.lower()] = pygame.image.load(img_path)
        else:
            PIECE_IMAGES[piece.lower()] = None

def square_to_coords(square):
    """Convert chess square to pixel coordinates"""
    file_idx = chess.square_file(square)
    rank_idx = 7 - chess.square_rank(square)
    return (
        BOARD_OFFSET_X + file_idx * SQUARE_SIZE,
        BOARD_OFFSET_Y + rank_idx * SQUARE_SIZE
    )

def coords_to_square(x, y):
    """Convert pixel coordinates to chess square"""
    if (x < BOARD_OFFSET_X or x >= BOARD_OFFSET_X + BOARD_SIZE or
            y < BOARD_OFFSET_Y or y >= BOARD_OFFSET_Y + BOARD_SIZE):
        return None
    
    file_idx = (x - BOARD_OFFSET_X) // SQUARE_SIZE
    rank_idx = (y - BOARD_OFFSET_Y) // SQUARE_SIZE
    return chess.square(file_idx, 7 - rank_idx)

def draw_board(screen, board, selected_square=None, last_move=None):
    # Draw squares
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)
            x = BOARD_OFFSET_X + file * SQUARE_SIZE
            y = BOARD_OFFSET_Y + rank * SQUARE_SIZE
            color = LIGHT_SQUARE if (file + rank) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
            
            # Draw coordinates for the squares
            if rank == 7:
                font = pygame.font.SysFont('Arial', 12)
                text = font.render(chess.FILE_NAMES[file], True, BLACK if color == LIGHT_SQUARE else WHITE)
                screen.blit(text, (x + SQUARE_SIZE - 12, y + SQUARE_SIZE - 14))
            if file == 0:
                font = pygame.font.SysFont('Arial', 12)
                text = font.render(str(8 - rank), True, BLACK if color == LIGHT_SQUARE else WHITE)
                screen.blit(text, (x + 4, y + 4))
    
    # Highlight selected square
    if selected_square is not None:
        x, y = square_to_coords(selected_square)
        highlight = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        highlight.fill(HIGHLIGHT)
        screen.blit(highlight, (x, y))
        
        # Show legal moves from selected square
        for move in board.legal_moves:
            if move.from_square == selected_square:
                to_x, to_y = square_to_coords(move.to_square)
                move_highlight = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                move_highlight.fill(MOVE_HIGHLIGHT)
                screen.blit(move_highlight, (to_x, to_y))
    
    # Highlight last move
    if last_move:
        for square in [last_move.from_square, last_move.to_square]:
            x, y = square_to_coords(square)
            last_highlight = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            last_highlight.fill((255, 165, 0, 100))  # Orange
            screen.blit(last_highlight, (x, y))
    
    # Draw pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x, y = square_to_coords(square)
            piece_symbol = piece.symbol()
            
            if PIECE_IMAGES[piece_symbol] is not None:
                # Scale the image to fit the square
                scaled_image = pygame.transform.scale(
                    PIECE_IMAGES[piece_symbol], 
                    (SQUARE_SIZE, SQUARE_SIZE)
                )
                screen.blit(scaled_image, (x, y))
            else:
                # Fallback to text rendering
                font = pygame.font.SysFont('Arial', 40)
                text = font.render(piece_symbol, True, BLACK)
                text_x = x + (SQUARE_SIZE - text.get_width()) // 2
                text_y = y + (SQUARE_SIZE - text.get_height()) // 2
                screen.blit(text, (text_x, text_y))

def draw_info(screen, board, status_text):
    # Draw info panel at the top
    pygame.draw.rect(screen, (230, 230, 230), (0, 0, WIDTH, INFO_HEIGHT))
    pygame.draw.line(screen, BLACK, (0, INFO_HEIGHT), (WIDTH, INFO_HEIGHT), 2)
    
    # Draw turn indicator
    font = pygame.font.SysFont('Arial', 18)
    turn_text = f"{'White' if board.turn else 'Black'} to move"
    text = font.render(turn_text, True, BLACK)
    screen.blit(text, (20, 20))
    
    # Draw status text
    status_font = pygame.font.SysFont('Arial', 16)
    status_render = status_font.render(status_text, True, BLACK)
    screen.blit(status_render, (20, 50))
    
    # Draw game state
    if board.is_checkmate():
        state_text = "Checkmate!"
    elif board.is_stalemate():
        state_text = "Stalemate!"
    elif board.is_check():
        state_text = "Check!"
    else:
        state_text = ""
    
    if state_text:
        state_font = pygame.font.SysFont('Arial', 24)
        state_render = state_font.render(state_text, True, (255, 0, 0))
        screen.blit(state_render, (WIDTH - 150, 20))

def show_promotion_dialog(screen):
    """Show dialog for pawn promotion and return the chosen piece type"""
    # Create semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))  # Semi-transparent black
    screen.blit(overlay, (0, 0))
    
    # Create dialog box
    dialog_width, dialog_height = 350, 120
    dialog_x = (WIDTH - dialog_width) // 2
    dialog_y = (HEIGHT - dialog_height) // 2
    
    # Draw dialog background
    pygame.draw.rect(screen, DIALOG_BG, (dialog_x, dialog_y, dialog_width, dialog_height))
    pygame.draw.rect(screen, DIALOG_BORDER, (dialog_x, dialog_y, dialog_width, dialog_height), 2)
    
    # Draw dialog title
    font = pygame.font.SysFont('Arial', 18, bold=True)
    title = font.render("Promote Pawn To:", True, BLACK)
    title_x = dialog_x + (dialog_width - title.get_width()) // 2
    screen.blit(title, (title_x, dialog_y + 15))
    
    # Create piece buttons
    piece_types = [(chess.QUEEN, "Queen"), (chess.ROOK, "Rook"), 
                   (chess.BISHOP, "Bishop"), (chess.KNIGHT, "Knight")]
    buttons = []
    
    button_width = 70
    button_height = 35
    button_y = dialog_y + 50
    
    total_width = len(piece_types) * button_width + (len(piece_types) - 1) * 10  # 10px spacing
    start_x = dialog_x + (dialog_width - total_width) // 2
    
    for i, (piece_type, label) in enumerate(piece_types):
        button_x = start_x + i * (button_width + 10)
        buttons.append((pygame.Rect(button_x, button_y, button_width, button_height), piece_type, label))
    
    # Event loop for promotion dialog
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for button, piece_type, _ in buttons:
                    if button.collidepoint(mouse_pos):
                        return piece_type
            
        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        for button, piece_type, label in buttons:
            if button.collidepoint(mouse_pos):
                pygame.draw.rect(screen, BUTTON_HOVER, button)
            else:
                pygame.draw.rect(screen, DIALOG_BG, button)
            pygame.draw.rect(screen, DIALOG_BORDER, button, 1)
            
            font = pygame.font.SysFont('Arial', 14)
            text = font.render(label, True, BLACK)
            text_x = button.x + (button.width - text.get_width()) // 2
            text_y = button.y + (button.height - text.get_height()) // 2
            screen.blit(text, (text_x, text_y))
        
        pygame.display.flip()        

def minimax_wrapper(board, depth):
    from chess_engine import minimax
    
    log("Starting minimax search")
    score, move = minimax(board, depth, -float('inf'), float('inf'), board.turn)
    log(f"Minimax search completed: score = {score}, move = {move}")
    return score, move

def distribute_and_collect(board, depth):
    from chess_engine import evaluate_board
    
    legal_moves = list(board.legal_moves)
    log(f"Distributing {len(legal_moves)} moves to {size-1} workers")
    
    # Send tasks to workers
    for i, move in enumerate(legal_moves):
        board_copy = board.copy()
        board_copy.push(move)
        task = serialize((board_copy.fen(), depth - 1))
        dest = (i % (size - 1)) + 1
        comm.send(task, dest=dest)
        log(f"Sent task to worker {dest}: {move}")
    
    # Collect results
    results = []
    for i in range(len(legal_moves)):
        result = comm.recv(source=MPI.ANY_SOURCE)
        score, _ = deserialize(result)
        results.append((score, legal_moves[i]))
        log(f"Received result: score = {score}")
    
    # Find best move
    if board.turn:  # White's turn, maximize
        best_result = max(results, key=lambda x: x[0])
    else:  # Black's turn, minimize
        best_result = min(results, key=lambda x: x[0])
    
    log(f"Best move found: {best_result[1]} with score {best_result[0]}")
    return best_result

def ai_move_thread(board, depth, callback):
    try:
        if size > 1:
            score, move = distribute_and_collect(board, depth)
        else:
            score, move = minimax_wrapper(board, depth)
        
        callback(move)
    except Exception as e:
        log(f"Error in AI calculation: {e}")
        callback(None)


def show_difficulty_dialog(screen):
    """Show dialog for difficulty selection and return the chosen depth"""
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))

    dialog_width, dialog_height = 350, 150
    dialog_x = (WIDTH - dialog_width) // 2
    dialog_y = (HEIGHT - dialog_height) // 2

    pygame.draw.rect(screen, DIALOG_BG, (dialog_x, dialog_y, dialog_width, dialog_height))
    pygame.draw.rect(screen, DIALOG_BORDER, (dialog_x, dialog_y, dialog_width, dialog_height), 2)

    font = pygame.font.SysFont('Arial', 18, bold=True)
    title = font.render("Select AI Difficulty:", True, BLACK)
    title_x = dialog_x + (dialog_width - title.get_width()) // 2
    screen.blit(title, (title_x, dialog_y + 15))

    difficulties = [("Easy", 2), ("Medium", 3), ("Hard", 4)]
    buttons = []
    button_width, button_height = 100, 35
    start_x = dialog_x + 25
    spacing = (dialog_width - 3*button_width) // 4

    for i, (label, depth) in enumerate(difficulties):
        x = start_x + i*(button_width + spacing)
        y = dialog_y + 60
        buttons.append((pygame.Rect(x, y, button_width, button_height), depth, label))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for button, depth, _ in buttons:
                    if button.collidepoint(mouse_pos):
                        return depth

        for button, depth, label in buttons:
            color = BUTTON_HOVER if button.collidepoint(pygame.mouse.get_pos()) else DIALOG_BG
            pygame.draw.rect(screen, color, button)
            pygame.draw.rect(screen, DIALOG_BORDER, button, 1)
            text = pygame.font.SysFont('Arial', 14).render(label, True, BLACK)
            text_x = button.x + (button.width - text.get_width()) // 2
            text_y = button.y + (button.height - text.get_height()) // 2
            screen.blit(text, (text_x, text_y))

        pygame.display.flip()


def main():
    if rank != 0:
        # Worker nodes should run the worker process
        from worker import worker_process
        log("Starting worker process")
        worker_process()
        return
    
    # Only the master process (rank 0) runs the GUI
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess Game with MPI")
    clock = pygame.time.Clock()

    # Let user select difficulty
    ai_depth = show_difficulty_dialog(screen)
    
    try:
        load_piece_images()
    except Exception as e:
        log(f"Error loading piece images: {e}")
        print("Couldn't load piece images. Using text fallback.")
    
    board = chess.Board()
    selected_square = None
    status_text = "Your turn (White)"
    last_move = None
    ai_thinking = False
    game_over = False
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Send stop signal to all worker processes
                for i in range(1, size):
                    comm.send("STOP", dest=i)
                pygame.quit()
                sys.exit()
            
            if not game_over and not ai_thinking and board.turn:  # Human's turn (White)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    square = coords_to_square(x, y)
                    
                    if square is not None:
                        if selected_square is None:
                            # Select a piece
                            piece = board.piece_at(square)
                            if piece and piece.color == board.turn:
                                selected_square = square
                        else:
                            # Try to move the selected piece
                            move = chess.Move(selected_square, square)
                            # Check if it's a pawn promotion
                            piece = board.piece_at(selected_square)
                            is_promotion = (piece and piece.piece_type == chess.PAWN and 
                                           ((board.turn and square >= 56) or  # White pawn to 8th rank
                                            (not board.turn and square < 8)))  # Black pawn to 1st rank
                            
                            if is_promotion :
                                # Show promotion dialog
                                promotion_piece = show_promotion_dialog(screen)
                                move = chess.Move(selected_square, square, promotion=promotion_piece)
                            
                            if move in board.legal_moves:
                                board.push(move)
                                last_move = move
                                log(f"Human move: {move.uci()}")
                                selected_square = None
                                
                                if board.is_game_over():
                                    status_text = f"Game over! Result: {board.result()}"
                                    game_over = True
                                else:
                                    status_text = "AI is thinking..."
                                    ai_thinking = True
                                    
                                    # Start AI move calculation in a separate thread
                                    def on_ai_move_done(move):
                                        nonlocal ai_thinking, status_text, last_move, game_over
                                        if move:
                                            board.push(move)
                                            last_move = move
                                            log(f"AI move: {move.uci()}")
                                            
                                            if board.is_game_over():
                                                status_text = f"Game over! Result: {board.result()}"
                                                game_over = True
                                            else:
                                                status_text = "Your turn (White)"
                                        else:
                                            status_text = "AI error. Your turn again."
                                        ai_thinking = False
                                    
                                    threading.Thread(
                                        target=ai_move_thread,
                                        args=(board.copy(), ai_depth, on_ai_move_done)
                                    ).start()
                            else:
                                selected_square = square if board.piece_at(square) and board.piece_at(square).color == board.turn else None
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    # New game
                    board = chess.Board()
                    selected_square = None
                    status_text = "Your turn (White)"
                    last_move = None
                    ai_thinking = False
                    game_over = False
                
                elif event.key == pygame.K_u and not ai_thinking:
                    # Undo last two moves (human and AI)
                    if len(board.move_stack) >= 2:
                        board.pop()
                        board.pop()
                        last_move = board.peek() if board.move_stack else None
                        status_text = "Moves undone"
        
        # Update display
        screen.fill(WHITE)
        draw_board(screen, board, selected_square, last_move)
        draw_info(screen, board, status_text)
        
        # Draw footer with instructions
        font = pygame.font.SysFont('Arial', 12)
        instructions = "Press N for New Game | Press U to Undo Move"
        text = font.render(instructions, True, BLACK)
        screen.blit(text, (10, HEIGHT - 20))
        
        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()