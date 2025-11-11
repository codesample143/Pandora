import chess 
import numpy as np
import eval
import torch
import torch.nn as nn
import torch.optim as optim
import chess.engine
import chess.pgn
import math
import chess.syzygy
import chess.polyglot

#    .
#   /\
#   . . 
#   /\/\
#  . .. .
# 8x8 grid of pieces.
# Options for piece encoding: OHE, Keys, pairs 
# MLP that takes board as input 
# use key values for piece encoding, feed into MLP 
# 8x8 matrix of values that correspond to pieces


# P -> 1, B -> 2, N -> 3, R -> 4, Q -> 5, K-> 6, p -> 7, b -> 8, n -> 9, r -> 10, q -> 11, k -> 12 
# function for encoding a board 
# returning a 2d tensor as numpy array
# Starts at a1 and comes down to h8
# enc(board).flatten() gives the 64x1 vector for the board
def enc(board):
    ret = []
    current_row = []
    for square in chess.SQUARE_NAMES:
        try:
            current_row.append((p:=board.piece_at(chess.parse_square(square))).piece_type + int(not p.color)*6)
            if len(current_row) == 8:
                ret.append(current_row)
                current_row = [] 
        except:
            current_row.append(0)
            if len(current_row) == 8:
                ret.append(current_row)
                current_row = [] 
            continue 
    return torch.tensor(ret).to(dtype=torch.float32)

# Finds the basic eval of an encoded board(flattened)
def basic_eval(enc_board):
    valdict = {1:1, 2:3, 3:3, 4:5, 5:9, 7:-1, 8:-3, 9:-3, 10:-5, 11:-9}
    s = 0
    for val in enc_board:
        if val in valdict.keys():
            s = s + valdict[val]
    return s


# training data: inps = [chess board objects], outs = [values]
def get_static_eval(b, stockfish_path='/Users/akaei/Desktop/C++ Projects/Pandora/stockfish/stockfish-macos-m1-apple-silicon'):
    fen = b.fen()
    # Connect to the Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = chess.Board(fen)

    # Define a search limit with depth=0
    # This is the key: it forces the engine to return the score
    # of the current node *before* beginning any search (minimax/quiescence).
    limit = chess.engine.Limit(depth=0)
    info = engine.analyse(board, limit, info=chess.engine.INFO_SCORE)
    score = info["score"]
    engine.quit()    
    return score

def sig(x,alpha=1,beta=1):
    return beta/(1+np.exp(-alpha*x))

def get_sf_eval(board):
    info = get_static_eval(board).white()
    if str(info)[:2] == '#+':
        return(1)
    elif str(info)[:2] == '#-':
        return(0)
    return sig(info.score()/100)

# reutrns all legal moves in uci format 
def retmoves(b):
    return [move.uci() for move in list(b.legal_moves)]

# train a neural network and return it
# numgames -> how many games to pick moves from
# layers -> the layer list of tuples for network architecture
# save name -> will not save if None, saves with name otherwise
# trainfile -> the file name holding the master games
def train(numgames, layers, epochs=100, lr=0.01, save_name=None, trainfile='/Users/akaei/Desktop/C++ Projects/Pandora/AJ-OTB-PGN-00/AJ-OTB-PGN-000-002.pgn', show=False):
    # Model: Chess board obj -> score(float)
    # 64 nodes -> 128 nodes -> 32 nodes -> 10 nodes -> 1 node
    model = eval.Evaluation(layers)

    # list of encoded boards
    inpx = []

    # list of valuations 
    outs = []

    with open(trainfile, encoding="utf-8") as pgn_file:
        # Use a loop to read all games in the file
        for _ in range(numgames):
            print(f'on game number {_}')
            game = chess.pgn.read_game(pgn_file)

            board = game.board() 
            mvc = 0
            # Iterate over the moves in the main line of the game
            for move_num, move in enumerate(game.mainline_moves(), 1):
                board.push(move)
                print(board)
                print('-')
                print(board.fen())
                print(get_sf_eval(board))
                try:
                    if not board.is_checkmate() and not board.is_stalemate():
                        inpx.append(enc(board).flatten())
                        outs.append(get_sf_eval(board))
                except:
                    continue 

    train_inps = torch.stack(inpx)

    # --- 💡 FIX 2: Convert outs (Target data) ---
    # Convert the list of floats into a single 1D tensor
    train_outs = torch.tensor(outs, dtype=torch.float32).unsqueeze(1)

    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.SGD(model.parameters(), lr=lr) # Stochastic Gradient Descent

    # 4. Training loop with forward pass and backpropagation
    for epoch in range(epochs):
        # Forward pass
        outputs = model(train_inps)
        loss = criterion(outputs, train_outs)

        # Backward and optimize
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update model parameters

        if (epoch+1) % 10 == 0 and show:
            print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')

    if save_name != None:
        torch.save(model, save_name)

    return model

# This line is for if you are training a model
#train(15, [(64,'relu'),(32,'relu'),(10,'relu'),(1,'sigmoid')], epochs=2000, lr=0.05, save_name='mdl__64_32_10_1__15_2000_005.pth', show=True)

# Define the constants for clarity
INF = math.inf

def minimax_alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float, 
                      is_maximizing_player: bool, eval_func) -> float:
    """
    Implements the Minimax algorithm with Alpha-Beta Pruning.
    """
    
    # 1. Base Case: Terminal Node or Max Depth Reached
    if depth == 0 or board.is_game_over():
        # Evaluate the position from White's (maximizing) perspective
        return eval_func(board)

    # 2. Recursive Step

    if is_maximizing_player:
        # Maximizing (White) seeks the highest score
        max_eval = -INF
        
        for move in board.legal_moves:
            board.push(move)
            
            # Recursive call: The maximizing player's move is the minimizing player's turn next.
            # Alpha and Beta are passed down.
            evaluation = minimax_alpha_beta(board, depth - 1, alpha, beta, False, eval_func)
            
            board.pop()
            
            max_eval = max(max_eval, evaluation)
            
            # --- Alpha-Beta Pruning Check (MAX Node) ---
            alpha = max(alpha, max_eval)
            if beta <= alpha:
                # The minimizing player (parent node) already has a better option. 
                # We can stop searching this branch.
                break
                
        return max_eval

    else:
        # Minimizing (Black) seeks the lowest score
        min_eval = INF
        
        for move in board.legal_moves:
            board.push(move)
            
            # Recursive call: The minimizing player's move is the maximizing player's turn next.
            evaluation = minimax_alpha_beta(board, depth - 1, alpha, beta, True, eval_func)
            
            board.pop()
            
            min_eval = min(min_eval, evaluation)
            
            # --- Alpha-Beta Pruning Check (MIN Node) ---
            beta = min(beta, min_eval)
            if beta <= alpha:
                # The maximizing player (parent node) already has a better option. 
                # We can stop searching this branch.
                break
                
        return min_eval

def get_book_move(board: chess.Board, BOOK_PATH='/Users/akaei/Desktop/C++ Projects/Pandora/Titans.bin') -> chess.Move | None:
    """
    Looks up a move in the Polyglot opening book.
    Returns a Move object or None if the position is not in the book.
    """
    try:
        # Open the book reader
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            
            # Use weighted_choice to randomly select a move based on its popularity/success score
            # This makes the engine less predictable than always picking the "best" move.
            entry = reader.weighted_choice(board) 
            
            return entry.move
            
    except FileNotFoundError:
        # Handle the case where the book file isn't found
        print(f"Warning: Opening book not found at {BOOK_PATH}")
        return None
    except IndexError:
        # This occurs if the position is not found in the book
        return None
    except Exception as e:
        print(f"Error reading Polyglot book: {e}")
        return None

TABLEBASE_DIR = "/Users/akaei/Desktop/C++ Projects/Pandora/syzygy" # <--- Update this path!

# Initialize the tablebase reader once at the start of your program
try:
    # This keeps the tablebase files open for quick access
    tablebase_reader = chess.syzygy.open_tablebase(TABLEBASE_DIR)
    print("Syzygy Tablebases loaded successfully.")
except Exception as e:
    tablebase_reader = None
    print(f"Warning: Could not load Syzygy Tablebases. {e}")


def get_tablebase_move(board: chess.Board) -> chess.Move | None:
    """
    Probes the loaded tablebases for a perfect endgame move.
    """
    if tablebase_reader is None:
        return None
    
    # Only probe if the position is covered (typically <= 7 pieces)
    if len(board.piece_map()) > 5: # Example: Only probe 5-piece and below
        return None

    try:
        # Probe the tablebases for the perfect move at the root position
        # `dtz` stands for Distance To Zero (distance to checkmate or draw/50-move rule zero)
        entry = tablebase_reader.probe_root(board)
        
        # If an entry is found, choose the move that yields the best result (score)
        if entry.moves:
            # The probe_root entry provides the move with the best 'dtz' score
            best_move_in_entry = entry.best_move 
            
            # Optionally, you can check the WDL (Win/Draw/Loss) score:
            # wdl_score = tablebase_reader.probe_wdl(board)
            
            print(f"Tablebase found! Score: {entry.score}, Best move: {board.san(best_move_in_entry)}")
            return best_move_in_entry
            
        return None

    except chess.syzygy.MissingTableError:
        # Position is not covered by the available tables
        return None
    except Exception as e:
        print(f"Error during tablebase probe: {e}")
        return None

def select_best_move_ab(board: chess.Board, depth: int, eval_func) -> chess.Move:
    """
    Finds the best move for the current player using Minimax with Alpha-Beta Pruning.
    """
    book_move = get_book_move(board)
    if book_move:
        return book_move

    tabl_move = get_tablebase_move(board)
    if tabl_move:
        return tabl_move

    # Initialize Alpha and Beta for the root node search
    alpha = -INF
    beta = INF
    
    # Set initial best_eval based on whose turn it is
    if board.turn == chess.WHITE:
        best_eval = -INF  # White (Maximizing) starts low
    else:
        best_eval = INF   # Black (Minimizing) starts high
        
    best_move = None
    
    # Check all legal moves
    for move in board.legal_moves:
        board.push(move)
        
        # Call the new function with initial alpha and beta
        # The next turn is the *opposite* of the current turn
        current_eval = minimax_alpha_beta(
            board, 
            depth - 1, 
            alpha, 
            beta, 
            board.turn == chess.WHITE, # Pass the boolean for the NEXT player's turn
            eval_func
        )
        
        board.pop() # Undo the move

        # Update best move and the Alpha/Beta boundary for the root node
        if board.turn == chess.WHITE:
            if current_eval > best_eval:
                best_eval = current_eval
                best_move = move
            # Update the root node's alpha value (critical for pruning subsequent root moves)
            alpha = max(alpha, best_eval) 
        else:
            if current_eval < best_eval:
                best_eval = current_eval
                best_move = move
            # Update the root node's beta value
            beta = min(beta, best_eval) 
                
    print(f"Best score found: {best_eval}")
    return best_move
