import chess
import numpy as np
import eval
import torch
from torch import nn
from torch import optim
import chess.engine
# 8x8 grid of pieces.
# OHE? options for piece encoding: OHE, Keys, Pairs, 
# MLP that takes board as input (feed forward neural network)
board = chess.Board()

# use KV for piece encoding, feed into MLP
# 8x8 matrix of values that correspond to pieces

# P -> 1, B-> 2, N->3, R->4, Q->5, K->6, p -> 7, b->8, n -> 9, r->10 q->11, k->12
# lowcase black
# function to encoding a board
# output: [[]]
# A1 -> starts with White Rook ends with A8
def enc(board):
    # return matrix
    ret = [] 
    current_row = []
    for square in chess.SQUARE_NAMES:
        try:
            current_row.append((p:= board.piece_at(chess.parse_square(square))).piece_type + int(not p.color) * 6)
            if(len(current_row) == 8):
                ret.append(current_row)
                current_row = []
        except:
            current_row.append(0)
            if(len(current_row) == 8):
                ret.append(current_row)
                current_row = []
            continue
    return torch.tensor(ret).to(dtype=torch.float32)

# 8x8 2d tensor->flatten 1d vector -> 
enc_board = enc(board).flatten()

# everything happens from this board state, value, minmax
# 64 x 1 -> 128- > 32 nodes -> 10 nodes -> 1
model = eval.Evaluation([(64, 'relu'), (128, 'relu'), (32, 'relu'), (10, 'relu'), (1, 'sigmoid')])
print(model(enc_board))

def get_static_eval(b, stockfish_path = '/Users/akaei/Desktop/C++ Projects/Pandora/stockfish/stockfish-macos-m1-apple-silicon'):
    fen = b.fen()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = chess.Board(fen)
    #define a sarch limit with depth = 0
    # This is the ke: it forces engine to return the score before any min max sequence
    limit = chess.engine.Limit(depth=0)
    info = engine.analyse(board, limit, info=chess.engine.INFO_SCORE)
    score = info["score"]
    engine.quit()
    return score
def sig(x, alpha=1, beta=1):
    return beta/(1 + np.exp(-alpha * x))

# chess board boj -> score(float)
# training data
print(sig(get_static_eval(board).white().score()/100))


# training set
def get_sf_eval(board):
    return sig(get_static_eval(board).white().score()/100)

def retmoves(board):
    return [move.uci() for move in list(board.legal_moves)]

inpx = []
outx = []
#returns all legal moves in uci format

#play one move, then pipe
for i in range(1):
    cm = retmoves(board)
    np.random.shuffle(cm)
    for move1 in cm[:10]:
        print(f"move1: {move1}")
        board.push_uci(move1)
        inpx.append(enc(board).flatten())
        outx.append(get_sf_eval(board))
        cm2 = retmoves(board)
        np.random.shuffle(cm2)
        for move2 in cm2[:10]:
            print(f"move2: {move2}")
            board.push_uci(move2)
            inpx.append(enc(board).flatten())
            outx.append(get_sf_eval(board))
            board.pop()
        board.pop()
print(inpx)
print(outx)

train_inps = torch.stack(inpx)
train_outs = torch.tensor(outx, dtype=torch.float32).unsqueeze(1) # Need a tensor of tessor of data

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

for epoch in range(250):
    outputs = model(train_inps)
    loss = criterion(outputs, train_outs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
