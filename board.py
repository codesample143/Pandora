import chess
import numpy as np
import eval
import torch
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