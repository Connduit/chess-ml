import chess
import chess.pgn
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

#TODO: NUMPY'S np.flip(dims) is faster than PYTORCH'S torch.tensor.flip(dims)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unravel_index(indices, shape):
    shape = torch.tensor(shape)
    indices = indices % shape.prod()

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        #indices = indices // dim
        indices = torch.div(indices, dim, rounding_mode="trunc")
    
    return coord.flip(-1)


def encode_board(board: chess.Board):
    #board3d = torch.zeros(14, 8, 8, dtype=torch.uint8)
    board3d = np.zeros((14, 8, 8), dtype=np.uint8)

    """
    tensor order:
        White:
            Pawns
            Knights
            Bishops
            Rooks
            Queen
            King

        Black:
            Pawns
            Knights
            Bishops
            Rooks
            Queen
            King
    """

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            ind = piece.piece_type - 1 + (0 if piece.color else 6)
            #ind = piece.piece_type - 1 + (6 if piece.color else 0)
            #board3d[ind][square // 8][square % 8] = 1
            #board3d[ind][7 - (square // 8)][square % 8] = 1
            #print(f"board3d[ind={ind}][square // 8 = {square // 8}][square % 8 = {square % 8}] = 1")

            #idx = unravel_index(square, (8,8))
            idx = np.unravel_index(square, (8,8))
            board3d[ind][7 - idx[0]][idx[1]] = 1
            #print(f"board3d[ind={ind}][7 - idx[0] = {7 - idx[0]}][idx[1] = {idx[1]}] = 1")


    board3d[12] = attacked_squares(board, chess.WHITE)
    board3d[13] = attacked_squares(board, chess.BLACK)

    return torch.from_numpy(board3d)
    #return board3d


def attacked_squares(board: chess.Board, color: chess.Color):
    a = chess.SquareSet()
    for attacker in chess.SquareSet(board.occupied_co[color]):
        a |= board.attacks(attacker)

    return totensor(a)
    #return a

def totensor(s: chess.SquareSet):
    l = list(reversed(s.tolist()))
    #l = l[::-1]
    #l = list(reversed(l))


    # len(l) = 64
    l = [1 if cond else 0 for cond in l]
    #t = torch.tensor(l,device=device).reshape(8,8).fliplr()
    #t = torch.tensor(l).reshape(8,8).fliplr()
    t = np.fliplr(np.array(l, dtype=np.uint8).reshape((8,8)))

    return t

def decode_board(encoded: torch.tensor):
    pass


