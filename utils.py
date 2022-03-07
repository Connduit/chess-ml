import chess
import chess.engine
import os
import torch

#TODO: NUMPY'S np.flip(dims) is faster than PYTORCH'S torch.tensor.flip(dims)

def stockfish(board: chess.Board, depth: int, enginename: str):
    with chess.engine.SimpleEngine.popen_uci(f"{os.getcwd()}/{enginename}") as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result["score"].white().score()
        return score

def unravel_index(indices, shape):
    shape = torch.tensor(shape)
    indices = indices % shape.prod()

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        #indices = indices // dim
        indices = torch.div(indices, dim, rounding_mode="trunc")
    
    return coord.flip(-1)

def piece_matrix(board: chess.Board):
    #board3d = torch.zeros(14, 8, 8)
    #board3d = torch.zeros(12, 8, 8,dtype=torch.uint8)
    board3d = torch.zeros(14, 8, 8,dtype=torch.int8)
    #board3d = torch.zeros(12, 8, 8,dtype=torch.int8)

    for piece in chess.PIECE_TYPES:
        #for square in board.pieces(piece, chess.White):
        for square in board.pieces(piece, 1):
            idx = unravel_index(square, (8,8))
            #print(f"board3d[{piece -1}][7 - {idx[0]}][{idx[1]}] = 1")
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        
        #for square in board.pieces(piece, chess.Black):
        for square in board.pieces(piece, 0):
            idx = unravel_index(square, (8,8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1
    
    return board3d

def attacked_squares(board: chess.Board, color: chess.Color):
    a = chess.SquareSet()
    for attacker in chess.SquareSet(board.occupied_co[color]):
        a |= board.attacks(attacker)
    #print(a)
    #print()
    return totensor(a)
    #print()
    #return a

def totensor(s: chess.SquareSet):
    l = s.tolist()
    #l = l[::-1]
    l = list(reversed(l))


    l = [1 if cond else 0 for cond in l]
    t = torch.tensor(l).reshape(8,8).fliplr()
    #t = torch.flip(t, (8,8))
    #torch.reshape(t, (8,8))

    #print(t)
    return t

def fromBoard(board: chess.Board):
    #board3d = torch.zeros(14, 8, 8,dtype=torch.int8)
    board3d = piece_matrix(board)
    board3d[12] = attacked_squares(board, chess.WHITE)
    board3d[13] = attacked_squares(board, chess.BLACK)
    return board3d

def toDict(t: torch.tensor):
    d = dict()
    d["P"] = t[0]
    d["N"] = t[1]
    d["B"] = t[2]
    d["R"] = t[3]
    d["Q"] = t[4]
    d["K"] = t[5]

    d["p"] = t[6]
    d["n"] = t[7]
    d["b"] = t[8]
    d["r"] = t[9]
    d["q"] = t[10]
    d["k"] = t[11]

    d["wAttacks"] = t[12]
    d["bAttacks"] = t[13]
    return d
