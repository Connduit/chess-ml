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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #board3d = torch.zeros(14, 8, 8, dtype=torch.uint8, device=device)
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
            # ind is always correct
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
    return board3d


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


def get_dataset(num_samples=None):
    import os
    X,Y = [], []
    game_number = 0
    values = {'1/2-1/2':0, '0-1':-1, '1-0':1}
    # pgn files in the data folder
    for fn in os.listdir("data"):
        if os.path.splitext(fn)[1] != ".pgn":
            continue
        pgn = open(os.path.join("data", fn))
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            res = game.headers['Result']
            if res not in values:
                continue
            value = values[res]
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                #ser = State(board).serialize()
                ser = encode_board(board)
                X.append(ser)
                Y.append(value)
            #print("parsing game %d, got %d examples" % (game_number, len(X)))
            if num_samples is not None and len(X) > num_samples:
                return X,Y
            game_number += 1

    X = np.array(X)
    Y = np.array(Y)
    #X = torch.tensor(X)
    #Y = torch.tensor(Y)

    return X,Y

#linear layer = dense layer
# 14 separate channels? one for each matrix/tensor?
class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        """
        self.layer1 = nn.Sequential(
            #nn.Conv2d(8?, 16, kernel_size=3, padding=1),
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
        )
        """
        self.a1 = nn.Conv2d(14, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)

        self.last = nn.Linear(128, 1)

        #self.conv1 = nn.Conv2d() 
        #self.fc1 = nn.Linear(2048,64)
        #self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(-1, 128)
        x = self.last(x)
        """
        #print(x.size())
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, 128)
        x = self.last(x)

        return torch.tanh(x)

def main():
    #board = chess.Board()
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    board3d = encode_board(board)
    print(board3d)

    #X,Y = get_dataset(25000000)
    #X,Y = get_dataset(25000)
    #np.savez("processed/dataset_25M.npz", X, Y)
    #np.savez("processed/dataset_25K.npz", X, Y)

if __name__ == "__main__":
    main()
