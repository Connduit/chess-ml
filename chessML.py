"""
Websites for Chess Data:
    https://www.ficsgames.org/
    http://database.chessbase.com/
    https://www.kaggle.com/datasnaek/chess
    https://www.kaggle.com/milesh1/35-million-chess-games

Other:
    https://lczero.org/
    https://github.com/mrklees/deepconv-chess
    https://github.com/geochri/AlphaZero_Chess
    https://python-chess.readthedocs.io/en/latest/

    KERAS
    TENSORFLOW
    PYTORCH

"""
import os
import random
import argparse
import chess
import chess.engine
import chess.pgn
from chessboard import display

def cli():
    arg_parser = argparse.ArgumentParser(description="Plays Chess Against Chess Engine")
    arg_parser.add_argument("-e", "--enginename", required=True, help="Chess Engine Name")
    return arg_parser


# A Simple Chess Engine
class Asce:
    def __init__(self, board: chess.Board, color: chess.Color):
        self.board = board
        self.color = color
        self.material = 0


    def evaluate(self):
        #self.board.pieces()
        mat = 0
        d = self.board.piece_map()
        for k,v in d.items():
            #print(f"key = {k}, value = {v}")
            #self.material += self.count(v)
            mat += self.count(v)
        #return self.material
        return mat

    def count(self, piece: chess.Piece):
        val = 0
        if piece.piece_type == chess.PAWN:
            val += 1
        elif piece.piece_type == chess.KNIGHT:
            val += 3
        elif piece.piece_type == chess.BISHOP:
            val += 3
        elif piece.piece_type == chess.ROOK:
            val += 5
        elif piece.piece_type == chess.QUEEN:
            val += 9
        elif piece.piece_type == chess.KING:
            val += 0

        if piece.color:
            return val
        else:
            return -1 * val

    def search(self, depth):
        if depth == 0:
            return self.evaluate()

        best_eval = float("-inf")
        best_move = "temp move"
        #print(self.board.turn)
        for move in self.board.legal_moves:
            #make move
            self.board.push(move)
            new_eval = self.search(depth - 1)
            if new_eval > best_eval:
                best_move = move
                best_eval = new_eval
            
            #best_eval = max(best_eval, new_eval)
            # unmake move
            self.board.pop()
        
        if depth == 3:
            #print(best_move)
            return best_move
        return best_eval






def random_move(board):
    move = random.choice(list(board.legal_moves))
    board.push(move)
    return move


def my_engine(board):
    move = random.choice(list(board.legal_moves))
    #print(type(uci_move))
    #move = chess.Move.from_uci(uci_move)
    board.push(move)
    return None

def human(board):
    #print(board)

    while True:
        user_move = input("Enter move (eg., e4): ")
        try:
            board.push_san(user_move)
            break
        except ValueError:
            print("not a legal move. try again")
        
    #move = chess.Move.from_uci(user_move)
    #move = chess.Move.
    #board.push(move)

# TODO: add piece weights (might already be part of chess modual)
def main():

    args = cli()
    enginename = vars(args.parse_args())["enginename"]

    h = False
    if enginename == "human":
        h = True
    elif not os.path.exists(enginename):
        print("Not a valid file or path to file")
        args.print_usage()
        exit()
    else:
        engine = chess.engine.SimpleEngine.popen_uci(f"{os.getcwd()}/{enginename}")

    board = chess.Board()
    if True:
        display.start(board.fen())

    """
    chess.WHITE = True
    chess.BLACK = False
    """
    e = Asce(board, chess.WHITE)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            #my_engine(board)
            b_move = e.search(3)
            board.push(b_move)
        elif board.turn == chess.BLACK:
            if h:
                human(board)
            else:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
            #my_engine(board)
        if True:
            display.update(board.fen())
        #print(board)
        #print()

    if not h:
        engine.quit()
    print(board.outcome())
    # TODO: make this pgn stuff its own class/method
    game = chess.pgn.Game()
    game = game.from_board(board)
    print(game)

    if True:
        display.terminate()
    

    


if __name__ == "__main__":
    main()
    #b = chess.Board("r3kb1r/ppp1pppp/2n2n2/4q3/8/4Q1P1/PPPPbPBP/RNB1K2R w KQkq - 0 8")
    b = chess.Board()
    #print(b.turn)
    #print(random_move(b))
    #print(b.pop())
    #print(b.turn)
    eng = Asce(b, chess.WHITE)
    #eng.evaluate()
    
    print(eng.search(3))
    if b.turn:
        b.push(chess.Move.from_uci("e3e5"))
        
    b.push(chess.Move.from_uci("c6e5"))

    print(eng.search(3))
    #main()
