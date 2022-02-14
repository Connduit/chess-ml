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


TODO: it seems the engine is making the same opening moves everytime

"""
from asce import Asce
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
    arg_parser.add_argument("-v", "--visual", action="store_true", help="Board UI")
    arg_parser.add_argument("-d", "--depth", type=int, default=3, help="Search Depth")
    return arg_parser



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
    visual = vars(args.parse_args())["visual"]
    depth = vars(args.parse_args())["depth"]

    h = False
    if enginename == "human":
        h = True
    elif not os.path.exists(enginename):
        print("Not a valid file or path to file")
        args.print_usage()
        exit()
    else:
        engine = chess.engine.SimpleEngine.popen_uci(f"{os.getcwd()}/{enginename}")

    #board = chess.Board()
    board = chess.Board()
    if visual:
        display.start(board.fen())

    
    
    e = Asce(board, depth)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            b_move = e.alphaBeta(board, depth, -99999, 99999, True)[0]
            board.push(b_move)
        elif board.turn == chess.BLACK:
            if h:
                human(board)
            else:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
        if visual:
            display.update(board.fen())

    if not h:
        engine.quit()
    print(board.outcome())
    # TODO: make this pgn stuff its own class/method
    game = chess.pgn.Game()
    game = game.from_board(board)
    print(game)

    if visual:
        input()
        display.terminate()
    

    


if __name__ == "__main__":
    main()
    """
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
    """