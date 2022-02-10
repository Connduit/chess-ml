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

"""
import os
import random
import argparse
import chess
import chess.engine
import chess.pgn
#import chess.svg
#import asyncio

def cli():
    arg_parser = argparse.ArgumentParser(description="Plays Chess Against Chess Engine")
    arg_parser.add_argument("-e", "--enginename", required=True, help="Chess Engine Name")
    #args = arg_parser.parse_args()
    return arg_parser



def my_engine(board):
    move = random.choice(list(board.legal_moves))
    #print(type(uci_move))
    #move = chess.Move.from_uci(uci_move)
    board.push(move)
    return None

def main():

    args = cli()
    enginename = vars(args.parse_args())["enginename"]
    if not os.path.exists(enginename):
        print("Not a valid file or path to file")
        args.print_usage()
        exit()


    engine = chess.engine.SimpleEngine.popen_uci(f"{os.getcwd()}/{enginename}")
    board = chess.Board()
    """
    chess.WHITE = True
    chess.BLACK = False
    """
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            my_engine(board)
        elif board.turn == chess.BLACK:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)
            #my_engine(board)
        print(board)
        print()

    engine.quit()
    
    print(board.outcome())
    # TODO: make this pgn stuff its own class/method
    game = chess.pgn.Game()
    game = game.from_board(board)
    print(game)

    





#asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
#asyncio.run(main())
if __name__ == "__main__":
    main()
