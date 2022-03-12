from encoder import encode_board
import os
import chess
import chess.pgn
import chess.engine
import numpy as np
import argparse

def cli():
    arg_parser = argparse.ArgumentParser(description="Builds a Dataset from a .PGN File")
    arg_parser.add_argument("-l", "--load_path", required=True, help=".PGN File Location")
    arg_parser.add_argument("-s", "--store_path", required=True, help="Save Location")
    arg_parser.add_argument("-d", "--depth", type=int, default=5, help="Search Depth")
    #arg_parser.add_argument("-ns", "--num_samples", type=int, default=None, help="")
    #arg_parser.add_argument("-ng", "--num_games", type=int, default=None, help="")
    return arg_parser

#def generate_dataset(num_samples, path = None):
def generate_dataset(path, depth, num_board_states = None, num_games = None):
    board_states, board_evals = [], []
    gn = 0
    values = {'1/2-1/2':0, '0-1':-1, '1-0':1}
    pgn = open(path)
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        res = game.headers["Result"]
        if res not in values:
            continue
        board = game.board()
        for _, move in enumerate(game.mainline_moves()):
            board.push(move)
            curr_board_state = encode_board(board)
            board_states.append(curr_board_state)
            curr_board_eval = evaluator(board, depth)
            board_evals.append(curr_board_eval)
        if (num_board_states is not None and len(board_states) > num_board_states) or (num_games is not None and gn > num_games):
            return board_states, board_evals
        gn += 1
    board_states = np.array(board_states)
    board_evals = np.array(board_evals)
    pgn.close()
    return board_states, board_evals


def evaluator(board, depth=5, enginename="stockfish_14.1_win_x64_avx2.exe"):
    with chess.engine.SimpleEngine.popen_uci(f"{os.getcwd()}/{enginename}") as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result["score"].white().score()
        return score

def main():
    args = cli()
    data = vars(args.parse_args())["load_path"]
    processed_data = vars(args.parse_args())["store_path"]
    depth = vars(args.parse_args())["depth"]

    board_state, board_eval = generate_dataset(data, depth)
    np.savez(processed_data, board_state=board_state, board_eval=board_eval)

if __name__ == "__main__":
    main()