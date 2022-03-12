import os
import chess
import chess.engine
import chess.pgn
import numpy as np
import encoder
from chess_net import CNN
import torch
import argparse

def cli():
    arg_parser = argparse.ArgumentParser(description="Trained Model Plays Against Stockfish")
    arg_parser.add_argument("-l", "--load_path", required=True, help="Model Location")
    arg_parser.add_argument("-d", "--depth", type=int, default=2, help="Search Depth")
    return arg_parser


def minimax_eval(board, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board3d = encoder.encode_board(board)
    board3d = np.expand_dims(board3d, 0)
    board3d = torch.from_numpy(board3d)
    board3d = board3d.to(device)
    board3d = board3d.float()
    output = model(board3d)
    return float(output.data[0][0])


def minimax(board, depth, alpha, beta, maximizing_player, model):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board, model)
  
    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def get_ai_move(board, depth, model):
    max_move = None
    max_eval = -np.inf

    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -np.inf, np.inf, False, model)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            max_move = move
  
    return max_move


def play_engine(model_path, net_depth, enginename = "stockfish_14.1_win_x64_avx2.exe"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board = chess.Board()
    model = CNN()
    #model.load_state_dict(torch.load(f"{os.getcwd()}/nets/alpha1-5M.pth"))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)

    with chess.engine.SimpleEngine.popen_uci(f"{os.getcwd()}/{enginename}") as sf:
        sf.configure({"Skill Level": 1})
        while True:
            move = get_ai_move(board, net_depth, model)
            #move = get_ai_move(board, 1)
            board.push(move)
            #print(board)
            if board.is_game_over():
                break

            result = sf.play(board, chess.engine.Limit(time=0.05))
            board.push(result.move)
            if board.is_game_over():
                break

    g = chess.pgn.Game()
    g = g.from_board(board)
    print(g)

if __name__ == "__main__":
    args = cli()
    model_path = vars(args.parse_args())["load_path"]
    depth = vars(args.parse_args())["depth"]
    play_engine(model_path, depth)