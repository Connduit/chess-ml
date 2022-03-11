import os
import chess
import chess.engine
import chess.pgn
import numpy as np
import encoder_decoder
from chess_net import CNN
import torch



def minimax_eval(board, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board3d = encoder_decoder.encode_board(board)
    #print(board3d)
    board3d = np.expand_dims(board3d, 0)
    board3d = torch.from_numpy(board3d)
    board3d = board3d.to(device)
    board3d = board3d.float()
    #print(board3d)
    #print("=======================================================")
    #print("=======================================================")
    #print("=======================================================")
    #print(board3d)
    return model(board3d)
    return model.predict(board3d)[0][0]


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


# this is the actual function that gets the move from the neural network
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board = chess.Board()
    model = CNN()
    model.load_state_dict(torch.load(f"{os.getcwd()}/nets/my_values.pth"))
    model.eval()
    model = model.to(device)

    enginename = "stockfish_14.1_win_x64_avx2.exe"
    with chess.engine.SimpleEngine.popen_uci(f"{os.getcwd()}/{enginename}") as sf:
        sf.configure({"Skill Level": 3})
        while True:
            move = get_ai_move(board, 3, model)
            #move = get_ai_move(board, 1)
            board.push(move)
            #print(board)
            if board.is_game_over():
                break

            result = sf.play(board, chess.engine.Limit(time=0.05))
            board.push(result.move)
            #move = sf.analyse(board, chess.engine.Limit(time=1), info=chess.engine.INFO_PV)['pv'][0]
            #board.push(move)
            #print(board)
            if board.is_game_over():
                break

    g = chess.pgn.Game()
    g = g.from_board(board)
    print(g)

if __name__ == "__main__":
    main()
    #import torch.multiprocessing as mp
    #mp.Process