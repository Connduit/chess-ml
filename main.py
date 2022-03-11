import pgn_reader as pr
import numpy as np
import chess_net as cn

import chess
import torch
import os
import encoder

def test(board):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cn.CNN()
    #model.load_state_dict(torch.load(f"{os.getcwd()}/nets/my_values_25k.pth"))
    model.load_state_dict(torch.load(f"{os.getcwd()}/nets/my_values_1-5M.pth"))
    model.eval()
    model = model.to(device)

    board3d = encoder.encode_board(board)
    board3d = np.expand_dims(board3d, 0)
    board3d = torch.from_numpy(board3d)
    print(board3d.size())
    board3d = board3d.to(device)
    board3d = board3d.float()
    output = model(board3d)
    print(float(output.data[0][0]))


def main():
    #board_state, board_eval = pr.generate_dataset(25000)
    #board_state, board_eval = generate_dataset("data/ficsgamesdb_2010_standard2000_nomovetimes_239004.pgn")
    #np.savez("processed/my_dataset25k.npz", board_state=board_state, board_eval=board_eval)
    #board_state, board_eval = pr.generate_dataset(25000)
    #board_state, board_eval = generate_dataset("data/ficsgamesdb_2010_standard2000_nomovetimes_239004.pgn")
    #ficsgamesdb_2021_standard2000_nomovetimes_238993
    #cn.train("dataset.npz", "nets/my_values_1-5M.pth", 50)
    #cn.train("processed/my_dataset25k.npz", "nets/my_values_25k.pth", 200)
    b = chess.Board("1N6/8/p2p2rq/4p3/PK3PkP/4bR2/2P1N2P/8 w - - 0 1")
    x= test(b)


if __name__ == "__main__":
    main()