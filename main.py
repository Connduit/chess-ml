import pgn_reader as pr
import numpy as np
import chess_net as cn





def main():
    #board_state, board_eval = pr.generate_dataset(25000)
    #board_state, board_eval = generate_dataset("data/ficsgamesdb_2010_standard2000_nomovetimes_239004.pgn")
    #np.savez("processed/my_dataset25k.npz", board_state=board_state, board_eval=board_eval)

    cn.train("dataset.npz", "nets/my_values_1-5M.pth", 1000)


if __name__ == "__main__":
    main()