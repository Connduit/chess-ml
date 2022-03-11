import os
import chess
import chess.pgn
import chess.engine
import numpy as np
import encoder_decoder
import timeit




def generate_dataset(num_samples, path = None):
    X,Y = [], []
    gn = 0
    values = {'1/2-1/2':0, '0-1':-1, '1-0':1}
    # pgn files in the data folder
    if path is None:
        for fn in os.listdir("data"):
            if os.path.splitext(fn)[1] != ".pgn":
                continue
            pgn = open(os.path.join("data", fn))
            while 1:
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
                    ser = encoder_decoder.encode_board(board)
                    X.append(ser)
                    Y.append(value)
                #print("parsing game %d, got %d examples" % (gn, len(X)))
                if num_samples is not None and len(X) > num_samples:
                    return X,Y
                gn += 1
        X = np.array(X)
        Y = np.array(Y)
        return X,Y
    else:
        while True:
            pgn = open(path)
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            res = game.headers["Result"]
            if res not in values:
                continue
            value = values[res]
            board = game.board()
            for _, move in enumerate(game.mainline_moves()):
                board.push(move)
                ser = encoder_decoder.encode_board(board)
                X.append(ser)
                #start = timeit.default_timer()
                val = stockfish(board, 10)
                #stop = timeit.default_timer()
                #t = stop - start
                #print(round(t, 5))
                Y.append(val)
            if num_samples is not None and len(X) > num_samples:
                return X,Y
            gn += 1
        X = np.array(X)
        Y = np.array(Y)
        return X,Y


#def stockfish(path, board, depth):
def stockfish(board, depth):
    #with chess.engine.SimpleEngine.popen_uci(path) as sf:
    enginename = "stockfish_14.1_win_x64_avx2.exe"
    with chess.engine.SimpleEngine.popen_uci(f"{os.getcwd()}/{enginename}") as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result["score"].white().score()
        return score

def main():
    board_state, board_eval = generate_dataset(25000)
    #board_state, board_eval = generate_dataset("data/ficsgamesdb_2010_standard2000_nomovetimes_239004.pgn")
    np.savez("processed/my_dataset25k.npz", board_state=board_state, board_eval=board_eval)
    #np.savez("processed/my_dataset_10games.npz", board_state=board_state, board_eval=board_eval)

if __name__ == "__main__":
    main()