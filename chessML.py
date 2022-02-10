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
import chess
import chess.engine
import os
import asyncio




async def main():

    transport, engine = await chess.engine.SimpleEngine.popen_uci(f"{os.getcwd()}/stockfish_14.1_win_x64_avx2.exe")
    board = chess.Board()

    while not board.is_game_over():
        result = await engine.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
    
    await engine.quit()

asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
asyncio.run(main())
#main()
