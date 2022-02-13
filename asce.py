import chess

# A Simple Chess Engine
class Asce:
    #def __init__(self, board: chess.Board, color: chess.Color):
    def __init__(self, board: chess.Board, depth: int):
        self.board = board
        self.depth = depth
        self.material = 0


    def evaluate(self):
        material = 0
        board_pieces = self.board.piece_map()
        for piece in board_pieces.values():
            material += self.count(piece)
        
        state = 0
        if self.board.turn:
            state += 1
        else:
            state -= 1

        return material * state

    def count(self, piece: chess.Piece):
        val = 0
        if piece.color:
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
        else:
            if piece.piece_type == chess.PAWN:
                val -= 1
            elif piece.piece_type == chess.KNIGHT:
                val -= 3
            elif piece.piece_type == chess.BISHOP:
                val -= 3
            elif piece.piece_type == chess.ROOK:
                val -= 5
            elif piece.piece_type == chess.QUEEN:
                val -= 9
            elif piece.piece_type == chess.KING:
                val -= 0

        return val

    def search(self, depth):

        if depth == 0:
            return self.evaluate()

        best_eval = float("-inf")
        best_move = "temp move"
        for move in self.board.legal_moves:
            self.board.push(move)
            new_eval = -self.search(depth - 1)
            if new_eval > best_eval:
                best_move = move
                best_eval = new_eval
            
            self.board.pop()
        
        if depth == 3:
            #print(best_move)
            #print(best_eval)
            return best_move
        return best_eval


