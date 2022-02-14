import chess

# A Simple Chess Engine
class Asce:
    #def __init__(self, board: chess.Board, color: chess.Color):
    def __init__(self, board: chess.Board, depth: int):
        self.board = board
        self.depth = depth
        self.material = 0

    def alphaBeta(self, board: chess.Board, depth, alpha, beta, maximize):
        if board.is_checkmate():
            if board.turn:
                # if it's white's turn
                return None,-10000
            else:
                return None, 10000
        if depth == 0:
            return None, self.e(board)

        if maximize:
            max_val = -999999
            for move in board.legal_moves:
                board.push(move)
                current =  self.alphaBeta(board, depth-1, alpha, beta, (not maximize))[1]
                board.pop()
                if current > max_val:
                    max_val = current
                    best_move = move
                alpha = max(alpha, current)
                if alpha >= beta:
                    break
                    #return bestVal
            return best_move, max_val
        else:
            min_val = 999999
            for move in board.legal_moves:
                board.push(move)
                curr = self.alphaBeta(board, depth-1, alpha, beta, (not maximize))[1]
                board.pop()
                if curr < min_val:
                    min_val = curr
                    best_move = move
                beta = min(beta, curr)
                if beta <= alpha: 
                    break
                    #return bestVal
            return best_move, min_val


    def e(self, board: chess.Board):
        material = 0
        board_pieces = board.piece_map()
        for piece in board_pieces.values():
            material += self.count(piece)
        
        """ #state = 0
        if board.turn:
            state += 1
        else:
            state -= 1
        """
        return material #* state

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


