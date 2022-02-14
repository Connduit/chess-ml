import chess
pieceValues = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

pieceSquareTable = {
    chess.A1: -50, chess.B1: -40, chess.C1: -30, chess.D1: -30, chess.E1: -30, chess.F1: -30, chess.G1: -40, chess.H1: -50,
    chess.A2: -40, chess.B2: -20, chess.C2:   0, chess.D2:   0, chess.E2:   0, chess.F2:   0, chess.G2: -20, chess.H2: -50,
    chess.A3: -30, chess.B3:   0, chess.C3:  10, chess.D3:  15, chess.E3:  15, chess.F3:  10, chess.G3:   0, chess.H3: -50,
    chess.A4: -30, chess.B4:   5, chess.C4:  15, chess.D4:  20, chess.E4:  20, chess.F4:  15, chess.G4:   5, chess.H4: -50,
    chess.A5: -30, chess.B5:   0, chess.C5:  15, chess.D5:  20, chess.E5:  20, chess.F5:  15, chess.G5:   0, chess.H5: -50,
    chess.A6: -30, chess.B6:   5, chess.C6:  10, chess.D6:  15, chess.E6:  15, chess.F6:  10, chess.G6:   5, chess.H6: -50,
    chess.A7: -40, chess.B7: -20, chess.C7:   0, chess.D7:   5, chess.E7:   5, chess.F7:   0, chess.G7: -20, chess.H7: -50,
    chess.A8: -50, chess.B8: -40, chess.C8: -30, chess.D8: -30, chess.E8: -30, chess.F8: -30, chess.G8: -40, chess.H8: -50
}
