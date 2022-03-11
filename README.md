# chess-ml


asce.py: A Simple Chess Engine

chessML.py: FILE I/O and CLI


## TODO
- Add more planes
    - in addition to piece planes (12x), encode n number previous positions (12 x n)
    - set plane to all ones if one or more repetitions occurred 
    - set plane to all ones if white/black can castle king/queen side (4x planes)




## NOTES
- the data that has been encoded are the positions of 
every given piece, and every square under attack from a 
given piece color. 
- there's a total of 14, 8 by 8, tensors for every board state

