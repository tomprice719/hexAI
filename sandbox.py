from board import Board
from utils import rb

problem = [2, 3, 5, 4, 19, 23, 22, 11, 15, 14, 21, 24, 8, 18, 6, 16, 7, 12, 17, 20]

b = Board(5)

for i, move in enumerate(problem):
    b.update(rb[i % 2], move)

print(b)
#b.print_stuff()