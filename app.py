import string
import random

def create_grid(size):
    return [[random.choice(string.ascii_uppercase) for _ in range(size)] for _ in range(size)]

def hide_word(grid, word, directions):
    n = len(grid)
    dir_map = {
        "H": (0,1), "V": (1,0),
        "D1": (1,1), "D2": (1,-1)
    }

    while True:
        dx, dy = dir_map[random.choice(directions)]
        x = random.randint(0, n-1)
        y = random.randint(0, n-1)

        if 0 <= x + dx*(len(word)-1) < n and 0 <= y + dy*(len(word)-1) < n:
            for i in range(len(word)):
                grid[x+dx*i][y+dy*i] = word[i]
            return
