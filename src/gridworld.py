class GridWorld:

    """
    O means passable. X means impassable. Grid counts downs. (Row, column) coordinates
    """

    def __init__(self):
        start = (5, 3)
        self.x = 9
        self.y = 6

        self.grid = [["O" for _ in range(self.x)] for _ in range(self.y)]

        for column in range(8):
            self.make_cell_impassible(3, column)

        self.size = sum([len([x for x in y if x != "X"]) for y in self.grid])

        self.insert_agent(start[0], start[1], 0)

    def make_cell_impassible(self, row, col):
        self.grid[row][col] = "X"

    def insert_agent(self, row, col, reward):
        self.grid[row][col] = GridAgent(reward)

    def __str__(self):
        return '\n'.join(' '.join(str(x) for x in y) for y in self.grid)


class GridAgent:

    def __init__(self, reward):
        self.reward = reward

    def __str__(self):
        return "A"



if __name__ == '__main__':
    gridworld = GridWorld()
    print(gridworld)