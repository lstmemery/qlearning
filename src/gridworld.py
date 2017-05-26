class Gridworld:

    """
    O means passable. X means impassable. Grid counts downs. (Row, column) coordinates
    """

    def __init__(self):
        self.grid = [["O" for _ in range(9)] for _ in range(6)]

        for column in range(8):
            self.make_cell_impassible(3, column)

        self.size = sum([len([x for x in y if x != "X"]) for y in self.grid])

    def make_cell_impassible(self, row, col):
        self.grid[row][col] = "X"

    def __str__(self):
        return '\n'.join(' '.join(x for x in y) for y in self.grid)



if __name__ == '__main__':
    gridworld = Gridworld()
    print(gridworld)