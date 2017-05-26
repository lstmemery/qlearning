from collections import namedtuple

location = namedtuple("location", ["x", "y"])


class GridWorld:
    """
    O means passable. X means impassable. Grid counts downs. (Row, column) coordinates
    """

    def __init__(self):
        self.start = location(5, 3)
        self.x = 9
        self.y = 6

        self.grid = [["O" for _ in range(self.x)] for _ in range(self.y)]

        for column in range(8):
            self.make_cell_impassible(3, column)

        self.size = sum([len([x for x in y if x != "X"]) for y in self.grid])

        self.agent = GridAgent(self, self.start.x, self.start.y, 0)
        self.goal = location(0, 8)
        self.grid[self.goal.x][self.goal.y] = "G"

    def make_cell_impassible(self, row, col):
        self.grid[row][col] = "X"

    def is_impassible(self, row, col):
        return self.grid[row][col] == "X"

    def __str__(self):
        return '\n'.join(' '.join(str(x) for x in y) for y in self.grid)



class GridAgent:
    def __init__(self, gridworld, row, column, reward):
        self.gridworld = gridworld
        self.row = row
        self.column = column
        self.reward = reward

        self.gridworld.grid[self.row][self.column] = self

    def move(self, direction):
        assert direction in ("up", "down", "left", "right"), "Can only move in the 4 cardinal directions"

        previous = location(self.row, self.column)

        if direction == "left":
            if self.column > 0 and \
                    not self.gridworld.is_impassible(self.row, self.column - 1):
                self.column -= 1

        elif direction == "right":
            if self.column < self.gridworld.x - 1 and \
                    not self.gridworld.is_impassible(self.row, self.column + 1):
                self.column += 1

        elif direction == "up":
            if self.row > 0 and \
                    not self.gridworld.is_impassible(self.row - 1, self.column):
                self.row -= 1

        elif direction == "down":
            if self.row < self.gridworld.y - 1 and \
                    not self.gridworld.is_impassible(self.row + 1, self.column):
                self.row += 1

        self.gridworld.grid[previous.x][previous.y] = "O"

        if location(self.row, self.column) == self.gridworld.goal:
            self.reward += 1
            self.gridworld.grid[self.gridworld.start.x][self.gridworld.start.y] = self
        else:
            self.gridworld.grid[self.row][self.column] = self

    def get_neighbours(self):
        neighbours = []

        if self.column > 0 and \
                not self.gridworld.is_impassible(self.row, self.column - 1):
            neighbours.append(("left", self.row, self.column - 1))

        if self.column < self.gridworld.x - 1 and \
                not self.gridworld.is_impassible(self.row, self.column + 1):
            neighbours.append(("right", self.row, self.column + 1))

        if self.row > 0 and \
                not self.gridworld.is_impassible(self.row - 1, self.column):
            neighbours.append(("up", self.row - 1, self.column))

        if self.row < self.gridworld.y - 1 and \
                not self.gridworld.is_impassible(self.row + 1, self.column):
            neighbours.append(("down", self.row + 1, self.column))

        return neighbours

    def __str__(self):
        return "A"


if __name__ == '__main__':
    gridworld = GridWorld()
    print(gridworld)
    gridworld.agent.move("right")
    gridworld.agent.move("right")
    gridworld.agent.move("right")
    gridworld.agent.move("right")
    gridworld.agent.move("right")
    gridworld.agent.move("up")
    gridworld.agent.move("up")
    gridworld.agent.move("up")
    gridworld.agent.move("up")
    gridworld.agent.move("up")
    print(gridworld)
