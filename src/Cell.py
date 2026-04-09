from typing import Tuple
from repast4py import core, space

class Cell(core.Agent):
    TYPE = 1 # only type available
    def __init__(self, id, rank, point: space.DiscretePoint):
        super().__init__(id=id, rank=rank, type=Cell.TYPE)
        self.state = 0 # stores the current state
        self.point = point
        self.future_state = 0 # stores the state to apply after the step

    def get_optimal_state(self, grid): # see
        neighbors = self.get_neighbors(grid)
        live_neighbors = sum(n.state for n in neighbors)
        # TODO check again the rules
        if self.state == 1:
            optimal_state = 1 if 2 <= live_neighbors <= 3 else 0
        else:
            optimal_state = 1 if live_neighbors == 3 else 0
        return optimal_state

    def set_next_state(self, grid): # action
        self.future_state = self.get_optimal_state(grid)

    def apply_state(self): # do
        self.state = self.future_state

    def save(self) -> Tuple:
        return (self.uid, self.state, self.point.coordinates)

    def get_neighbors(self, grid):
        candidate = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                nPoint = space.DiscretePoint(self.point.x + i, self.point.y + j)
                candidate.append(grid.get_agent(nPoint))

        return [c for c in candidate if c is not None]
