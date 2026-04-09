from src.Cell import Cell
from repast4py import space


class RationalCell(Cell):
    def __init__(self, id, rank, point: space.DiscretePoint):
        super().__init__(id=id, rank=rank, point=point)

    def state_transformer(self, action: int, live_neighbors: int) -> str:
        """
        tau: Ac -> Omega
        Simulates the environment to predict the outcome.
        """
        if (action == 1 and (live_neighbors == 3 or (self.state == 1 and live_neighbors == 2)))\
                or (action == 0 and (live_neighbors < 2 or live_neighbors > 3)):
            return "correct"
        else:
            return "violation"

    def utility_function(self, outcome: str) -> float:
        """
        u: Omega -> R
        Assigns a real number indicating how 'good' the outcome is for the agent.
        """
        if outcome == "correct":
            return 1.0
        elif outcome == "violation":
            return -1.0
        return 0.0

    def agent_function(self, grid) -> int:
        """
        Ag: R^E -> Ac
        The decision-making process based on preference orderings.
        """
        neighbors = self.get_neighbors(grid)
        live_neighbors = sum(n.state for n in neighbors)

        best_action = None
        max_utility = float('-inf')
        for action in [0, 1]:
            outcome = self.state_transformer(action, live_neighbors)
            utility = self.utility_function(outcome)
            if utility > max_utility:
                max_utility = utility
                best_action = action
        return best_action

    def set_next_state(self, grid):
        self.future_state = self.agent_function(grid)