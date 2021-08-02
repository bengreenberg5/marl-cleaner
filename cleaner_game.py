

class CleanerGame:

    MOVES = {
        0: ( 0,  0),
        1: (-1,  0),
        2: ( 1,  0),
        3: ( 0, -1),
        4: ( 0,  1),
    }


    def __init__(
        self,
        size,
        contents,
        num_agents,
        agent_pos,
    ):
        pass  # TODO

    def __repr__(self):
        return f"CleanerGame({self.size}, {self.num_agents} agents"

    def render(self, fig=None, ax=None):
        pass  # TODO

    def step(self, actions):
        pass  # TODO
