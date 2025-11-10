from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .scoring_helper import get_all_scores


def all_dice_states():  # noqa: ANN201
    """Generate all possible dice states (5 dice with values 1-6)."""
    for i in range(1, 7):
        for j in range(1, 7):
            for k in range(1, 7):
                for l in range(1, 7):  # noqa: E741
                    for m in range(1, 7):
                        dice = np.array([i, j, k, l, m], dtype=np.int_)
                        yield dice


def all_dice_masks():  # noqa: ANN201
    """Generate all possible dice hold masks (5 dice, each can be held or not)."""
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                for l in [0, 1]:  # noqa: E741
                    for m in [0, 1]:
                        mask = np.array([i, j, k, l, m], dtype=bool)
                        yield mask


@dataclass
class State:  # noqa: D101
    dice: NDArray[np.int_]
    """Current dice values as numpy array of shape (5,)"""

    @property
    def hash(self) -> str:
        """Hash the sorted dice as string '12345'."""
        return "".join(map(str, self.dice))


lookup: dict[str, "Node"] = {}


class Node:
    """Expectimax tree node."""

    def __init__(self, state: State, action: NDArray[np.int_]):
        self.state: State = state
        self.action: NDArray[np.int_] = action
        self.children: dict[str, Node] = {}

    def get_max_scoring_action(self, open_scores: NDArray[np.int_]) -> tuple[int, int]:
        """Get the scoring action that yields the maximum score."""
        d = self.state.dice
        potential_scores, _ = get_all_scores(d, open_scores)
        best = int(np.argmax(potential_scores))
        return best, potential_scores[best]

    def add_children(self) -> None:
        """Add child nodes to the current node based on possible actions."""
        if len(self.children) > 0:
            pass  # Children already added

        for dice in all_dice_states():
            for action in all_dice_masks():
                new_dice = action * dice + (1 - action) * self.state.dice
                new_state = State(dice=new_dice)

                if new_state.hash in lookup:
                    self.children[new_state.hash] = lookup[new_state.hash]
                else:
                    child_node = Node(state=new_state, action=action)
                    self.children[new_state.hash] = child_node
                    lookup[new_state.hash] = child_node

    # Generate children nodes (this is a placeholder; actual implementation would depend on game rules)
    pass
