from typing import Callable
from karpathy.microgpt import main as kmain


def main(num_training_steps=1000, emit: Callable[[str], None] = lambda emission: print(emission)) -> None:
    kmain(num_training_steps, emit)
