from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Action:
    type: str   # 'fold', 'call', 'check', 'raise', 'bet', 'allin'
    amount: int = 0

    def __str__(self):
        if self.amount:
            return f"{self.type} {self.amount}"
        return self.type


class BaseBot(ABC):
    def __init__(self, name: str, seat: int):
        self.name = name
        self.seat = seat
        self.hole_cards: list[str] = []

    @abstractmethod
    def decide_preflop(
        self,
        position: str,
        stack: int,
        pot: int,
        to_call: int,
        facing_raise: bool,
        raise_position: str,
        last_raise: int,
        bb: int,
    ) -> Action:
        ...

    @abstractmethod
    def decide_postflop(
        self,
        board: list[str],
        position: str,
        stack: int,
        pot: int,
        to_call: int,
        is_first_to_act: bool,
    ) -> Action:
        ...

    def set_cards(self, cards: list[str]):
        self.hole_cards = cards

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"
