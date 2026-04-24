"""Writes hand history in PokerTracker/HoldemManager compatible format."""
import datetime
from pathlib import Path
from config import HISTORY_DIR


class HandHistoryWriter:
    def __init__(self):
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = HISTORY_DIR / f"session_{ts}.txt"
        self._hand_num = 0
        self._lines: list[str] = []
        self._file = open(self._path, "w")

    def begin_hand(self, hand_num: int, button_seat: int, stacks: dict[str, int], blinds=(50, 100)):
        self._hand_num = hand_num
        ts = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S ET")
        sb, bb = blinds
        self._lines = [
            f"PokerStars Hand #{hand_num}: Hold'em No Limit (${sb}/${bb}) - {ts}",
            f"Table 'OmegaPoker' 6-max Seat #{button_seat + 1} is the button",
        ]
        for name, stack in stacks.items():
            self._lines.append(f"Seat {list(stacks).index(name) + 1}: {name} (${stack} in chips)")

    def post_blinds(self, sb_name: str, bb_name: str, sb: int, bb: int):
        self._lines += [
            f"{sb_name}: posts small blind ${sb}",
            f"{bb_name}: posts big blind ${bb}",
        ]

    def hole_cards(self, human_name: str, c1: str, c2: str):
        self._lines += ["*** HOLE CARDS ***", f"Dealt to {human_name} [{c1} {c2}]"]

    def action(self, player: str, action: str, amount: int = 0):
        if amount:
            self._lines.append(f"{player}: {action} ${amount}")
        else:
            self._lines.append(f"{player}: {action}")

    def street(self, street_name: str, cards: list[str]):
        card_str = " ".join(cards)
        self._lines.append(f"*** {street_name.upper()} *** [{card_str}]")

    def showdown(self, player: str, cards: list[str], hand_desc: str):
        self._lines.append(f"{player}: shows [{' '.join(cards)}] ({hand_desc})")

    def collected(self, player: str, amount: int):
        self._lines.append(f"{player}: collected ${amount} from pot")

    def summary(self, pot: int, board: list[str], winners: list[tuple[str, int]]):
        board_str = f"Board [{' '.join(board)}]" if board else ""
        self._lines += [
            "*** SUMMARY ***",
            f"Total pot ${pot}",
        ]
        if board_str:
            self._lines.append(board_str)
        for name, amount in winners:
            self._lines.append(f"{name} collected ${amount} from pot")
        self._lines.append("")

    def flush_hand(self):
        self._file.write("\n".join(self._lines) + "\n\n")
        self._file.flush()
        self._lines = []

    def close(self):
        self._file.close()

    @property
    def path(self) -> Path:
        return self._path
