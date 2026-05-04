#!/usr/bin/env python3
"""
Chode Poker — Rich terminal CLI.
Runs the full game in the main thread; no server or browser needed.
"""
import sys
import time

from rich.prompt import Prompt

import engine.display as disp
from engine.game import PokerGame
from bots.base import Action
from config import BUY_IN


class CLIGame(PokerGame):
    """Subclass that swaps WebSocket I/O for Rich terminal I/O."""

    def _emit(self, event: str, data: dict):
        if event == "hand_start":
            disp.show_separator()
            disp.show_message(
                f"[bold cyan]── Hand #{data['hand_num']} ──[/bold cyan]"
            )

        elif event == "deal":
            self._render()

        elif event == "blinds":
            disp.show_message(
                f"  [dim]{data['sb']} posts SB, {data['bb']} posts BB[/dim]"
            )

        elif event == "street":
            street = data["street"].upper()
            board = data.get("board", [])
            board_str = "  ".join(board) if board else "(none)"
            disp.show_message(f"\n[bold yellow]── {street} ──[/bold yellow]  {board_str}")

        elif event == "action":
            amt = data.get("amount", 0)
            note = data.get("strategy_note", "")
            disp.show_action(data["player"], data["action"], amt, strategy_note=note)

        elif event == "showdown":
            disp.show_message("\n[bold]── SHOWDOWN ──[/bold]")
            for p in data["players"]:
                cards = "  ".join(p["cards"])
                disp.show_message(f"  {p['name']:20s} [{cards}]")
            time.sleep(2)

        elif event == "winner":
            self._render()
            disp.show_winner(data["player"], data["amount"])
            time.sleep(1.5)

        elif event == "rebuy":
            disp.show_message(
                f"[dim]{data['name']} rebuys to ${data['stack']:,}[/dim]"
            )

        elif event == "game_over":
            if data["reason"] == "bust":
                disp.show_message(
                    "\n[bold red]You're out of chips. Game over.[/bold red]"
                )
            else:
                disp.show_message(
                    "\n[bold green]You busted all the bots! You win![/bold green]"
                )

    def _get_human_action(self, seat, preflop: bool) -> Action:
        self._render()
        to_call = max(0, self.to_call - seat.current_bet)
        action_type, amount = disp.get_human_action(
            to_call=to_call,
            pot=self.pot,
            stack=seat.stack,
            can_check=(to_call == 0),
            min_raise=self.min_raise,
        )
        return Action(action_type, amount)

    def _render(self):
        seat_dicts = [s.to_dict() for s in self.seats]
        human_cards = self.seats[0].hole_cards
        disp.show_table(
            seats=seat_dicts,
            board=self.board,
            pot=self.pot,
            human_cards=human_cards,
            street=self.street,
            hand_num=self.hand_num,
        )


def main():
    disp.show_message("\n[bold magenta]♠ CHODE POKER ♦  [dim]CLI Edition[/dim][/bold magenta]")
    disp.show_message("[dim]6-player NLHE · $10,000 buy-in · 50/100 blinds[/dim]\n")
    name = Prompt.ask("Your name", default="Hero")

    game = CLIGame(human_name=name)

    while True:
        seats = game.seats
        human = seats[0]
        bots_alive = any(not s.is_human and s.stack > 0 for s in seats)

        if human.stack == 0:
            choice = Prompt.ask("\nRebuy and play again?", choices=["y", "n"], default="y")
            if choice == "y":
                for s in seats:
                    s.stack = BUY_IN
            else:
                break
        elif not bots_alive:
            break

        game.start_hand()

    disp.show_message("\n[dim]Thanks for playing. Hand histories saved to data/hand_histories/[/dim]")
    game.history.close()


if __name__ == "__main__":
    main()
