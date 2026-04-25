"""Rich-based terminal display for Omega Poker."""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.prompt import Prompt
from rich.columns import Columns

console = Console()

SUIT_COLOR = {"h": "red", "d": "red", "c": "green", "s": "white"}
SUIT_SYMBOL = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}


def _fmt_card(card: str) -> Text:
    if not card or card == "?":
        return Text("[??]", style="dim")
    rank, suit = card[0].upper(), card[1].lower()
    symbol = SUIT_SYMBOL.get(suit, suit)
    color = SUIT_COLOR.get(suit, "white")
    return Text(f"[{rank}{symbol}]", style=f"bold {color}")


def _cards_str(cards: list[str]) -> Text:
    result = Text()
    for c in cards:
        result.append_text(_fmt_card(c))
        result.append(" ")
    return result


def show_table(
    seats: list[dict],   # [{name, stack, bet, position, folded, is_human, cards}]
    board: list[str],
    pot: int,
    human_cards: list[str],
    street: str,
    hand_num: int,
):
    console.clear()
    console.rule(f"[bold cyan] Omega Poker — Hand #{hand_num} — {street} [/bold cyan]")

    # Board
    board_text = Text()
    for c in board:
        board_text.append_text(_fmt_card(c))
        board_text.append(" ")
    if not board:
        board_text = Text("(no board yet)", style="dim")

    console.print(Panel(board_text, title=f"[yellow]Board[/yellow]   Pot: [bold green]${pot:,}[/bold green]", expand=False))

    # Seats table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Seat", justify="center", width=6)
    table.add_column("Player", width=20)
    table.add_column("Pos", width=6)
    table.add_column("Stack", justify="right", width=10)
    table.add_column("Bet", justify="right", width=8)
    table.add_column("Status", width=14)

    for s in seats:
        name = s["name"]
        stack = f"${s['stack']:,}"
        bet = f"${s['current_bet']:,}" if s.get("current_bet") else "-"
        pos = s.get("position", "")
        folded = s.get("folded", False)
        is_human = s.get("is_human", False)

        if folded:
            status = Text("FOLDED", style="dim")
        elif is_human:
            status = Text("← YOU", style="bold cyan")
        else:
            status = Text("", style="")

        row_style = "bold cyan" if is_human else ("dim" if folded else "")
        table.add_row(str(s["seat"]), name, pos, stack, bet, status, style=row_style)

    console.print(table)

    # Human cards
    if human_cards:
        card_display = Text("Your hand: ")
        for c in human_cards:
            card_display.append_text(_fmt_card(c))
            card_display.append(" ")
        console.print(Panel(card_display, expand=False))


def show_action(player: str, action: str, amount: int = 0):
    color = {
        "fold": "red", "call": "yellow", "check": "dim",
        "raise": "bold green", "bet": "bold green", "allin": "bold magenta",
    }.get(action.lower(), "white")
    amt_str = f" ${amount:,}" if amount else ""
    console.print(f"  [dim]{player}[/dim] → [{color}]{action.upper()}{amt_str}[/{color}]")


def show_winner(player: str, amount: int, hand_desc: str = ""):
    desc = f" ({hand_desc})" if hand_desc else ""
    console.print(Panel(
        f"[bold green]{player}[/bold green] wins [bold yellow]${amount:,}[/bold yellow]{desc}",
        title="[bold]Winner[/bold]",
        expand=False,
    ))


def show_bust(player: str):
    console.print(f"[bold red]{player} has busted out![/bold red]")


def show_stats(tracker, human_seat: int):
    s = tracker.get(human_seat)
    console.print(Panel(
        f"[cyan]Your stats[/cyan]  {s.profile()}",
        title="Session Stats",
        expand=False,
    ))


def get_human_action(to_call: int, pot: int, stack: int, can_check: bool, min_raise: int) -> tuple[str, int]:
    """Prompt human for action. Returns (action_type, amount)."""
    options = []
    if to_call == 0:
        options.append("(C)heck")
    else:
        options.append(f"(C)all ${to_call:,}")
    options.append("(R)aise")
    options.append("(F)old")

    console.print(f"\n[bold]Pot:[/bold] ${pot:,}  [bold]Stack:[/bold] ${stack:,}  [bold]To call:[/bold] ${to_call:,}")
    console.print("[dim]" + "  ".join(options) + "[/dim]")

    while True:
        raw = Prompt.ask("Action", default="c").strip().lower()

        if raw in ("f", "fold"):
            return "fold", 0

        if raw in ("c", "call", "check"):
            if to_call == 0:
                return "check", 0
            if to_call >= stack:
                return "allin", stack
            return "call", to_call

        if raw in ("r", "raise", "bet"):
            default_raise = min(pot, stack)
            while True:
                amt_str = Prompt.ask(
                    f"Raise to (min ${min_raise:,}, stack ${stack:,})",
                    default=str(default_raise),
                )
                try:
                    amt = int(amt_str.replace(",", "").replace("$", ""))
                    if amt >= min_raise and amt <= stack:
                        return "raise", amt
                    if amt >= stack:
                        return "allin", stack
                    console.print(f"[red]Amount must be between ${min_raise:,} and ${stack:,}[/red]")
                except ValueError:
                    console.print("[red]Enter a number[/red]")

        console.print("[red]Enter F, C, or R[/red]")


def show_message(msg: str, style: str = ""):
    console.print(msg, style=style)


def show_separator():
    console.rule(style="dim")


def pause(msg: str = "Press Enter to continue..."):
    Prompt.ask(f"[dim]{msg}[/dim]", default="")
