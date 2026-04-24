from dataclasses import dataclass, field


@dataclass
class PlayerStats:
    hands_dealt: int = 0
    vpip_count: int = 0       # voluntarily put $ in preflop
    pfr_count: int = 0        # preflop raise
    three_bet_opps: int = 0
    three_bet_count: int = 0
    fold_to_3bet: int = 0
    fold_to_3bet_count: int = 0
    cbet_faced: int = 0
    fold_to_cbet: int = 0
    total_bets_raises: int = 0
    total_calls: int = 0
    total_folds: int = 0

    @property
    def vpip(self) -> float:
        return self.vpip_count / self.hands_dealt if self.hands_dealt else 0.25

    @property
    def pfr(self) -> float:
        return self.pfr_count / self.hands_dealt if self.hands_dealt else 0.15

    @property
    def three_bet_pct(self) -> float:
        return self.three_bet_count / self.three_bet_opps if self.three_bet_opps else 0.06

    @property
    def fold_to_3bet_pct(self) -> float:
        return self.fold_to_3bet / self.fold_to_3bet_count if self.fold_to_3bet_count else 0.55

    @property
    def fold_to_cbet_pct(self) -> float:
        return self.fold_to_cbet / self.cbet_faced if self.cbet_faced else 0.45

    @property
    def aggression_factor(self) -> float:
        total = self.total_bets_raises + self.total_calls
        return self.total_bets_raises / total if total else 1.0

    def profile(self) -> str:
        return (
            f"VPIP={self.vpip:.0%} PFR={self.pfr:.0%} "
            f"3B={self.three_bet_pct:.0%} F3B={self.fold_to_3bet_pct:.0%} "
            f"AF={self.aggression_factor:.1f}"
        )


class OpponentTracker:
    def __init__(self):
        self._stats: dict[int, PlayerStats] = {}

    def get(self, seat: int) -> PlayerStats:
        if seat not in self._stats:
            self._stats[seat] = PlayerStats()
        return self._stats[seat]

    def record_hand_dealt(self, seat: int):
        self.get(seat).hands_dealt += 1

    def record_vpip(self, seat: int):
        self.get(seat).vpip_count += 1

    def record_pfr(self, seat: int):
        self.get(seat).pfr_count += 1

    def record_3bet_opportunity(self, seat: int):
        self.get(seat).three_bet_opps += 1

    def record_3bet(self, seat: int):
        self.get(seat).three_bet_count += 1

    def record_fold_to_3bet_opportunity(self, seat: int):
        self.get(seat).fold_to_3bet_count += 1

    def record_fold_to_3bet(self, seat: int):
        self.get(seat).fold_to_3bet += 1

    def record_cbet_faced(self, seat: int):
        self.get(seat).cbet_faced += 1

    def record_fold_to_cbet(self, seat: int):
        self.get(seat).fold_to_cbet += 1

    def record_action(self, seat: int, action: str):
        s = self.get(seat)
        if action in ("raise", "bet"):
            s.total_bets_raises += 1
        elif action == "call":
            s.total_calls += 1
        elif action == "fold":
            s.total_folds += 1
