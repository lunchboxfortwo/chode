"""
Opponent statistics tracker.

Stats are keyed by seat number in memory.  For the human player (seat 0),
stats are persisted to data/tracker/<player_name>.json so the adaptive bot
builds a cumulative profile across sessions.
"""
import json
import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path

TRACKER_DIR = Path(__file__).parent.parent / "data" / "tracker"


@dataclass
class PlayerStats:
    hands_dealt: int = 0
    vpip_count: int = 0
    pfr_count: int = 0
    three_bet_opps: int = 0
    three_bet_count: int = 0
    fold_to_3bet: int = 0
    fold_to_3bet_count: int = 0
    cbet_faced_dry: int = 0
    cbet_fold_dry: int = 0
    cbet_faced_wet: int = 0
    cbet_fold_wet: int = 0
    check_raise_opps: int = 0
    check_raise_count: int = 0
    donk_bet_opps: int = 0
    donk_bet_count: int = 0
    river_bet_count: int = 0
    river_fold_to_bet: int = 0
    river_bet_faced: int = 0
    total_bets_raises: int = 0
    total_calls: int = 0
    total_folds: int = 0
    # Limp tendency
    limp_opps: int = 0
    limp_count: int = 0
    # 3-bet response (call vs fold vs re-raise)
    three_bet_call_opps: int = 0
    call_3bet_count: int = 0
    # Positional VPIP buckets (EP=UTG/UTG+1/UTG+2, MP=LJ/HJ, LP=CO/BTN, blinds=SB/BB)
    vpip_ep: int = 0
    hands_ep: int = 0
    vpip_mp: int = 0
    hands_mp: int = 0
    vpip_lp: int = 0
    hands_lp: int = 0
    vpip_blinds: int = 0
    hands_blinds: int = 0
    # Turn probe (bet turn after flop check-check)
    turn_probe_opps: int = 0
    turn_probe_count: int = 0

    # ── Computed properties ───────────────────────────────────────────────

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
    def fold_to_cbet_dry(self) -> float:
        return self.cbet_fold_dry / self.cbet_faced_dry if self.cbet_faced_dry else 0.50

    @property
    def fold_to_cbet_wet(self) -> float:
        return self.cbet_fold_wet / self.cbet_faced_wet if self.cbet_faced_wet else 0.40

    @property
    def fold_to_cbet_pct(self) -> float:
        total = self.cbet_faced_dry + self.cbet_faced_wet
        if total == 0:
            return 0.45
        return (self.cbet_fold_dry + self.cbet_fold_wet) / total

    @property
    def check_raise_pct(self) -> float:
        return self.check_raise_count / self.check_raise_opps if self.check_raise_opps else 0.08

    @property
    def donk_bet_pct(self) -> float:
        return self.donk_bet_count / self.donk_bet_opps if self.donk_bet_opps else 0.10

    @property
    def river_fold_to_bet_pct(self) -> float:
        return self.river_fold_to_bet / self.river_bet_faced if self.river_bet_faced else 0.50

    @property
    def aggression_factor(self) -> float:
        total = self.total_bets_raises + self.total_calls
        return self.total_bets_raises / total if total else 1.0

    @property
    def limp_pct(self) -> float:
        return self.limp_count / self.limp_opps if self.limp_opps else 0.05

    @property
    def call_3bet_pct(self) -> float:
        return self.call_3bet_count / self.three_bet_call_opps if self.three_bet_call_opps else 0.30

    @property
    def vpip_ep_pct(self) -> float:
        return self.vpip_ep / self.hands_ep if self.hands_ep else 0.15

    @property
    def vpip_mp_pct(self) -> float:
        return self.vpip_mp / self.hands_mp if self.hands_mp else 0.20

    @property
    def vpip_lp_pct(self) -> float:
        return self.vpip_lp / self.hands_lp if self.hands_lp else 0.35

    @property
    def vpip_blinds_pct(self) -> float:
        return self.vpip_blinds / self.hands_blinds if self.hands_blinds else 0.40

    @property
    def positional_awareness(self) -> float:
        """Ratio of LP VPIP to EP VPIP. >2 = normally position-aware, <1.5 = loose EP."""
        if self.hands_ep == 0 or self.hands_lp == 0:
            return 2.0  # insufficient data
        ep = self.vpip_ep_pct
        return self.vpip_lp_pct / ep if ep > 0 else 2.0

    @property
    def turn_probe_pct(self) -> float:
        return self.turn_probe_count / self.turn_probe_opps if self.turn_probe_opps else 0.25

    def profile(self) -> str:
        return (
            f"VPIP={self.vpip:.0%} PFR={self.pfr:.0%} "
            f"3B={self.three_bet_pct:.0%} F3B={self.fold_to_3bet_pct:.0%} "
            f"FCBET={self.fold_to_cbet_pct:.0%} AF={self.aggression_factor:.1f}"
        )

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerStats":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def merge_from(self, other: "PlayerStats"):
        """Add another session's counts into this one (cumulative merge)."""
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


class OpponentTracker:
    def __init__(self):
        TRACKER_DIR.mkdir(parents=True, exist_ok=True)
        self._stats: dict[int, PlayerStats] = {}
        self._names: dict[int, str] = {}   # seat → player name for persistence

    def set_player_name(self, seat: int, name: str):
        """Register a player name for a seat (enables persistence for that seat)."""
        self._names[seat] = name
        self._load_from_disk(seat)

    def get(self, seat: int) -> PlayerStats:
        if seat not in self._stats:
            self._stats[seat] = PlayerStats()
        return self._stats[seat]

    # ── Persistence ───────────────────────────────────────────────────────

    def _path(self, seat: int) -> Path | None:
        name = self._names.get(seat)
        if not name:
            return None
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).lower()
        return TRACKER_DIR / f"{safe}.json"

    def _load_from_disk(self, seat: int):
        path = self._path(seat)
        if path is None or not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            saved = PlayerStats.from_dict(data)
            if seat not in self._stats:
                self._stats[seat] = saved
            else:
                # Merge: saved stats are a prior session baseline
                self._stats[seat].merge_from(saved)
        except Exception:
            pass

    def save(self, seat: int):
        """Persist one seat's stats to disk."""
        path = self._path(seat)
        if path is None:
            return
        try:
            with open(path, "w") as f:
                json.dump(self.get(seat).to_dict(), f, indent=2)
        except Exception:
            pass

    def save_all(self):
        for seat in self._names:
            self.save(seat)

    def stats_dict(self, seat: int) -> dict:
        """Return computed stats as a plain dict for API/UI consumption."""
        s = self.get(seat)
        return {
            "hands_dealt": s.hands_dealt,
            "vpip": round(s.vpip, 4),
            "pfr": round(s.pfr, 4),
            "three_bet_pct": round(s.three_bet_pct, 4),
            "fold_to_3bet_pct": round(s.fold_to_3bet_pct, 4),
            "fold_to_cbet_dry": round(s.fold_to_cbet_dry, 4),
            "fold_to_cbet_wet": round(s.fold_to_cbet_wet, 4),
            "check_raise_pct": round(s.check_raise_pct, 4),
            "river_fold_to_bet_pct": round(s.river_fold_to_bet_pct, 4),
            "aggression_factor": round(s.aggression_factor, 3),
            "limp_pct": round(s.limp_pct, 4),
            "call_3bet_pct": round(s.call_3bet_pct, 4),
            "vpip_ep": round(s.vpip_ep_pct, 4),
            "vpip_mp": round(s.vpip_mp_pct, 4),
            "vpip_lp": round(s.vpip_lp_pct, 4),
            "vpip_blinds": round(s.vpip_blinds_pct, 4),
            "positional_awareness": round(s.positional_awareness, 2),
            "turn_probe_pct": round(s.turn_probe_pct, 4),
        }

    # ── Recording methods ─────────────────────────────────────────────────

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

    def record_cbet_faced(self, seat: int, texture: str = "neutral"):
        s = self.get(seat)
        if texture == "dry":
            s.cbet_faced_dry += 1
        else:
            s.cbet_faced_wet += 1

    def record_fold_to_cbet(self, seat: int, texture: str = "neutral"):
        s = self.get(seat)
        if texture == "dry":
            s.cbet_fold_dry += 1
        else:
            s.cbet_fold_wet += 1

    def record_check_raise_opp(self, seat: int):
        self.get(seat).check_raise_opps += 1

    def record_check_raise(self, seat: int):
        self.get(seat).check_raise_count += 1

    def record_donk_bet_opp(self, seat: int):
        self.get(seat).donk_bet_opps += 1

    def record_donk_bet(self, seat: int):
        self.get(seat).donk_bet_count += 1

    def record_river_bet(self, seat: int):
        self.get(seat).river_bet_count += 1

    def record_river_bet_faced(self, seat: int):
        self.get(seat).river_bet_faced += 1

    def record_river_fold_to_bet(self, seat: int):
        self.get(seat).river_fold_to_bet += 1

    def record_action(self, seat: int, action: str):
        s = self.get(seat)
        if action in ("raise", "bet"):
            s.total_bets_raises += 1
        elif action == "call":
            s.total_calls += 1
        elif action == "fold":
            s.total_folds += 1

    def record_limp_opp(self, seat: int):
        self.get(seat).limp_opps += 1

    def record_limp(self, seat: int):
        self.get(seat).limp_count += 1

    def record_3bet_call_opp(self, seat: int):
        self.get(seat).three_bet_call_opps += 1

    def record_call_3bet(self, seat: int):
        self.get(seat).call_3bet_count += 1

    def record_hand_dealt_pos(self, seat: int, pos_bucket: str):
        """Increment hands_<bucket> for positional VPIP tracking."""
        s = self.get(seat)
        if pos_bucket == "ep":
            s.hands_ep += 1
        elif pos_bucket == "mp":
            s.hands_mp += 1
        elif pos_bucket == "lp":
            s.hands_lp += 1
        else:
            s.hands_blinds += 1

    def record_vpip_pos(self, seat: int, pos_bucket: str):
        """Increment vpip_<bucket> for positional VPIP tracking."""
        s = self.get(seat)
        if pos_bucket == "ep":
            s.vpip_ep += 1
        elif pos_bucket == "mp":
            s.vpip_mp += 1
        elif pos_bucket == "lp":
            s.vpip_lp += 1
        else:
            s.vpip_blinds += 1

    def record_turn_probe_opp(self, seat: int):
        self.get(seat).turn_probe_opps += 1

    def record_turn_probe(self, seat: int):
        self.get(seat).turn_probe_count += 1
