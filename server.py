"""
Chode Poker — FastAPI WebSocket server.
Each WebSocket connection gets its own isolated GameSession with its own
PokerGame instance. Multiple users can play simultaneously without interference.

NN-only: tabular solver preloading removed. Games start immediately using
neural preflop and postflop strategy networks.
"""
import asyncio
import json
import logging
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chode")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engine.game import PokerGame
from config import HISTORY_DIR, DEFAULT_BOT_CONFIGS, BOT_PROFILES
from strategy.preflop_nn import PreflopNN, nn_status as preflop_nn_status_mod
from strategy.postflop_nn import PostflopNN, nn_status as postflop_nn_status_mod

app = FastAPI(title="Chode Poker")


# ─── Per-connection session ───────────────────────────────────────────────────

class GameSession:
    def __init__(self, ws: WebSocket, loop: asyncio.AbstractEventLoop):
        self.ws = ws
        self.loop = loop
        self.game: PokerGame | None = None
        self.game_id: int = 0
        self.game_thread: threading.Thread | None = None
        self.next_hand_event = threading.Event()
        self.between_hands: bool = False

    def _send(self, msg: dict):
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.ws.send_text(json.dumps(msg)), self.loop
            )

    def event_cb(self, event: str, data: dict):
        self._send({"event": event, "data": data})

    def start_game(self, human_name: str, bot_configs: list, bot_delay: float, god_mode: bool):
        self.game_id += 1
        self.game = PokerGame(
            human_name=human_name,
            event_cb=self.event_cb,
            bot_configs=bot_configs,
            bot_delay=bot_delay,
        )
        self.game.god_mode = god_mode
        self.between_hands = False
        self.next_hand_event.clear()
        self.game_thread = threading.Thread(
            target=self._run_loop, args=(self.game_id,), daemon=True
        )
        self.game_thread.start()

    def restart(self):
        self.game_id += 1
        self.between_hands = False
        for s in self.game.seats:
            s.stack = 10_000
        self.next_hand_event.clear()
        self.game_thread = threading.Thread(
            target=self._run_loop, args=(self.game_id,), daemon=True
        )
        self.game_thread.start()

    def _run_loop(self, my_id: int):
        try:
            first_hand = True
            while self.game_id == my_id:
                if self.game is None:
                    break
                gs = self.game.game_state()
                if any(s["stack"] == 0 and s["is_human"] for s in gs["seats"]):
                    break
                if not any(not s["is_human"] and s["stack"] > 0 for s in gs["seats"]):
                    break

                if not first_hand:
                    self.next_hand_event.clear()
                    self.next_hand_event.wait(timeout=300)
                    self.next_hand_event.clear()
                    self.between_hands = False
                    if self.game_id != my_id or self.game is None:
                        break
                first_hand = False

                self.game.start_hand()

                if self.game_id != my_id or self.game is None:
                    break

                gs2 = self.game.game_state()
                if (any(s["stack"] == 0 and s["is_human"] for s in gs2["seats"])
                        or not any(not s["is_human"] and s["stack"] > 0 for s in gs2["seats"])):
                    break
                self.between_hands = True
                self._send({"event": "waiting_next_hand", "data": {}})

        except Exception as exc:
            import traceback as _tb
            tb = _tb.format_exc()
            logger.error("Game loop exception:\n%s", tb)
            if self.game_id == my_id:
                self._send({
                    "event": "game_error",
                    "data": {"message": str(exc), "detail": tb[-1500:]},
                })

    def stop(self):
        """Invalidate the running loop so it exits on next iteration."""
        self.game_id = -1
        self.next_hand_event.set()


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Chode server ready — NN strategy loaded on-demand")


# ─── REST API ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()


@app.get("/api/bot-profiles")
async def bot_profiles():
    return JSONResponse(BOT_PROFILES)


@app.get("/api/history")
async def list_history():
    files = sorted(HISTORY_DIR.glob("session_*.txt"), reverse=True)
    result = []
    for f in files[:50]:
        stat = f.stat()
        result.append({
            "filename": f.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })
    return JSONResponse(result)


@app.get("/api/history/{filename}")
async def get_history(filename: str):
    if "/" in filename or ".." in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    path = HISTORY_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"filename": filename, "content": path.read_text()})


# ─── Chart & Progress pages ──────────────────────────────────────────────────

@app.get("/charts", response_class=HTMLResponse)
async def charts_page():
    html_path = Path(__file__).parent / "static" / "charts.html"
    return html_path.read_text()


@app.get("/progress", response_class=HTMLResponse)
async def progress_page():
    html_path = Path(__file__).parent / "static" / "progress.html"
    return html_path.read_text()


@app.get("/api/progress")
async def progress_api():
    from strategy import training_progress
    import glob
    ckpts = sorted(glob.glob("data/preflop_nn/preflop_nn_*.pt") + glob.glob("data/postflop_nn/postflop_nn_*.pt"))
    ckpts = [c.split("/")[-1] for c in ckpts if "_optimizer" not in c]
    return JSONResponse({
        "jobs": training_progress.all_jobs(),
        "system": training_progress.system_state(),
        "checkpoints": ckpts[-10:],
        "timestamp": time.time(),
    })


# ─── Neural preflop API ─────────────────────────────────────────────────────

_nn_model: PreflopNN | None = None
_postflop_nn_model: PostflopNN | None = None


def _get_nn() -> PreflopNN:
    global _nn_model
    if _nn_model is None:
        _nn_model = PreflopNN()
    return _nn_model


def _get_postflop_nn() -> PostflopNN:
    global _postflop_nn_model
    if _postflop_nn_model is None:
        _postflop_nn_model = PostflopNN()
    return _postflop_nn_model


@app.get("/api/preflop/nn/status")
async def preflop_nn_status():
    """Return NN model status and metadata."""
    return JSONResponse(preflop_nn_status_mod())


@app.get("/api/preflop/nn/chart")
async def preflop_nn_chart(n: int = 5, bb: float = 30, pidx: int = 0, hist: str = ""):
    """Query the neural preflop strategy for a full 13×13 chart grid."""
    history = [a for a in hist.split(",") if a] if hist else []
    try:
        nn = _get_nn()
        chart = nn.query_chart(n, bb, pidx, history)
        return JSONResponse(chart)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/preflop/nn/query")
async def preflop_nn_query(
    n: int = 5, bb: float = 30, pidx: int = 0,
    c1: int = 0, c2: int = 5, hist: str = "",
):
    """Query the neural preflop strategy for a single hand."""
    history = [a for a in hist.split(",") if a] if hist else []
    try:
        nn = _get_nn()
        probs = nn.query(n, bb, pidx, (c1, c2), history)
        return JSONResponse({"probs": probs, "source": "nn"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── Neural postflop API ─────────────────────────────────────────────────────

@app.get("/api/postflop/nn/status")
async def postflop_nn_status_api():
    """Return postflop NN model status and metadata."""
    return JSONResponse(postflop_nn_status_mod())


@app.get("/api/postflop/nn/query")
async def postflop_nn_query(
    hand_cat: int = 0,
    position: int = 0,
    street: int = 0,
    texture_id: int = 0,
    n_players: int = 2,
    pot_bb: float = 5.0,
    spr: float = 10.0,
    facing_size: float = 0.0,
    agg_actions: int = 0,
    hist: str = "",
):
    """Query the neural postflop strategy for a single spot.

    All parameters match PostflopNN.query() keyword arguments directly.
    """
    action_history = [int(a) for a in hist.split(",") if a] if hist else []
    try:
        nn = _get_postflop_nn()
        probs = nn.query(
            hand_cat=hand_cat,
            position=position,
            street=street,
            texture_id=texture_id,
            n_players=n_players,
            pot_size=pot_bb,
            stack_ratio=spr,
            facing_size=facing_size,
            agg_actions=agg_actions,
            action_history=action_history,
        )
        return JSONResponse({"probs": probs, "source": "postflop_nn"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    session = GameSession(ws, loop)

    try:
        await ws.send_text(json.dumps({
            "event": "solver_status",
            "data": {"ready": True},
        }))
    except WebSocketDisconnect:
        return

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get("cmd")

            try:
                if cmd == "new_game":
                    human_name = msg.get("name", "Hero")
                    raw_bots = msg.get("bot_configs")
                    if isinstance(raw_bots, list) and len(raw_bots) > 0:
                        bot_configs = [
                            {"name": b.get("name", f"Bot{i+1}"), "type": b.get("type", "gto")}
                            for i, b in enumerate(raw_bots)
                        ]
                    else:
                        bot_configs = DEFAULT_BOT_CONFIGS
                    bot_delay = 1.0 if msg.get("bot_delay", True) else 0.0
                    god_mode = bool(msg.get("god_mode", False))
                    session.start_game(human_name, bot_configs, bot_delay, god_mode)

                elif cmd == "action" and session.game:
                    if session.game.waiting_for_human and session.game._human_action_queue is None:
                        session.game.submit_human_action(
                            msg.get("action", "fold"), int(msg.get("amount", 0))
                        )

                elif cmd == "next_hand":
                    if session.between_hands:
                        session.next_hand_event.set()

                elif cmd == "restart" and session.game:
                    session.restart()

            except Exception as exc:
                import traceback as _tb
                logger.error("WS command '%s' error:\n%s", cmd, _tb.format_exc())
                try:
                    await ws.send_text(json.dumps({
                        "event": "game_error",
                        "data": {"message": str(exc), "detail": _tb.format_exc()[-1000:]},
                    }))
                except Exception:
                    pass

    except WebSocketDisconnect:
        session.stop()


# ─── Static files ─────────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
