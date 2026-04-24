"""
Omega Poker — FastAPI WebSocket server.
Runs the game engine in a background thread; pushes state to all
connected browser clients via WebSocket.
"""
import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from engine.game import PokerGame

app = FastAPI(title="Omega Poker")

# ─── Game state ──────────────────────────────────────────────────────────────

_game: PokerGame | None = None
_clients: list[WebSocket] = []
_loop: asyncio.AbstractEventLoop | None = None
_game_thread: threading.Thread | None = None


async def _broadcast(msg: dict):
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.remove(ws)


def _sync_broadcast(msg: dict):
    """Called from game thread → schedule coroutine on the asyncio loop."""
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(msg), _loop)


def _event_cb(event: str, data: dict):
    _sync_broadcast({"event": event, "data": data})


def _run_game_loop():
    """Runs the full game in a background thread until game over."""
    global _game
    while True:
        if _game is None:
            break
        gs = _game.game_state()
        # Check game over
        if any(s["stack"] == 0 and s["is_human"] for s in gs["seats"]):
            break
        bots_alive = any(not s["is_human"] and s["stack"] > 0 for s in gs["seats"])
        if not bots_alive:
            break
        _game.start_hand()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _loop, _game, _game_thread
    await ws.accept()
    _clients.append(ws)
    _loop = asyncio.get_event_loop()

    # Send current state immediately
    if _game:
        await ws.send_text(json.dumps({"event": "state", "data": _game.game_state()}))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get("cmd")

            if cmd == "new_game":
                human_name = msg.get("name", "Hero")
                _game = PokerGame(human_name=human_name, event_cb=_event_cb)
                _game_thread = threading.Thread(target=_run_game_loop, daemon=True)
                _game_thread.start()

            elif cmd == "action" and _game:
                action_type = msg.get("action", "fold")
                amount = int(msg.get("amount", 0))
                _game.submit_human_action(action_type, amount)

            elif cmd == "restart" and _game:
                # Reset all stacks and restart
                for s in _game.seats:
                    s.stack = 10_000
                _game_thread = threading.Thread(target=_run_game_loop, daemon=True)
                _game_thread.start()

    except WebSocketDisconnect:
        _clients.remove(ws)


# ─── Static files ─────────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
