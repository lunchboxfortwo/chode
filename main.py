#!/usr/bin/env python3
"""
Chode Poker — CLI entry point.
Starts the FastAPI/WebSocket server and opens the browser UI.
"""
import sys
import os
import subprocess
import webbrowser
import time

PORT = int(os.environ.get("PORT", 8765))


def main():
    print(f"\n  Chode Poker")
    print(f"  Starting server on http://0.0.0.0:{PORT}")
    print(f"  Open your browser to: http://localhost:{PORT}")
    print(f"  Ctrl-C to quit\n")

    try:
        import uvicorn
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=PORT,
            log_level="warning",
            reload=False,
        )
    except ImportError:
        print("uvicorn not installed. Run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
