#!/usr/bin/env python3
# ~/qdrant/ingestion/watcher.py
"""
Watch ~/Downloads for AI conversation export files and queue them for review.
Runs as a persistent daemon. Started by start.sh.
"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from backend.db import get_connection, init_tables, now_iso

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [watcher] %(message)s")
log = logging.getLogger(__name__)

def wait_for_file_complete(path: Path, stable_secs: float = 3.0,
                            timeout_secs: float = 60.0) -> bool:
    deadline = time.time() + timeout_secs
    last_size, stable_since = -1, None
    while time.time() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last_size:
            if stable_since and (time.time() - stable_since) >= stable_secs:
                return True
        else:
            stable_since = time.time()
        last_size = size
        time.sleep(0.5)
    return False

def enqueue(path: Path, conn) -> None:
    known = conn.execute(
        "SELECT id FROM quarantine WHERE path=?", [str(path)]
    ).fetchone()
    if known:
        return
    if path.stat().st_size > config.MAX_FILE_SIZE_BYTES:
        log.warning(f"Skipping (too large): {path}")
        conn.execute(
            "INSERT INTO quarantine (path, detected_at, status, reason) VALUES (?,?,?,?)",
            [str(path), now_iso(), "rejected", "too_large"]
        )
        conn.commit()
        return
    conn.execute(
        "INSERT INTO quarantine (path, detected_at, status) VALUES (?,?,?)",
        [str(path), now_iso(), "pending"]
    )
    conn.commit()
    log.info(f"Queued for review: {path.name}")

def startup_scan(conn) -> None:
    """Enqueue any .json files in Downloads not already in quarantine."""
    watch_dir = Path(config.WATCHER_WATCH_DIR)
    if not watch_dir.exists():
        return
    known = {r["path"] for r in conn.execute("SELECT path FROM quarantine").fetchall()}
    for f in watch_dir.glob("*.json"):
        if str(f) not in known:
            log.info(f"Startup scan found: {f.name}")
            if wait_for_file_complete(f):
                enqueue(f, conn)

def process_approved(conn) -> None:
    """Trigger ingestion for any quarantine rows newly approved via the API."""
    import subprocess
    rows = conn.execute(
        "SELECT id, path FROM quarantine WHERE status='approved'"
    ).fetchall()
    for row in rows:
        log.info(f"Ingesting approved file: {row['path']}")
        try:
            subprocess.Popen(
                [sys.executable, str(Path(__file__).parent / "ingest_convos.py"), row["path"]],
                cwd=str(Path(__file__).parent.parent),
            )
            conn.execute(
                "UPDATE quarantine SET status='ingesting' WHERE id=?", [row["id"]]
            )
            conn.commit()
        except Exception as exc:
            log.error(f"Failed to start ingest: {exc}")

def main():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    conn = get_connection()
    init_tables(conn)

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() not in config.WATCHER_ALLOWLIST:
                return
            log.info(f"Detected: {path.name} — waiting for complete...")
            if wait_for_file_complete(path):
                enqueue(path, conn)

    log.info(f"Starting watcher on {config.WATCHER_WATCH_DIR}")
    startup_scan(conn)

    observer = Observer()
    observer.schedule(Handler(), config.WATCHER_WATCH_DIR, recursive=False)
    observer.start()
    log.info("Watcher running. Press Ctrl+C to stop.")

    try:
        while True:
            process_approved(conn)
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
