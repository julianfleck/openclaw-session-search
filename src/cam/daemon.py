#!/usr/bin/env python3
"""
CAM Daemon - Background indexer with warm models.

Watches session directories and indexes new/changed sessions.
Keeps ML models loaded in memory for fast indexing (~200ms per session
instead of ~10s with cold start).

Architecture:
- Single process with models loaded once at startup (~1.3GB memory)
- Priority queue for manual requests (cam index --queue)
- Normal queue for watcher-detected changes
- Processes queues sequentially, priority first
- Models stay warm between indexing operations

Uses watchdog for cross-platform file system monitoring.
"""

import fcntl
import json
import os
import shutil
import socket
import sys
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Set, List, Dict, Tuple
from collections import deque
import logging
import threading

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from cam.search import SearchIndex

# Configuration
IDLE_TIMEOUT = 300  # 5 minutes of no activity before queueing
DEBOUNCE_SECONDS = 5  # Wait for rapid changes to settle
QUEUE_POLL_INTERVAL = 10  # Check queue every 10 seconds when idle
QUEUE_POLL_INTERVAL_ACTIVE = 2  # Check every 2 seconds when actively processing
SESSION_TIMEOUT = 300  # 5 minutes max per session before giving up
SYNC_INTERVAL = 300  # 5 minutes between syncs when idle

# Incremental indexing settings
INCREMENTAL_DEBOUNCE = 10  # 10 seconds quiet before incremental index
FULL_REINDEX_DEBOUNCE = 300  # 5 minutes quiet before full re-index
MIN_NEW_MESSAGES = 6  # Minimum new messages to trigger incremental index


def _clean_env() -> dict:
    """Get environment dict with malloc debugging vars removed.

    macOS prints 'MallocStackLogging: can't turn off...' warnings when
    subprocess inherits malloc debugging environment variables from parent.
    """
    env = os.environ.copy()
    # Remove macOS malloc debugging variables that cause noisy warnings
    for key in list(env.keys()):
        if key.startswith('Malloc') or key.startswith('MallocStack'):
            del env[key]
    return env

# Queue files
QUEUE_DIR = Path.home() / ".cam"
PRIORITY_QUEUE_FILE = QUEUE_DIR / ".index-queue-priority"
NORMAL_QUEUE_FILE = QUEUE_DIR / ".index-queue"
LOCK_FILE = QUEUE_DIR / ".index.lock"
STATE_FILE = QUEUE_DIR / "sessions" / ".indexed_sessions"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('cam-daemon')


# =============================================================================
# Queue Management
# =============================================================================

def queue_add(path: str, priority: bool = False) -> bool:
    """Add a session path to the indexing queue.

    Args:
        path: Absolute path to session JSONL file
        priority: If True, add to priority queue (for manual cam index)

    Returns:
        True if added, False if already queued or skippable
    """
    # Reject skippable subagent types at entry
    if is_skippable_subagent(path):
        return False

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    queue_file = PRIORITY_QUEUE_FILE if priority else NORMAL_QUEUE_FILE

    # Read existing queue
    existing = set()
    if queue_file.exists():
        existing = set(queue_file.read_text().strip().split('\n'))
        existing.discard('')

    # Check if already queued
    if path in existing:
        return False

    # Also check the other queue
    other_file = NORMAL_QUEUE_FILE if priority else PRIORITY_QUEUE_FILE
    if other_file.exists():
        other_queue = set(other_file.read_text().strip().split('\n'))
        if path in other_queue:
            if priority:
                # Move to priority queue
                other_queue.discard(path)
                other_file.write_text('\n'.join(sorted(other_queue)))
            else:
                return False  # Already in priority queue

    # Add to queue
    existing.add(path)
    queue_file.write_text('\n'.join(sorted(existing)))
    return True


def is_skippable_subagent(path: str) -> bool:
    """Check if a session file is a skippable subagent type.

    These are internal Claude Code subagents that produce minimal content:
    - prompt_suggestion: Auto-generated prompt suggestions (2-3 messages)
    - compact: Context compaction summaries (2-3 messages)
    """
    name = Path(path).name.lower()
    return (
        'prompt_suggestion' in name or
        name.startswith('agent-acompact-') or
        name.startswith('agent-aprompt_suggestion-')
    )


def is_subagent_session(path: str) -> bool:
    """Check if a session file is in a subagents directory."""
    return '/subagents/' in path or '\\subagents\\' in path


def queue_pop() -> Optional[Tuple[str, bool]]:
    """Pop the next session to index (priority first, main sessions first, newest first).

    Sessions are sorted by:
    1. Priority queue before normal queue
    2. Main sessions before subagent sessions (to prioritize recent user work)
    3. File modification time (most recent first)

    Checks whether sessions need incremental or full indexing based on debounce.
    Also filters out low-value subagent types (prompt_suggestion, compact).

    Returns:
        Tuple of (path, incremental) where incremental=True means use incremental indexing,
        or None if queues are empty or no sessions are ready.
    """
    indexed = get_indexed_sessions()

    # Check priority queue first
    for queue_file in [PRIORITY_QUEUE_FILE, NORMAL_QUEUE_FILE]:
        if queue_file.exists():
            lines = [l for l in queue_file.read_text().strip().split('\n') if l]
            if lines:
                # Filter out low-value subagent types entirely
                original_count = len(lines)
                lines = [l for l in lines if not is_skippable_subagent(l)]

                # Categorize sessions by what kind of indexing they need
                ready_for_full = []  # quiet >= 5 min
                ready_for_incremental = []  # quiet >= 10s but < 5 min
                not_ready = []  # still active (< 10s since modification)

                for path in lines:
                    if needs_full_reindex(path, indexed):
                        ready_for_full.append(path)
                    elif needs_incremental_index(path, indexed):
                        ready_for_incremental.append(path)
                    else:
                        not_ready.append(path)

                if original_count != len(lines):
                    log.debug(f"Filtered {original_count - len(lines)} skippable sessions")

                # Prefer full reindex over incremental
                ready = ready_for_full + ready_for_incremental

                if not ready:
                    # Nothing ready, keep not_ready in queue for later
                    queue_file.write_text('\n'.join(not_ready))
                    continue

                # Sort by: main sessions first, then by mtime (newest first)
                def sort_key(p):
                    try:
                        mtime = Path(p).stat().st_mtime
                    except (OSError, FileNotFoundError):
                        mtime = 0
                    is_subagent = 1 if is_subagent_session(p) else 0
                    return (is_subagent, -mtime)

                ready.sort(key=sort_key)

                path = ready[0]
                incremental = path in ready_for_incremental

                # Remove selected from queue, keep the rest
                remaining = ready[1:] + not_ready
                queue_file.write_text('\n'.join(remaining))

                return (path, incremental)

    return None


def queue_size() -> tuple[int, int]:
    """Get queue sizes (priority, normal)."""
    priority_size = 0
    normal_size = 0

    if PRIORITY_QUEUE_FILE.exists():
        lines = [l for l in PRIORITY_QUEUE_FILE.read_text().strip().split('\n') if l]
        priority_size = len(lines)

    if NORMAL_QUEUE_FILE.exists():
        lines = [l for l in NORMAL_QUEUE_FILE.read_text().strip().split('\n') if l]
        normal_size = len(lines)

    return priority_size, normal_size


def queue_clear():
    """Clear all queues."""
    if PRIORITY_QUEUE_FILE.exists():
        PRIORITY_QUEUE_FILE.unlink()
    if NORMAL_QUEUE_FILE.exists():
        NORMAL_QUEUE_FILE.unlink()


def queue_clean() -> int:
    """Remove up-to-date and skippable sessions from the queue.

    Returns:
        Number of entries removed
    """
    indexed = get_indexed_sessions()
    total_removed = 0

    for queue_file in [PRIORITY_QUEUE_FILE, NORMAL_QUEUE_FILE]:
        if queue_file.exists():
            lines = [l for l in queue_file.read_text().strip().split('\n') if l]
            original_count = len(lines)
            # Keep only sessions that need (re)indexing and aren't skippable
            lines = [l for l in lines if needs_reindex(l, indexed) and not is_skippable_subagent(l)]
            removed = original_count - len(lines)

            if removed > 0:
                queue_file.write_text('\n'.join(lines))
                total_removed += removed

    return total_removed


def queue_contents() -> List[str]:
    """Get all paths in both queues (priority first)."""
    paths = []
    for queue_file in [PRIORITY_QUEUE_FILE, NORMAL_QUEUE_FILE]:
        if queue_file.exists():
            lines = [l for l in queue_file.read_text().strip().split('\n') if l]
            paths.extend(lines)
    return paths


def queue_stats_by_source() -> Dict[str, int]:
    """Get queue counts grouped by source agent.

    Returns dict like: {"Claude Code": 45, "Cursor": 3}
    """
    paths = queue_contents()
    stats = {}

    for path in paths:
        # Determine source from path
        if "/.claude/" in path:
            source = "Claude Code"
        elif "/.cursor/" in path:
            source = "Cursor"
        elif "/.openclaw/" in path:
            source = "OpenClaw"
        elif "/.codex/" in path:
            source = "Codex"
        else:
            source = "Other"

        stats[source] = stats.get(source, 0) + 1

    return stats


# =============================================================================
# Indexed Sessions State
# =============================================================================

# Cache for indexed sessions to avoid re-parsing 93MB file every 2 seconds
_indexed_sessions_cache: Dict[str, float] = {}
_indexed_sessions_mtime: float = 0.0


def get_indexed_sessions(force_reload: bool = False) -> Dict[str, float]:
    """Get dict of indexed session paths -> mtime when indexed.

    Format in file: path:mtime (one per line)
    Also supports legacy format (path only) for backwards compatibility.

    Uses in-memory cache, only reloads when file changes.
    """
    global _indexed_sessions_cache, _indexed_sessions_mtime

    # Check if file has changed
    try:
        current_mtime = STATE_FILE.stat().st_mtime
    except (OSError, FileNotFoundError):
        return {}

    if not force_reload and current_mtime == _indexed_sessions_mtime and _indexed_sessions_cache:
        return _indexed_sessions_cache

    # Reload from file
    indexed = {}
    if STATE_FILE.exists():
        for line in STATE_FILE.read_text().strip().split('\n'):
            if not line:
                continue
            # Format: path:mtime. Paths may contain ':' (Windows drive, etc.) so split from the right.
            # Historical bug: a line with mtime=0.0 was treated as legacy (path-only) and the writer
            # re-emitted it as `<line>:0.0` each cycle, growing unboundedly. Defensive parse: peel
            # all trailing ':<float>' segments and keep the leftmost (= original) mtime.
            head, mtime, found = line, 0.0, False
            while ':' in head:
                pre, _, tail = head.rpartition(':')
                try:
                    mtime = float(tail)
                    head = pre
                    found = True
                except ValueError:
                    break
            if found:
                indexed[head] = mtime
            else:
                indexed[line] = 0.0

    _indexed_sessions_cache = indexed
    _indexed_sessions_mtime = current_mtime
    return indexed


def invalidate_indexed_sessions_cache():
    """Invalidate the indexed sessions cache after writing."""
    global _indexed_sessions_mtime
    _indexed_sessions_mtime = 0.0


def mark_session_indexed(path: str):
    """Mark a session as indexed with current mtime."""
    global _indexed_sessions_cache, _indexed_sessions_mtime

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    indexed = get_indexed_sessions()

    # Get current mtime
    try:
        mtime = Path(path).stat().st_mtime
    except (OSError, FileNotFoundError):
        mtime = time.time()

    indexed[path] = mtime

    # Write in new format
    lines = [f"{p}:{m}" for p, m in sorted(indexed.items())]
    STATE_FILE.write_text('\n'.join(lines))

    # Update cache in-place instead of invalidating
    _indexed_sessions_cache = indexed
    try:
        _indexed_sessions_mtime = STATE_FILE.stat().st_mtime
    except (OSError, FileNotFoundError):
        _indexed_sessions_mtime = 0.0


def needs_reindex(path: str, indexed: Dict[str, float], debounce_seconds: int = 60) -> bool:
    """Check if a session needs (re)indexing based on mtime.

    For sessions that were already indexed, we only re-index if:
    1. File has been modified since last index
    2. File hasn't been modified for debounce_seconds

    Args:
        path: Session file path
        indexed: Dict of path -> indexed mtime
        debounce_seconds: How long file must be quiet (default 60s)

    Returns:
        True if session needs indexing
    """
    if path not in indexed:
        return True  # Never indexed

    indexed_mtime = indexed[path]
    if indexed_mtime == 0:
        return False  # Legacy entry, don't re-index

    try:
        stat = Path(path).stat()
        current_mtime = stat.st_mtime
        now = time.time()

        # Must be modified since last index (with small tolerance)
        if current_mtime <= indexed_mtime + 1:
            return False  # Not modified

        # Must be "quiet" for debounce period
        if now - current_mtime < debounce_seconds:
            return False  # Still active

        return True
    except (OSError, FileNotFoundError):
        return False


def needs_incremental_index(path: str, indexed: Dict[str, float]) -> bool:
    """Check if a session needs incremental indexing (short debounce).

    Uses INCREMENTAL_DEBOUNCE (10s) instead of full debounce.
    """
    return needs_reindex(path, indexed, debounce_seconds=INCREMENTAL_DEBOUNCE)


def needs_full_reindex(path: str, indexed: Dict[str, float]) -> bool:
    """Check if a session needs full re-indexing (long debounce).

    Uses FULL_REINDEX_DEBOUNCE (5 min) - session has been quiet long enough
    for optimal re-segmentation.
    """
    return needs_reindex(path, indexed, debounce_seconds=FULL_REINDEX_DEBOUNCE)


class SessionWatcher(FileSystemEventHandler):
    """Watch for changes in session directories and queue them for indexing."""

    def __init__(self):
        super().__init__()
        self.last_change: Dict[str, datetime] = {}  # path -> last change time
        self.pending_paths: Set[str] = set()
        self.running = True
        self.indexed = get_indexed_sessions()

    def on_any_event(self, event: FileSystemEvent):
        """Handle any file system event."""
        # Only care about JSONL files (session files)
        if not event.src_path.endswith('.jsonl'):
            return

        # Ignore temporary files
        if '.tmp' in event.src_path or '.swp' in event.src_path:
            return

        # Ignore subagents
        if 'subagents' in event.src_path:
            return

        path = event.src_path
        self.last_change[path] = datetime.now()
        self.pending_paths.add(path)
        log.debug(f"Change detected: {path}")

    def check_and_queue(self):
        """Check pending paths and queue those that have settled."""
        now = datetime.now()
        to_queue = []

        # Refresh indexed sessions to pick up worker updates
        self.indexed = get_indexed_sessions()

        for path in list(self.pending_paths):
            last_change = self.last_change.get(path)
            if not last_change:
                continue

            # Wait for debounce period
            idle_time = (now - last_change).total_seconds()
            if idle_time >= DEBOUNCE_SECONDS:
                # Check if needs (re)indexing based on mtime
                if needs_reindex(path, self.indexed):
                    to_queue.append(path)
                self.pending_paths.discard(path)
                del self.last_change[path]

        # Queue settled paths
        for path in to_queue:
            if queue_add(path, priority=False):
                log.info(f"Queued for indexing: {Path(path).name}")

    def stop(self):
        """Stop the watcher."""
        self.running = False


# =============================================================================
# Index Worker
# =============================================================================

class IndexWorker:
    """Worker that processes the indexing queue with warm models."""

    def __init__(self, output_dir: Path, machine_id: str, sync_repo: Optional[str] = None):
        self.output_dir = output_dir
        self.machine_id = machine_id
        self.sync_repo = sync_repo
        self.running = True
        self.models_loaded = False
        # Initialize search index
        self.search_index = SearchIndex(
            output_dir.parent / "index.sqlite",
            workspace_dir=output_dir
        )

    def load_models(self):
        """Load all ML models into memory."""
        if self.models_loaded:
            return

        from . import providers
        if providers.is_headed():
            # Fail loudly at startup if headed mode is misconfigured, instead
            # of silently producing garbage titles for every indexed session.
            providers.verify_headed_setup()
            log.info(
                "Headed mode active (provider=%s): skipping local ML model load",
                providers.get_provider_key() or "unknown",
            )
            self.models_loaded = True  # mark "done" so we don't keep checking
            return

        log.info("Loading ML models...")
        from . import segment
        segment.preload_models()
        self.models_loaded = True
        log.info("Models loaded and ready")

    def index_session(self, session_path: str, incremental: bool = False) -> bool:
        """Index a single session file with timeout.

        Args:
            session_path: Path to session JSONL file
            incremental: If True, only index new messages since last index

        Returns:
            True if successful, False otherwise
        """
        from . import segment
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        session_file = Path(session_path)
        if not session_file.exists():
            log.warning(f"Session file not found: {session_path}")
            return False

        def do_full_index():
            session_meta, messages = segment.load_session_messages(session_file)

            if len(messages) < 6:
                log.info(f"Skipping {session_file.name} ({len(messages)} messages)")
                return (0, [])  # Skipped, no paths

            sections, _ = segment.segment_session(messages)
            written_paths = segment.write_sections(
                messages, sections, session_meta,
                output_dir=self.output_dir,
                machine_id=self.machine_id
            )
            return (len(sections), written_paths)

        def do_incremental_index():
            num_new, written_paths = segment.incremental_index_session(
                session_path=session_file,
                output_dir=self.output_dir,
                machine_id=self.machine_id,
                min_new_messages=MIN_NEW_MESSAGES
            )
            return (num_new, written_paths)

        try:
            # Run with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                if incremental:
                    future = executor.submit(do_incremental_index)
                else:
                    future = executor.submit(do_full_index)

                try:
                    result_count, written_paths = future.result(timeout=SESSION_TIMEOUT)
                    mark_session_indexed(session_path)

                    if result_count > 0:
                        if incremental:
                            log.info(f"Incremental indexed {session_file.name}: +{result_count} msgs")
                        else:
                            log.info(f"Indexed {session_file.name}: {result_count} sections")

                        # Update search index with new/updated segments
                        if written_paths:
                            self.search_index.index_segments(written_paths)
                    return True

                except FuturesTimeoutError:
                    log.error(f"Timeout indexing {session_file.name} (>{SESSION_TIMEOUT}s)")
                    # Mark as indexed to avoid infinite retry
                    mark_session_indexed(session_path)
                    return False

        except Exception as e:
            log.error(f"Failed to index {session_file.name}: {e}")
            # Mark as indexed to avoid infinite retry
            mark_session_indexed(session_path)
            return False

    def update_search_index(self, segment_paths: List[Path]):
        """Update search index with new segments."""
        if segment_paths:
            count = self.search_index.index_segments(segment_paths)
            log.debug(f"Updated search index: {count} segments")

    def run_loop(self):
        """Main worker loop - processes queue items."""
        log.info("Starting index worker")

        # Load models upfront
        self.load_models()

        # Initial sync to push any pending changes from previous runs
        if self.sync_repo:
            do_sync(self.output_dir, self.sync_repo, self.machine_id)

        sessions_indexed_since_sync = 0
        consecutive_failures = 0
        max_consecutive_failures = 10  # Restart daemon after this many failures
        last_sync = datetime.now()
        was_active = False  # Track if we were processing in the last iteration

        while self.running:
            try:
                # Pop next item from queue
                result = queue_pop()

                if result:
                    session_path, incremental = result
                    # Index the session (search index updated inline)
                    if self.index_session(session_path, incremental=incremental):
                        sessions_indexed_since_sync += 1
                        consecutive_failures = 0  # Reset on success
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            log.error(f"Too many consecutive failures ({consecutive_failures}), restarting daemon...")
                            # Exit to let launchd/systemd restart us fresh
                            self.running = False
                            break
                    was_active = True
                else:
                    # Queue empty - sync if we have new segments or enough time has passed
                    time_since_sync = (datetime.now() - last_sync).total_seconds()
                    should_sync = (
                        self.sync_repo and
                        (sessions_indexed_since_sync > 0 or time_since_sync >= SYNC_INTERVAL)
                    )

                    if should_sync:
                        if do_sync(self.output_dir, self.sync_repo, self.machine_id):
                            sessions_indexed_since_sync = 0
                        last_sync = datetime.now()

                    # Adaptive polling: shorter interval right after activity, longer when idle
                    poll_interval = QUEUE_POLL_INTERVAL_ACTIVE if was_active else QUEUE_POLL_INTERVAL
                    time.sleep(poll_interval)
                    consecutive_failures = 0  # Reset when queue is empty
                    was_active = False

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"Worker error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    log.error(f"Too many consecutive failures ({consecutive_failures}), restarting daemon...")
                    self.running = False
                    break
                time.sleep(10)

        # Final sync
        if self.sync_repo and sessions_indexed_since_sync > 0:
            do_sync(self.output_dir, self.sync_repo, self.machine_id)

        log.info("Index worker stopped")

    def stop(self):
        """Stop the worker."""
        self.running = False


def do_sync(workspace_dir: Path, sync_repo: str, machine_id: str) -> bool:
    """Sync segments with remote repository.

    Returns:
        True if sync succeeded, False otherwise
    """
    if not sync_repo:
        return False

    try:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(workspace_dir)

        try:
            # Use clean environment to avoid macOS malloc warnings
            env = _clean_env()

            # Initialize git if needed
            if not (workspace_dir / ".git").exists():
                log.info("Initializing git repo...")
                subprocess.run(["git", "init"], capture_output=True, env=env)
                subprocess.run(
                    ["git", "remote", "add", "origin", f"https://github.com/{sync_repo}.git"],
                    capture_output=True, env=env
                )

            # Pull latest
            log.info(f"Syncing with {sync_repo}...")
            subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, env=env)
            subprocess.run(
                ["git", "merge", "origin/main", "--no-edit"],
                capture_output=True, env=env
            )

            # Check for changes to push
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, env=env
            )

            if result.stdout.strip():
                # Has changes, commit and push
                subprocess.run(["git", "add", "-A"], capture_output=True, env=env)
                subprocess.run(
                    ["git", "commit", "-m", f"Add segments from {machine_id}"],
                    capture_output=True, env=env
                )
                subprocess.run(["git", "branch", "-M", "main"], capture_output=True, env=env)
                push_result = subprocess.run(
                    ["git", "push", "-u", "origin", "main"],
                    capture_output=True, text=True, env=env
                )
                if push_result.returncode == 0:
                    log.info("Sync complete: pushed new segments")
                else:
                    log.warning(f"Push failed: {push_result.stderr}")
                    return False
            else:
                log.debug("Sync complete: no changes to push")

            return True

        finally:
            os.chdir(original_dir)

    except Exception as e:
        log.error(f"Sync failed: {e}")
        return False


def queue_sessions_for_indexing(
    session_files: List[Path],
    priority: bool = True,
    force: bool = False
) -> int:
    """Queue session files for indexing by the daemon.

    Args:
        session_files: List of session file paths
        priority: If True, add to priority queue
        force: If True, queue even if already indexed

    Returns:
        Number of sessions queued
    """
    indexed = get_indexed_sessions() if not force else set()
    queued = 0

    for session_file in session_files:
        path = str(session_file)
        if path in indexed and not force:
            continue
        if queue_add(path, priority=priority):
            queued += 1

    return queued


def is_daemon_running() -> bool:
    """Check if the daemon is running."""
    import platform

    system = platform.system()
    if system == "Darwin":
        result = subprocess.run(
            ["launchctl", "list", "net.julianfleck.cam"],
            capture_output=True
        )
        return result.returncode == 0
    elif system == "Linux":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "cam"],
            capture_output=True
        )
        return result.returncode == 0
    return False


def get_watch_paths() -> list[Path]:
    """Get paths to watch for session changes."""
    paths = []

    # Claude Code
    claude_dir = Path.home() / ".claude" / "projects"
    if claude_dir.exists():
        paths.append(claude_dir)

    # Cursor
    cursor_dir = Path.home() / ".cursor" / "projects"
    if cursor_dir.exists():
        paths.append(cursor_dir)

    # OpenClaw
    openclaw_dir = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
    if openclaw_dir.exists():
        paths.append(openclaw_dir)

    # Codex CLI
    codex_dir = Path.home() / ".codex" / "sessions"
    if codex_dir.exists():
        paths.append(codex_dir)

    return paths


def get_hostname() -> str:
    """Get machine hostname."""
    return os.environ.get("CAM_MACHINE_ID") or socket.gethostname().split('.')[0]


def get_workspace_dir() -> Path:
    """Get workspace directory for segments."""
    env_dir = os.environ.get("CAM_WORKSPACE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".cam" / "sessions"


def run_daemon(
    sync_repo: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    foreground: bool = False
):
    """Run the CAM daemon.

    The daemon has two components:
    1. Watcher: Monitors session directories and queues changed files
    2. Worker: Processes the queue with warm ML models

    Both run in the same process to share models in memory (~1.3GB).
    """
    watch_paths = get_watch_paths()

    if not watch_paths:
        log.error("No session directories found to watch")
        sys.exit(1)

    log.info(f"Watching {len(watch_paths)} directories:")
    for p in watch_paths:
        log.info(f"  - {p}")

    # Get output directory
    output_dir = Path(workspace_dir) if workspace_dir else get_workspace_dir()
    machine_id = get_hostname()

    log.info(f"Output directory: {output_dir}")
    log.info(f"Machine ID: {machine_id}")
    if sync_repo:
        log.info(f"Sync repo: {sync_repo}")

    # Clean stale entries from queue (sessions that were indexed but still in queue)
    cleaned = queue_clean()
    if cleaned > 0:
        log.info(f"Cleaned {cleaned} already-indexed sessions from queue")

    priority, normal = queue_size()
    if priority + normal > 0:
        log.info(f"Queue has {priority + normal} sessions pending ({priority} priority, {normal} normal)")

    # Create watcher handler and observer
    watcher = SessionWatcher()
    observer = Observer()

    for path in watch_paths:
        observer.schedule(watcher, str(path), recursive=True)
        log.info(f"Watching: {path}")

    # Create index worker
    worker = IndexWorker(output_dir, machine_id, sync_repo=sync_repo)

    # Handle signals
    def signal_handler(signum, frame):
        log.info("Received signal, shutting down...")
        watcher.stop()
        worker.stop()
        observer.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start file watcher
    observer.start()
    log.info("CAM daemon started")

    # Start watcher check thread
    def watcher_loop():
        while watcher.running:
            try:
                watcher.check_and_queue()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                log.error(f"Watcher error: {e}")
                time.sleep(30)

    watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
    watcher_thread.start()

    # Run worker in main thread
    try:
        worker.run_loop()
    finally:
        watcher.stop()
        observer.stop()
        observer.join()
        log.info("CAM daemon stopped")


def write_launchd_plist(
    sync_repo: str,
    workspace_dir: str,
    machine_id: str
) -> Path:
    """Generate macOS launchd plist file."""
    # Find the cam executable
    cam_bin = shutil.which("cam") or str(Path.home() / ".local" / "bin" / "cam")

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>net.julianfleck.cam</string>

    <key>ProgramArguments</key>
    <array>
        <string>{cam_bin}</string>
        <string>daemon</string>
        <string>run</string>
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>CAM_SYNC_REPO</key>
        <string>{sync_repo}</string>
        <key>CAM_WORKSPACE_DIR</key>
        <string>{workspace_dir}</string>
        <key>CAM_MACHINE_ID</key>
        <string>{machine_id}</string>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin</string>
        <key>HF_HUB_OFFLINE</key>
        <string>1</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>ProcessType</key>
    <string>Background</string>

    <key>StandardOutPath</key>
    <string>{Path.home()}/Library/Logs/cam.log</string>

    <key>StandardErrorPath</key>
    <string>{Path.home()}/Library/Logs/cam.error.log</string>
</dict>
</plist>
"""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "net.julianfleck.cam.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)
    return plist_path


def write_systemd_service(
    sync_repo: str,
    workspace_dir: str,
    machine_id: str
) -> Path:
    """Generate Linux systemd service file."""
    service_content = f"""[Unit]
Description=Collective Agent Memory - Session Sync Daemon
After=network.target

[Service]
Type=simple
ExecStart={sys.executable} -m cam.daemon --foreground
Environment="CAM_SYNC_REPO={sync_repo}"
Environment="CAM_WORKSPACE_DIR={workspace_dir}"
Environment="CAM_MACHINE_ID={machine_id}"
Environment="PATH=/usr/local/bin:/usr/bin:/bin:{Path.home()}/.local/bin"
Environment="HF_HUB_OFFLINE=1"
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
"""
    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / "cam.service"
    service_path.write_text(service_content)
    return service_path


def install_service(
    sync_repo: str,
    workspace_dir: str,
    machine_id: str
) -> bool:
    """Install and start the appropriate service for this platform."""
    import platform

    system = platform.system()

    if system == "Darwin":
        # macOS - use launchd
        plist_path = write_launchd_plist(sync_repo, workspace_dir, machine_id)
        log.info(f"Created launchd plist: {plist_path}")

        # Unload if already loaded
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True
        )

        # Load and start
        result = subprocess.run(
            ["launchctl", "load", str(plist_path)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            log.info("CAM daemon installed and started via launchd")
            log.info(f"  Logs: ~/Library/Logs/cam.log")
            log.info(f"  Stop: launchctl unload {plist_path}")
            return True
        else:
            log.error(f"Failed to load launchd service: {result.stderr}")
            return False

    elif system == "Linux":
        # Linux - use systemd
        service_path = write_systemd_service(sync_repo, workspace_dir, machine_id)
        log.info(f"Created systemd service: {service_path}")

        # Reload systemd
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True
        )

        # Enable and start
        result = subprocess.run(
            ["systemctl", "--user", "enable", "--now", "cam.service"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            log.info("CAM daemon installed and started via systemd")
            log.info(f"  Status: systemctl --user status cam")
            log.info(f"  Logs: journalctl --user -u cam -f")
            log.info(f"  Stop: systemctl --user stop cam")
            return True
        else:
            log.error(f"Failed to start systemd service: {result.stderr}")
            return False

    else:
        log.error(f"Unsupported platform: {system}")
        return False


def uninstall_service() -> bool:
    """Uninstall the daemon service."""
    import platform

    system = platform.system()

    if system == "Darwin":
        plist_path = Path.home() / "Library" / "LaunchAgents" / "net.julianfleck.cam.plist"
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            plist_path.unlink()
            log.info("CAM daemon uninstalled from launchd")
            return True
        else:
            log.info("CAM daemon not installed")
            return True

    elif system == "Linux":
        service_path = Path.home() / ".config" / "systemd" / "user" / "cam.service"
        if service_path.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", "cam.service"],
                capture_output=True
            )
            service_path.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
            log.info("CAM daemon uninstalled from systemd")
            return True
        else:
            log.info("CAM daemon not installed")
            return True

    else:
        log.error(f"Unsupported platform: {system}")
        return False


# =============================================================================
# Watchdog - Monitors daemon health and restarts if frozen
# =============================================================================

WATCHDOG_SCRIPT_PATH = Path.home() / ".cam" / "scripts" / "cam-watchdog.sh"
WATCHDOG_STATE_FILE = Path.home() / ".cam" / ".watchdog-state"
WATCHDOG_LOG_FILE = Path.home() / ".cam" / "watchdog.log"


def write_watchdog_script() -> Path:
    """Generate the watchdog bash script."""
    cam_bin = shutil.which("cam") or str(Path.home() / ".local" / "bin" / "cam")

    script_content = f"""#!/bin/bash
#
# CAM Daemon Watchdog
# Detects frozen daemon (alive but not processing queue) and restarts it
#

QUEUE_FILE="$HOME/.cam/.index-queue"
PRIORITY_QUEUE_FILE="$HOME/.cam/.index-queue-priority"
STATE_FILE="$HOME/.cam/.watchdog-state"
LOG_FILE="$HOME/.cam/watchdog.log"

log() {{ echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"; }}

# Check if daemon process is running
if ! pgrep -f "cam daemon" > /dev/null 2>&1; then
    log "WARN: Daemon not running, starting..."
    "{cam_bin}" daemon start
    exit 0
fi

# Get current queue length (both queues)
CURRENT=0
[ -f "$QUEUE_FILE" ] && CURRENT=$((CURRENT + $(wc -l < "$QUEUE_FILE" 2>/dev/null || echo 0)))
[ -f "$PRIORITY_QUEUE_FILE" ] && CURRENT=$((CURRENT + $(wc -l < "$PRIORITY_QUEUE_FILE" 2>/dev/null || echo 0)))

# Get previous queue length
PREVIOUS=$(cat "$STATE_FILE" 2>/dev/null || echo -1)

# Save current for next run
echo "$CURRENT" > "$STATE_FILE"

# If queue not empty AND unchanged for 2 consecutive checks → frozen
if [ "$CURRENT" -gt 0 ] && [ "$CURRENT" = "$PREVIOUS" ]; then
    log "ERROR: Queue stuck at $CURRENT items, restarting daemon"
    "{cam_bin}" daemon stop
    sleep 2
    "{cam_bin}" daemon start
else
    log "OK: queue=$CURRENT (was $PREVIOUS)"
fi
"""
    WATCHDOG_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHDOG_SCRIPT_PATH.write_text(script_content)
    WATCHDOG_SCRIPT_PATH.chmod(0o755)
    return WATCHDOG_SCRIPT_PATH


def write_watchdog_launchd_plist() -> Path:
    """Generate macOS launchd plist for watchdog (runs every 5 min)."""
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>net.julianfleck.cam.watchdog</string>

    <key>ProgramArguments</key>
    <array>
        <string>{WATCHDOG_SCRIPT_PATH}</string>
    </array>

    <key>StartInterval</key>
    <integer>300</integer>

    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{WATCHDOG_LOG_FILE}</string>

    <key>StandardErrorPath</key>
    <string>{WATCHDOG_LOG_FILE}</string>
</dict>
</plist>
"""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "net.julianfleck.cam.watchdog.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)
    return plist_path


def write_watchdog_systemd_files() -> tuple:
    """Generate Linux systemd timer and service for watchdog."""
    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)

    # Timer (runs every 5 min)
    timer_content = """[Unit]
Description=CAM Daemon Watchdog Timer

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
"""
    timer_path = service_dir / "cam-watchdog.timer"
    timer_path.write_text(timer_content)

    # Service (oneshot)
    service_content = f"""[Unit]
Description=CAM Daemon Watchdog

[Service]
Type=oneshot
ExecStart={WATCHDOG_SCRIPT_PATH}
"""
    service_path = service_dir / "cam-watchdog.service"
    service_path.write_text(service_content)

    return timer_path, service_path


def install_watchdog() -> bool:
    """Install the watchdog for this platform."""
    import platform

    system = platform.system()

    # Write the watchdog script
    script_path = write_watchdog_script()
    log.info(f"Created watchdog script: {script_path}")

    if system == "Darwin":
        plist_path = write_watchdog_launchd_plist()
        log.info(f"Created launchd plist: {plist_path}")

        # Unload if already loaded
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)

        # Load and start
        result = subprocess.run(
            ["launchctl", "load", str(plist_path)],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            log.info("CAM watchdog installed (runs every 5 min)")
            log.info(f"  Log: {WATCHDOG_LOG_FILE}")
            return True
        else:
            log.error(f"Failed to load watchdog: {result.stderr}")
            return False

    elif system == "Linux":
        timer_path, service_path = write_watchdog_systemd_files()
        log.info(f"Created systemd files: {timer_path}, {service_path}")

        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)

        result = subprocess.run(
            ["systemctl", "--user", "enable", "--now", "cam-watchdog.timer"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            log.info("CAM watchdog installed (runs every 5 min)")
            log.info(f"  Status: systemctl --user status cam-watchdog.timer")
            log.info(f"  Log: {WATCHDOG_LOG_FILE}")
            return True
        else:
            log.error(f"Failed to enable watchdog timer: {result.stderr}")
            return False

    else:
        log.error(f"Unsupported platform: {system}")
        return False


def uninstall_watchdog() -> bool:
    """Uninstall the watchdog."""
    import platform

    system = platform.system()

    if system == "Darwin":
        plist_path = Path.home() / "Library" / "LaunchAgents" / "net.julianfleck.cam.watchdog.plist"
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            plist_path.unlink()
            log.info("CAM watchdog uninstalled from launchd")
        else:
            log.info("CAM watchdog not installed")

    elif system == "Linux":
        timer_path = Path.home() / ".config" / "systemd" / "user" / "cam-watchdog.timer"
        service_path = Path.home() / ".config" / "systemd" / "user" / "cam-watchdog.service"

        if timer_path.exists() or service_path.exists():
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", "cam-watchdog.timer"],
                capture_output=True
            )
            if timer_path.exists():
                timer_path.unlink()
            if service_path.exists():
                service_path.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
            log.info("CAM watchdog uninstalled from systemd")
        else:
            log.info("CAM watchdog not installed")

    else:
        log.error(f"Unsupported platform: {system}")
        return False

    # Clean up script and state
    if WATCHDOG_SCRIPT_PATH.exists():
        WATCHDOG_SCRIPT_PATH.unlink()
    if WATCHDOG_STATE_FILE.exists():
        WATCHDOG_STATE_FILE.unlink()

    return True


def is_watchdog_running() -> bool:
    """Check if the watchdog is installed and running."""
    import platform

    system = platform.system()

    if system == "Darwin":
        result = subprocess.run(
            ["launchctl", "list", "net.julianfleck.cam.watchdog"],
            capture_output=True
        )
        return result.returncode == 0

    elif system == "Linux":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "cam-watchdog.timer"],
            capture_output=True
        )
        return result.returncode == 0

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAM Daemon")
    parser.add_argument("--foreground", "-f", action="store_true",
                        help="Run in foreground (don't daemonize)")
    parser.add_argument("--sync-repo", help="GitHub repo for sync")
    parser.add_argument("--workspace-dir", help="Workspace directory")

    args = parser.parse_args()

    run_daemon(
        sync_repo=args.sync_repo or os.environ.get("CAM_SYNC_REPO"),
        workspace_dir=args.workspace_dir or os.environ.get("CAM_WORKSPACE_DIR"),
        foreground=args.foreground
    )
