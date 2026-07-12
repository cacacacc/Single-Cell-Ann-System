"""SQLite persistence layer for users, search snapshots, chat history and notices.

The Flask routes call ``UserStore`` instead of touching SQLite directly.  This
keeps validation, password hashing, last-admin protection and row-to-dict
serialization in one place, which makes the web layer thinner and safer.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash


USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
VALID_ROLES = {"admin", "user"}


class UserStoreError(ValueError):
    """Base error for user persistence and validation failures."""


class DuplicateUserError(UserStoreError):
    """Raised when a username already exists."""


class AuthenticationError(UserStoreError):
    """Raised when login credentials are invalid."""


class LastAdminError(UserStoreError):
    """Raised when an operation would remove the final active admin."""


def utc_now() -> str:
    """Return a compact UTC timestamp string for database audit fields."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def normalize_username(username: str) -> str:
    """Normalize usernames to lowercase and enforce the accepted id format."""
    cleaned = (username or "").strip().lower()
    if not USERNAME_RE.fullmatch(cleaned):
        raise UserStoreError("账号需为 3-32 位字母、数字、下划线或短横线")
    return cleaned


def validate_password(password: str) -> None:
    """Apply the minimum password rule shared by create/change/reset flows."""
    if len(password or "") < 8:
        raise UserStoreError("密码长度至少为 8 位")


def validate_role(role: str) -> str:
    cleaned = (role or "user").strip().lower()
    if cleaned not in VALID_ROLES:
        raise UserStoreError("角色只能是 admin 或 user")
    return cleaned


def row_to_user(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "full_name": row["full_name"] or "",
        "email": row["email"] or "",
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_login_at": row["last_login_at"],
    }


def row_to_search_snapshot(row: sqlite3.Row) -> Dict[str, Any]:
    """Serialize a saved search row and rebuild the payload needed to rerun it."""
    filter_field = row["filter_field"] or ""
    filter_value = row["filter_value"] or ""
    index_name = row["index_name"] or ""
    payload: Dict[str, Any] = {
        "dataset_id": row["dataset_id"],
        "cell_id": row["cell_id"],
        "k": int(row["k"]),
    }
    if index_name:
        payload["index_name"] = index_name
    else:
        if row["index_backend"]:
            payload["index_backend"] = row["index_backend"]
        if row["index_type"]:
            payload["index_type"] = row["index_type"]
        if row["index_metric"]:
            payload["index_metric"] = row["index_metric"]
    if filter_field:
        payload["filter_field"] = filter_field
        payload["filter_value"] = filter_value

    return {
        "id": int(row["id"]),
        "user_id": int(row["user_id"]),
        "dataset_id": row["dataset_id"],
        "cell_id": row["cell_id"],
        "k": int(row["k"]),
        "use_rep": row["use_rep"] or "",
        "index_name": index_name,
        "index_backend": row["index_backend"] or "",
        "index_type": row["index_type"] or "",
        "index_metric": row["index_metric"] or "",
        "filter_field": filter_field,
        "filter_value": filter_value,
        "elapsed_ms": row["elapsed_ms"],
        "total_elapsed_ms": row["total_elapsed_ms"],
        "result_count": int(row["result_count"]),
        "created_at": row["created_at"],
        "rerun_payload": payload,
    }


class UserStore:
    """Small repository object wrapping all user-facing SQLite tables."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success, and always close it."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(
        self,
        default_admin_username: str = "admin",
        default_admin_password: str = "Admin@123456",
        default_admin_name: str = "系统管理员",
        default_admin_email: str = "admin@example.com",
    ) -> None:
        """Create tables/indexes and ensure at least one admin account exists."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    email TEXT,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    dataset_id TEXT NOT NULL,
                    cell_id TEXT NOT NULL,
                    k INTEGER NOT NULL,
                    use_rep TEXT,
                    index_name TEXT,
                    index_backend TEXT,
                    index_type TEXT,
                    index_metric TEXT,
                    filter_field TEXT,
                    filter_value TEXT,
                    elapsed_ms REAL,
                    total_elapsed_ms REAL,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_search_snapshots_user_created
                ON search_snapshots(user_id, created_at DESC)
                """
            )
            # ── 对话历史持久化 ────────────────────────────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id       TEXT PRIMARY KEY,
                    user_id  INTEGER NOT NULL,
                    title    TEXT NOT NULL DEFAULT '新对话',
                    dataset_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                ON chat_sessions(user_id, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_id, id ASC)
                """
            )
            # ─────────────────────────────────────────────────────────────────
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notifications (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    INTEGER NOT NULL,
                    type       TEXT NOT NULL DEFAULT 'info',
                    title      TEXT NOT NULL,
                    content    TEXT,
                    is_read    INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_noti_user_read
                ON notifications(user_id, is_read)
                """
            )
            # ─────────────────────────────────────────────────────────────────

            admin_username = normalize_username(default_admin_username)
            existing_admin = conn.execute(
                "SELECT id FROM users WHERE role = 'admin' LIMIT 1"
            ).fetchone()
            if existing_admin is None:
                now = utc_now()
                conn.execute(
                    """
                    INSERT INTO users (
                        username, password_hash, full_name, email, role,
                        is_active, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, 'admin', 1, ?, ?)
                    """,
                    (
                        admin_username,
                        generate_password_hash(default_admin_password, method="pbkdf2:sha256"),
                        default_admin_name,
                        default_admin_email,
                        now,
                        now,
                    ),
                )

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return row_to_user(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        normalized = normalize_username(username)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (normalized,)
            ).fetchone()
        return row_to_user(row) if row else None

    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Validate credentials and update the user's last-login timestamp."""
        normalized = normalize_username(username)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (normalized,)
            ).fetchone()
            if row is None or not check_password_hash(row["password_hash"], password or ""):
                raise AuthenticationError("账号或密码错误")
            if not bool(row["is_active"]):
                raise AuthenticationError("账号已被禁用，请联系管理员")

            now = utc_now()
            conn.execute(
                "UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?",
                (now, now, row["id"]),
            )
            fresh = conn.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
        assert fresh is not None
        return row_to_user(fresh)

    def list_users(self, query: str = "") -> List[Dict[str, Any]]:
        like = f"%{query.strip().lower()}%"
        with self._connect() as conn:
            if query.strip():
                rows = conn.execute(
                    """
                    SELECT * FROM users
                    WHERE username LIKE ? OR lower(full_name) LIKE ? OR lower(email) LIKE ?
                    ORDER BY role = 'admin' DESC, created_at ASC, id ASC
                    """,
                    (like, like, like),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM users
                    ORDER BY role = 'admin' DESC, created_at ASC, id ASC
                    """
                ).fetchall()
        return [row_to_user(row) for row in rows]

    def create_user(
        self,
        username: str,
        password: str,
        full_name: str = "",
        email: str = "",
        role: str = "user",
        is_active: bool = True,
    ) -> Dict[str, Any]:
        normalized = normalize_username(username)
        validate_password(password)
        cleaned_role = validate_role(role)
        now = utc_now()
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (
                        username, password_hash, full_name, email, role,
                        is_active, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized,
                        generate_password_hash(password, method="pbkdf2:sha256"),
                        full_name.strip(),
                        email.strip(),
                        cleaned_role,
                        1 if is_active else 0,
                        now,
                        now,
                    ),
                )
                user_id = int(cursor.lastrowid)
                row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        except sqlite3.IntegrityError as exc:
            raise DuplicateUserError("账号已存在") from exc
        assert row is not None
        return row_to_user(row)

    def update_user(
        self,
        user_id: int,
        *,
        full_name: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update profile/status fields while preventing removal of last admin."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise UserStoreError("用户不存在")

            next_role = validate_role(role) if role is not None else row["role"]
            next_active = bool(is_active) if is_active is not None else bool(row["is_active"])
            if row["role"] == "admin" and (next_role != "admin" or not next_active):
                self._ensure_other_active_admin(conn, user_id)

            fields = []
            values: List[Any] = []
            if full_name is not None:
                fields.append("full_name = ?")
                values.append(full_name.strip())
            if email is not None:
                fields.append("email = ?")
                values.append(email.strip())
            if role is not None:
                fields.append("role = ?")
                values.append(next_role)
            if is_active is not None:
                fields.append("is_active = ?")
                values.append(1 if next_active else 0)

            if fields:
                fields.append("updated_at = ?")
                values.append(utc_now())
                values.append(user_id)
                conn.execute(
                    f"UPDATE users SET {', '.join(fields)} WHERE id = ?",
                    tuple(values),
                )

            fresh = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        assert fresh is not None
        return row_to_user(fresh)

    def change_password(self, user_id: int, old_password: str, new_password: str) -> None:
        validate_password(new_password)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise UserStoreError("用户不存在")
            if not check_password_hash(row["password_hash"], old_password or ""):
                raise AuthenticationError("原密码错误")
            conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (generate_password_hash(new_password, method="pbkdf2:sha256"), utc_now(), user_id),
            )

    def set_password(self, user_id: int, new_password: str) -> None:
        validate_password(new_password)
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise UserStoreError("用户不存在")
            conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (generate_password_hash(new_password, method="pbkdf2:sha256"), utc_now(), user_id),
            )

    def delete_user(self, user_id: int) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise UserStoreError("用户不存在")
            if row["role"] == "admin":
                self._ensure_other_active_admin(conn, user_id)
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def add_search_snapshot(
        self,
        user_id: int,
        *,
        dataset_id: str,
        cell_id: str,
        k: int,
        use_rep: str = "",
        index_name: str = "",
        index_backend: str = "",
        index_type: str = "",
        index_metric: str = "",
        filter_field: str = "",
        filter_value: str = "",
        elapsed_ms: Optional[float] = None,
        total_elapsed_ms: Optional[float] = None,
        result_count: int = 0,
    ) -> Dict[str, Any]:
        """Persist enough search parameters for the UI to rerun a past query."""
        now = utc_now()
        with self._connect() as conn:
            user = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if user is None:
                raise UserStoreError("用户不存在")
            cursor = conn.execute(
                """
                INSERT INTO search_snapshots (
                    user_id, dataset_id, cell_id, k, use_rep, index_name,
                    index_backend, index_type, index_metric, filter_field,
                    filter_value, elapsed_ms, total_elapsed_ms, result_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    dataset_id,
                    cell_id,
                    int(k),
                    use_rep,
                    index_name,
                    index_backend,
                    index_type,
                    index_metric,
                    filter_field,
                    filter_value,
                    elapsed_ms,
                    total_elapsed_ms,
                    int(result_count),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM search_snapshots WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        assert row is not None
        return row_to_search_snapshot(row)

    def list_search_snapshots(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 20), 100))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM search_snapshots
                WHERE user_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, safe_limit),
            ).fetchall()
        return [row_to_search_snapshot(row) for row in rows]

    def delete_search_snapshot(self, user_id: int, snapshot_id: int) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM search_snapshots WHERE id = ? AND user_id = ?",
                (snapshot_id, user_id),
            )
            if cursor.rowcount == 0:
                raise UserStoreError("检索快照不存在")

    # ── 对话历史 ──────────────────────────────────────────────────────────────

    def create_chat_session(
        self,
        session_id: str,
        user_id: int,
        title: str = "新对话",
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a chat session if it does not exist and return its row."""
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions
                    (id, user_id, title, dataset_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, title, dataset_id, now, now),
            )
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else {}

    def list_chat_sessions(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 50), 200))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.title, s.dataset_id, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON m.session_id = s.id
                WHERE s.user_id = ?
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (user_id, safe_limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chat_messages(self, session_id: str, user_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            # 验证 session 属于该用户
            row = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if row is None:
                return []
            rows = conn.execute(
                "SELECT role, content, created_at FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def append_chat_message(
        self,
        session_id: str,
        user_id: int,
        role: str,
        content: str,
        title_from_first_user: bool = True,
    ) -> None:
        """Append one chat message and keep the session title/timestamp fresh."""
        now = utc_now()
        with self._connect() as conn:
            # 若 session 不存在则自动创建
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions
                    (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, '新对话', ?, ?)
                """,
                (session_id, user_id, now, now),
            )
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            # 用第一条 user 消息的前 30 个字作为标题
            if title_from_first_user and role == "user":
                count = conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages WHERE session_id = ? AND role = 'user'",
                    (session_id,),
                ).fetchone()["c"]
                if count == 1:
                    title = content[:30].strip().replace("\n", " ") or "新对话"
                    conn.execute(
                        "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                        (title, now, session_id),
                    )
                    return
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    def rename_chat_session(self, session_id: str, user_id: int, title: str) -> None:
        title = (title or "").strip()[:50] or "新对话"
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (title, utc_now(), session_id, user_id),
            )
            if cursor.rowcount == 0:
                raise UserStoreError("对话不存在")

    def delete_chat_session(self, session_id: str, user_id: int) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
            if cursor.rowcount == 0:
                raise UserStoreError("对话不存在")

    def clear_all_chat_sessions(self, user_id: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_sessions WHERE user_id = ?", (user_id,)
            )
        return cursor.rowcount

    # ─────────────────────────────────────────────────────────────────────────
    # 通知中心
    # ─────────────────────────────────────────────────────────────────────────

    def add_notification(
        self, user_id: int, noti_type: str, title: str, content: str = ""
    ) -> Dict[str, Any]:
        """Create an unread notification for a user."""
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO notifications (user_id, type, title, content, is_read, created_at)
                VALUES (?, ?, ?, ?, 0, ?)
                """,
                (user_id, noti_type, title, content, now),
            )
            return {
                "id": cursor.lastrowid,
                "user_id": user_id,
                "type": noti_type,
                "title": title,
                "content": content,
                "is_read": False,
                "created_at": now,
            }

    def list_notifications(
        self, user_id: int, limit: int = 20, unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        limit = min(max(limit, 1), 100)
        sql = "SELECT * FROM notifications WHERE user_id = ?"
        params: list = [user_id]
        if unread_only:
            sql += " AND is_read = 0"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": int(r["id"]),
                "user_id": int(r["user_id"]),
                "type": r["type"],
                "title": r["title"],
                "content": r["content"] or "",
                "is_read": bool(r["is_read"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def count_unread(self, user_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM notifications WHERE user_id = ? AND is_read = 0",
                (user_id,),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def mark_read(self, notification_id: int, user_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?",
                (notification_id, user_id),
            )
        return cursor.rowcount > 0

    def mark_all_read(self, user_id: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE notifications SET is_read = 1 WHERE user_id = ? AND is_read = 0",
                (user_id,),
            )
        return cursor.rowcount

    def delete_notification(self, notification_id: int, user_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM notifications WHERE id = ? AND user_id = ?",
                (notification_id, user_id),
            )
        return cursor.rowcount > 0

    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_other_active_admin(self, conn: sqlite3.Connection, user_id: int) -> None:
        """Raise if ``user_id`` is the only currently active admin account."""
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM users
            WHERE role = 'admin' AND is_active = 1 AND id != ?
            """,
            (user_id,),
        ).fetchone()
        if int(row["count"]) == 0:
            raise LastAdminError("系统必须至少保留一个启用的管理员")
