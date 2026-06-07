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
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def normalize_username(username: str) -> str:
    cleaned = (username or "").strip().lower()
    if not USERNAME_RE.fullmatch(cleaned):
        raise UserStoreError("账号需为 3-32 位字母、数字、下划线或短横线")
    return cleaned


def validate_password(password: str) -> None:
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


class UserStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
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
                        generate_password_hash(default_admin_password),
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
                        generate_password_hash(password),
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
                (generate_password_hash(new_password), utc_now(), user_id),
            )

    def set_password(self, user_id: int, new_password: str) -> None:
        validate_password(new_password)
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise UserStoreError("用户不存在")
            conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (generate_password_hash(new_password), utc_now(), user_id),
            )

    def delete_user(self, user_id: int) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise UserStoreError("用户不存在")
            if row["role"] == "admin":
                self._ensure_other_active_admin(conn, user_id)
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def _ensure_other_active_admin(self, conn: sqlite3.Connection, user_id: int) -> None:
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
