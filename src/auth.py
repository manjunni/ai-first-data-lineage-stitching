"""Configured local authentication and role hierarchy for lineage review."""
from __future__ import annotations

import hmac
import json
from dataclasses import dataclass

from config.settings import settings

ROLE_RANK = {
    "Analyst": 1,
    "SME": 2,
    "Domain Steward": 3,
    "Data Owner": 4,
}


@dataclass(frozen=True)
class AuthenticatedUser:
    username: str
    role: str


def _users() -> dict:
    if not settings.lineage_users_json:
        return {}
    try:
        users = json.loads(settings.lineage_users_json)
    except json.JSONDecodeError as exc:
        raise ValueError("LINEAGE_USERS_JSON must contain valid JSON") from exc
    if not isinstance(users, dict):
        raise ValueError("LINEAGE_USERS_JSON must be a JSON object")
    return users


def authenticate(username: str, password: str) -> AuthenticatedUser | None:
    record = _users().get(username.strip())
    if not isinstance(record, dict):
        return None
    configured_password = str(record.get("password", ""))
    role = str(record.get("role", ""))
    if not hmac.compare_digest(configured_password, password) or role not in ROLE_RANK:
        return None
    return AuthenticatedUser(username=username.strip(), role=role)


def role_can_review(actor_role: str, current_reviewer_role: str | None) -> bool:
    if actor_role not in ROLE_RANK:
        return False
    if not current_reviewer_role:
        return True
    return ROLE_RANK[actor_role] >= ROLE_RANK.get(current_reviewer_role, 0)


def role_for_user(username: str | None) -> str | None:
    if not username:
        return None
    record = _users().get(username.strip())
    role = record.get("role") if isinstance(record, dict) else None
    return role if role in ROLE_RANK else None
