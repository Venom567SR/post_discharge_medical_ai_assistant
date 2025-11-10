from typing import Dict, Optional
from datetime import datetime, timedelta
from loguru import logger

from src.graph.state import GraphState


class SessionManager:
    """Manages session state persistence across conversation turns."""

    def __init__(self, ttl_minutes: int = 60):
        """
        Initialize session manager.

        Args:
            ttl_minutes: Time-to-live for sessions in minutes
        """
        self._sessions: Dict[str, Dict] = {}
        self._ttl_minutes = ttl_minutes

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve session state if it exists and is not expired.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None
        """
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]

        # Check if expired
        if datetime.now() - session["last_updated"] > timedelta(minutes=self._ttl_minutes):
            logger.info(f"Session {session_id} expired, removing")
            del self._sessions[session_id]
            return None

        return session.get("state")

    def save_session(self, session_id: str, state: GraphState):
        """
        Save session state.

        Args:
            session_id: Session identifier
            state: Graph state to save
        """
        self._sessions[session_id] = {
            "state": state,
            "last_updated": datetime.now()
        }
        logger.debug(f"Saved session {session_id}")

    def update_session(self, session_id: str, updates: Dict):
        """
        Update specific fields in session state.

        Args:
            session_id: Session identifier
            updates: Dictionary of field updates
        """
        session = self.get_session(session_id)
        if session:
            session.update(updates)
            self._sessions[session_id]["state"] = session
            self._sessions[session_id]["last_updated"] = datetime.now()
            logger.debug(f"Updated session {session_id}")

    def clear_session(self, session_id: str):
        """
        Clear a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session {session_id}")

    def cleanup_expired(self):
        """Remove all expired sessions."""
        now = datetime.now()
        expired = [
            sid for sid, data in self._sessions.items()
            if now - data["last_updated"] > timedelta(minutes=self._ttl_minutes)
        ]

        for sid in expired:
            del self._sessions[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def count_active_sessions(self) -> int:
        """Get count of active sessions."""
        self.cleanup_expired()
        return len(self._sessions)


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
