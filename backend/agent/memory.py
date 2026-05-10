"""
agent/memory.py — Conversation memory for multi-turn chat.

WHY DO WE NEED MEMORY?
LLMs are stateless — each call is independent with no memory of prior turns.
Without memory, this conversation breaks:

User: "Who invented the transformer architecture?"
Bot:  "Vaswani et al. in 2017."
User: "How many parameters did it have?"
Bot:  "How many parameters did WHAT have?" ← can't resolve "it"

With memory, we inject chat history into each new query so the LLM
can resolve references like "it", "they", "that approach", etc.

APPROACH — ConversationBufferMemory:
Simple: store all (user, assistant) pairs and prepend to each new query.
Good for short conversations (< 10 turns).

For production with long conversations:
- ConversationSummaryMemory: summarizes old turns to save tokens
- ConversationTokenBufferMemory: keeps last N tokens of history

We keep it simple here — buffer memory is enough for demos and interviews.

SESSION-BASED:
Each user session gets its own memory store (keyed by session_id).
Multiple users can chat simultaneously without their histories mixing.
"""

from collections import defaultdict
from loguru import logger


class ConversationMemory:
    """
    Simple in-memory conversation history store.

    In production: use Redis or a database for persistence across restarts.
    For this project: dict in memory is perfect.
    """

    def __init__(self):
        # session_id → list of {"role": "user"/"assistant", "content": "..."}
        self._histories: dict[str, list[dict]] = defaultdict(list)

    def add_turn(self, session_id: str, question: str, answer: str):
        """Store a completed conversation turn."""
        self._histories[session_id].append({
            "role": "user",
            "content": question
        })
        self._histories[session_id].append({
            "role": "assistant",
            "content": answer
        })

        # Keep last 10 turns (20 messages) to avoid context overflow
        if len(self._histories[session_id]) > 20:
            self._histories[session_id] = self._histories[session_id][-20:]

        logger.info(
            f"Memory: session {session_id[:8]} now has "
            f"{len(self._histories[session_id])} messages"
        )

    def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        return self._histories.get(session_id, [])

    def get_context_string(self, session_id: str) -> str:
        """
        Format history as a string to inject into prompts.

        Returns empty string if no history (first turn).
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        lines = []
        for msg in history[-6:]:  # Last 3 turns (6 messages)
            role = "Human" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")

        return "Previous conversation:\n" + "\n".join(lines) + "\n\n"

    def clear(self, session_id: str):
        """Clear history for a session."""
        if session_id in self._histories:
            del self._histories[session_id]
            logger.info(f"Memory cleared for session {session_id[:8]}")

    def get_all_sessions(self) -> list[str]:
        return list(self._histories.keys())


# Singleton memory store shared across all requests
memory = ConversationMemory()
