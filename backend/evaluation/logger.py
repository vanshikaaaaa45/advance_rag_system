"""
evaluation/logger.py — Query logging and metrics aggregation.

WHY LOG EVERY QUERY?
Without logging you can't answer:
- "Is our RAG getting better or worse over time?"
- "Which questions are users asking most?"
- "What's the P99 latency?" (slowest 1% of queries)
- "Which documents are most useful?"

This logger stores every query with:
- Timing for each pipeline step (embed, retrieve, rerank, generate)
- RAGAs scores if evaluation was run
- Which route the agent took
- How many rewrites were needed

In production this would go to a database (PostgreSQL, ClickHouse).
For this project: in-memory list + JSON file for persistence.

P50/P99 LATENCY EXPLAINED:
P50 (median): 50% of queries finish faster than this value
P99: 99% of queries finish faster than this value
P99 tells you about your worst-case user experience.
"Our P99 is 3.2s" is a much more useful metric than "average is 1.1s"
"""

import json
import time
from pathlib import Path
from collections import defaultdict
from loguru import logger
from typing import Optional
import statistics


class QueryLogger:
    """
    Tracks query metrics in memory with JSON file backup.
    Singleton — one instance shared across the entire app.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._logs: list[dict] = []
        self._log_file = Path("logs/query_log.jsonl")
        self._log_file.parent.mkdir(exist_ok=True)

        # Load existing logs from file on startup
        self._load_from_file()
        self._initialized = True
        logger.info(f"QueryLogger initialized with {len(self._logs)} existing records")

    def log_query(
        self,
        question: str,
        answer: str,
        route: str,
        latency_ms: float,
        num_docs: int,
        rewrite_count: int = 0,
        ragas_scores: Optional[dict] = None,
        step_timings: Optional[dict] = None,
    ):
        """
        Log a completed query with all its metadata.

        step_timings example:
        {
            "routing_ms": 120,
            "retrieval_ms": 450,
            "reranking_ms": 200,
            "generation_ms": 800,
        }
        """
        record = {
            "timestamp": time.time(),
            "question": question[:200],           # Truncate long questions
            "answer_length": len(answer),
            "route": route,
            "latency_ms": round(latency_ms, 1),
            "num_docs_retrieved": num_docs,
            "rewrite_count": rewrite_count,
            "ragas_scores": ragas_scores or {},
            "step_timings": step_timings or {},
        }

        self._logs.append(record)
        self._append_to_file(record)

    def get_summary_metrics(self) -> dict:
        """
        Compute aggregated metrics across all logged queries.

        Returns metrics useful for the dashboard:
        - Total queries
        - Average / P50 / P99 latency
        - Route distribution
        - Average RAGAs scores
        - Rewrite rate
        """
        if not self._logs:
            return self._empty_metrics()

        latencies = [r["latency_ms"] for r in self._logs]
        latencies_sorted = sorted(latencies)

        # P50 and P99
        n = len(latencies_sorted)
        p50 = latencies_sorted[n // 2]
        p99 = latencies_sorted[int(n * 0.99)]

        # Route distribution
        route_counts: dict = defaultdict(int)
        for r in self._logs:
            route_counts[r.get("route", "unknown")] += 1

        # RAGAs averages (only from queries that had evaluation)
        evaluated = [r for r in self._logs if r.get("ragas_scores")]
        avg_faithfulness = 0.0
        avg_relevancy = 0.0
        avg_precision = 0.0

        if evaluated:
            avg_faithfulness = statistics.mean(
                r["ragas_scores"].get("faithfulness", 0) for r in evaluated
            )
            avg_relevancy = statistics.mean(
                r["ragas_scores"].get("answer_relevancy", 0) for r in evaluated
            )
            avg_precision = statistics.mean(
                r["ragas_scores"].get("context_precision", 0) for r in evaluated
            )

        # Rewrite rate — how often did the self-RAG loop trigger?
        rewrites = [r for r in self._logs if r.get("rewrite_count", 0) > 0]
        rewrite_rate = len(rewrites) / len(self._logs) if self._logs else 0

        return {
            "total_queries": len(self._logs),
            "avg_latency_ms": round(statistics.mean(latencies), 1),
            "p50_latency_ms": round(p50, 1),
            "p99_latency_ms": round(p99, 1),
            "min_latency_ms": round(min(latencies), 1),
            "max_latency_ms": round(max(latencies), 1),
            "route_distribution": dict(route_counts),
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_answer_relevancy": round(avg_relevancy, 4),
            "avg_context_precision": round(avg_precision, 4),
            "evaluated_queries": len(evaluated),
            "rewrite_rate": round(rewrite_rate, 3),
            "recent_queries": [
                {
                    "question": r["question"][:80],
                    "latency_ms": r["latency_ms"],
                    "route": r["route"],
                    "ragas": r.get("ragas_scores", {})
                }
                for r in self._logs[-10:]  # Last 10 queries
            ]
        }

    def get_latency_history(self, last_n: int = 50) -> list[dict]:
        """Get latency over time for the chart."""
        recent = self._logs[-last_n:]
        return [
            {
                "index": i,
                "latency_ms": r["latency_ms"],
                "route": r.get("route", "unknown")
            }
            for i, r in enumerate(recent)
        ]

    def _empty_metrics(self) -> dict:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "p50_latency_ms": 0,
            "p99_latency_ms": 0,
            "min_latency_ms": 0,
            "max_latency_ms": 0,
            "route_distribution": {},
            "avg_faithfulness": 0,
            "avg_answer_relevancy": 0,
            "avg_context_precision": 0,
            "evaluated_queries": 0,
            "rewrite_rate": 0,
            "recent_queries": []
        }

    def _append_to_file(self, record: dict):
        """Append a single record to the JSONL file."""
        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def _load_from_file(self):
        """Load existing logs from JSONL file on startup."""
        if not self._log_file.exists():
            return
        try:
            with open(self._log_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._logs.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to load logs: {e}")


# Singleton instance
query_logger = QueryLogger()
