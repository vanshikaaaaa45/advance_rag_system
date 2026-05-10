"""
evaluation/ragas_eval.py — RAGAs evaluation metrics.

WHAT IS RAGAs?
RAGAs (Retrieval Augmented Generation Assessment) is a framework
for measuring the quality of RAG pipelines without human labels.

It measures 4 key metrics:

1. FAITHFULNESS (0 to 1)
   "Is the answer factually consistent with the retrieved context?"
   Low score = hallucination. The answer contains claims not in the docs.
   
   HOW: LLM breaks answer into statements → checks each against context
   Example: Answer says "RAG was invented in 2020" but docs say 2021 → low score

2. ANSWER RELEVANCY (0 to 1)
   "Does the answer actually address the question asked?"
   Low score = answer is on-topic but doesn't answer the specific question.
   
   HOW: LLM generates fake questions from the answer → checks if they
   match the original question via embedding similarity

3. CONTEXT PRECISION (0 to 1)
   "Are the retrieved chunks actually useful for answering?"
   Low score = retrieved noise — chunks that don't help answer the question.
   
   HOW: For each retrieved chunk, checks if it's relevant to the question

4. CONTEXT RECALL (0 to 1)
   "Did we retrieve all the information needed to answer?"
   Low score = missing information — we needed more chunks.
   Requires ground truth answers to compute.

WHY THESE METRICS MATTER FOR YOUR RESUME:
Most RAG projects just "work" — you ask questions and get answers.
Having actual NUMBERS (faithfulness: 0.87, relevancy: 0.92) shows
you care about quality and know how to measure it. This is what
senior engineers and researchers do in production.

FREE SETUP:
RAGAs can use any LLM as the judge. We use Groq (free) instead of
the default OpenAI, so evaluation costs nothing.
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain.schema import Document
from loguru import logger
from typing import Optional
import time

from backend.llm import get_llm
from backend.ingestion.embedder import EmbeddingStore
from backend.config import EVAL_SAMPLE_SIZE


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: Optional[list[str]] = None
) -> dict:
    """
    Run RAGAs evaluation on a batch of Q&A pairs.

    Args:
        questions:     List of user questions
        answers:       List of generated answers
        contexts:      List of lists — retrieved chunks for each question
        ground_truths: Optional list of correct answers (for context recall)

    Returns:
        Dict with metric scores and per-sample results

    DATASET FORMAT:
    RAGAs expects a HuggingFace Dataset with these columns:
    - question: str
    - answer: str
    - contexts: list[str]
    - ground_truth: str (optional)
    """
    logger.info(f"Running RAGAs evaluation on {len(questions)} samples...")

    # Build the evaluation dataset
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    if ground_truths:
        eval_data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(eval_data)

    # Use Groq as the judge LLM (free!)
    # RAGAs uses an LLM to assess faithfulness and relevancy
    try:
        llm = get_llm(fast=True)
        ragas_llm = LangchainLLMWrapper(llm)

        # Use our local embedding model for answer_relevancy metric
        store = EmbeddingStore()
        ragas_embeddings = LangchainEmbeddingsWrapper(store.embedding_model)

        # Select metrics to compute
        # context_recall requires ground_truth — skip if not provided
        metrics = [faithfulness, answer_relevancy, context_precision]

        start = time.time()

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,  # Don't crash on individual failures
        )

        elapsed = time.time() - start
        logger.info(f"RAGAs evaluation complete in {elapsed:.1f}s")

        # Extract scores
        scores = {}
        result_df = result.to_pandas()

        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            if metric in result_df.columns:
                val = result_df[metric].mean()
                scores[metric] = round(float(val), 4) if not __import__('math').isnan(val) else 0.0

        logger.info(f"RAGAs scores: {scores}")
        return {
            "success": True,
            "scores": scores,
            "num_samples": len(questions),
            "elapsed_seconds": round(elapsed, 1),
            "per_sample": result_df.to_dict(orient="records")
        }

    except Exception as e:
        logger.error(f"RAGAs evaluation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "scores": {},
            "num_samples": len(questions)
        }


def evaluate_single(
    question: str,
    answer: str,
    context_docs: list[Document]
) -> dict:
    """
    Evaluate a single Q&A pair.
    Called after each query to build up running metrics.
    """
    contexts = [[doc.page_content for doc in context_docs]]

    return run_ragas_evaluation(
        questions=[question],
        answers=[answer],
        contexts=contexts
    )
