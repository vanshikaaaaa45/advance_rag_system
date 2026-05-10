"""
backend/llm.py — LLM connector (Groq or Ollama).

WHY GROQ?
Groq runs LLaMA 3 on custom hardware (LPUs) that are absurdly fast.
Free tier gives you ~14,400 requests/day at ~500 tokens/second.
For local development, this is effectively unlimited.

THE RAG PROMPT:
The prompt is the most important part of the generation step.
Bad prompt → hallucinations even with perfect retrieval.

Key elements of a good RAG prompt:
1. ROLE: Tell the model what it is (a helpful assistant using documents)
2. CONTEXT: The retrieved chunks, clearly labeled
3. INSTRUCTION: Be factual, cite sources, admit when you don't know
4. QUESTION: The user's actual question
5. CONSTRAINTS: Don't make things up if not in context

"If the answer is not in the context, say so" is critical.
Without this, the model will hallucinate rather than admit ignorance.
"""

from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from typing import Optional

from backend.config import GROQ_API_KEY, GROQ_MODEL, GROQ_FAST_MODEL, LLM_PROVIDER


# ── RAG Prompt Template ────────────────────────────────────────────────────────
# {context} and {question} are filled in at runtime
RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful AI assistant that answers questions based on the provided context documents.

CONTEXT DOCUMENTS:
{context}

INSTRUCTIONS:
- Answer the question using ONLY information from the context above
- If the answer is not in the context, say "I don't have enough information in the provided documents to answer this question"
- Be concise and direct
- Reference specific parts of the context when relevant
- Do not make up information not present in the context

QUESTION: {question}

ANSWER:""")


def get_llm(fast: bool = False):
    """
    Get the configured LLM instance.

    Args:
        fast: Use the smaller/faster model (for routing decisions)
              vs the larger/better model (for final answer generation)
    """
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file. "
                "Get a free key at console.groq.com"
            )
        model = GROQ_FAST_MODEL if fast else GROQ_MODEL
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=model,
            temperature=0.1,      # Low temperature = factual, consistent answers
            max_tokens=1024,      # Max response length
        )
    elif LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            base_url="http://localhost:11434",
            model="llama3.1",
            temperature=0.1,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


def format_context(documents: list[Document]) -> str:
    """
    Format retrieved documents into a clean context string for the prompt.

    WHY FORMAT CAREFULLY?
    The LLM needs to clearly understand where each piece of information
    came from. Numbering sources makes it easy to reference them.
    Including the source filename helps the LLM cite correctly.
    """
    if not documents:
        return "No relevant documents found."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        page_str = f", page {page}" if page else ""

        context_parts.append(
            f"[Document {i}] Source: {source}{page_str}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(question: str, documents: list[Document]) -> str:
    """
    Generate an answer using retrieved documents as context.

    This is the "G" in RAG — the Generation step.
    We use LangChain's LCEL (pipe syntax) to chain:
      prompt | llm | output_parser

    LCEL pipe syntax:
      prompt.invoke(inputs) → formatted prompt
      llm.invoke(prompt) → AIMessage with generated text
      parser.invoke(message) → plain string
    """
    llm = get_llm(fast=False)
    parser = StrOutputParser()

    # Build the chain: prompt → llm → parse to string
    chain = RAG_PROMPT | llm | parser

    context = format_context(documents)

    logger.info(f"Generating answer with {len(documents)} context docs...")

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    logger.info(f"Answer generated ({len(answer)} chars)")
    return answer
