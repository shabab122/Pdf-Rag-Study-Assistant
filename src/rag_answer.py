import os
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

try:
    from .retrieval import (
        DEFAULT_RESULT_COUNT,
        NO_RELEVANT_CONTEXT_MESSAGE,
        RetrievalResponse,
        retrieve,
    )
except ImportError:  # Supports: python src/rag_answer.py
    from retrieval import (
        DEFAULT_RESULT_COUNT,
        NO_RELEVANT_CONTEXT_MESSAGE,
        RetrievalResponse,
        retrieve,
    )


LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def retrieve_context(
    question: str,
    result_count: int = DEFAULT_RESULT_COUNT,
    source: str | None = None,
) -> RetrievalResponse:
    return retrieve(question, result_count=result_count, source=source)


def generate_answer(question: str, context: RetrievalResponse) -> str:
    """Call Groq only when at least one chunk passes the threshold."""
    if not context.has_context:
        return NO_RELEVANT_CONTEXT_MESSAGE

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found. Set your API key in the terminal first."
        )

    client = Groq(api_key=api_key)
    prompt = f"""
You are a helpful university-level teaching assistant.
Answer the user's question using ONLY the information in the PDF context.
Write the answer in natural, fluent Banglish (Bengali written using English/Roman letters).
Do not write Bengali using Bangla script.
Keep technical terms such as Shell, Bash, Scrum, command, and Linux in English.
Explain technical concepts simply and accurately.
If the answer is not present in the context, say that the information is not available in the provided PDF.

PDF context:
{context.format_context()}

User question:
{question}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def main() -> None:
    question = input("Ask a question about the PDF: ").strip()
    if not question:
        print("Please enter a question.")
        return

    print("\nSearching the PDF and generating an answer...\n")
    context = retrieve_context(question)
    answer = generate_answer(question, context)
    print(answer)

    if context.has_context:
        print("\nSources:")
        for chunk in context.chunks:
            print(f"- {chunk.source}, Page {chunk.page} (distance {chunk.distance:.4f})")
    else:
        print(f"\nNo chunk passed the threshold ({context.threshold:.2f}); LLM call skipped.")


if __name__ == "__main__":
    main()
