"""
Step 2: Ollama LLM Answer Generation
Runs in venv_ollama with pydantic v2
"""
import json
import ollama


def ask_ollama(query, context_chunks, model_name="gemma3:12b"):
    """
    Sends a query and context to the local Ollama model for answer generation.
    """
    context_text = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant that answers questions using the provided context.

Context:
{context_text}

Question:
{query}

Answer:"""

    response = ollama.chat(model=model_name, messages=[
        {"role": "user", "content": prompt}
    ])

    return response["message"]["content"]


if __name__ == "__main__":
    # Load retrieved chunks from Step 1
    with open("retrieved_chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    query = data["query"]
    chunks = data["chunks"]

    print(f"Query: {query}")
    print(f"\nGenerating answer using Ollama...")

    # Generate answer
    answer = ask_ollama(query, chunks, model_name="gemma3:12b")

    print("\n=== Final Answer ===\n")
    print(answer)

    # Save answer
    with open("answer.txt", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        f.write(f"Answer: {answer}\n")

    print("\n=== Answer saved to answer.txt ===")
