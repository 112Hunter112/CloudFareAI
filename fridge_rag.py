from pypdf import PdfReader
import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

def pdf_chunker(pdf_name, chunk_size=500, overlap=100):
    reader = PdfReader(pdf_name, strict=True)
    pages = reader.pages
    print(f"Total pages: {len(pages)}")

    full_text = ""
    for i, page in enumerate(pages, start=1):
        print(f"Processing page {i}")
        text = page.extract_text()
        if "Refrigerators" in text and len(text.strip()) < 2000:
            continue
        full_text += text + "\n"

    print(f"Total characters: {len(full_text)}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(full_text)
    print(f"Total chunks: {len(chunks)}")

    return chunks


def create_embedding(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, model


def faiss_search(embeddings):
    dimensions = 384
    index = faiss.IndexFlatL2(dimensions)
    index.add(embeddings.astype("float32"))
    return index


def retrieve_chunks(query, index, chunks, model, top_k=3):
    query_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, top_k)

    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    results = [chunks[i] for i in indices[0]]
    return results


if __name__ == "__main__":
    # Process PDF and create chunks
    chunks = pdf_chunker("fridge-owners-manual.pdf", 1000, 500)

    # Create embeddings
    embeddings, model = create_embedding(chunks)

    # Create FAISS index
    index = faiss_search(embeddings)

    # Save chunks for later use
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Save embeddings and model for reuse
    np.save("embeddings.npy", embeddings)

    # Save FAISS index
    faiss.write_index(index, "faiss_index.bin")

    # Example: retrieve chunks for a query
    query = "How do I replace the LED lights?"
    top_chunks = retrieve_chunks(query, index, chunks, model, top_k=3)

    # Save retrieved chunks for Ollama (can be used by step2_ollama_answer.py)
    with open("retrieved_chunks.json", "w", encoding="utf-8") as f:
        json.dump({
            "query": query,
            "chunks": top_chunks
        }, f, ensure_ascii=False, indent=2)

    print("\n=== Retrieved chunks saved to retrieved_chunks.json ===")
    for i, chunk in enumerate(top_chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:300]}...")

    print("\n=== Processing Complete ===")
    print("Files saved:")
    print("  - chunks.json (all chunks)")
    print("  - embeddings.npy (embeddings array)")
    print("  - faiss_index.bin (FAISS index)")
    print("  - retrieved_chunks.json (retrieved chunks for Ollama)")
