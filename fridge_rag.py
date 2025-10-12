from pypdf import PdfReader
import json
from sentence_transformers import SentenceTransformer
import ollama
import faiss
import numpy as np
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def pdf_chucnker(pdf_name, chunk_size = 500, overlap = 100) :

    reader = PdfReader(pdf_name, strict=True) # reads file
    pages = reader.pages # this is an obj
    print(len(pages)) # prints the pages in the file


    full_text = ""
    for i , page in enumerate(pages,start=1):
        print(f"Page number {i} ")
        text = page.extract_text()
        if "Refrigerators" in text and len(text.strip()) < 2000:
            continue
        full_text += text + "\n"


    print(f"Total characters: {len(full_text)}")

    chunks = []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,  # tokens per chunk
        chunk_overlap=overlap,  # overlap tokens
        separators=["\n\n", "\n", ". ", " ", ""]  # tries these in order
    )

    chunks = text_splitter.split_text(full_text)

    print(f"Total chunks: {len(chunks)}")


    return chunks


def create_embedding(chunks) :

    model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, fast


    embeddings = model.encode(chunks, show_progress_bar=True)

    print("Shape of embeddings:", embeddings.shape)  # (137, 384)
    print("Embedding for first sentence:\n", embeddings[0])

    return embeddings, model


def faiss_search(embeddings) :

    dimensions = 384 # number of dimensions so FAISS can convert to L2 euclidian
    index = faiss.IndexFlatL2(dimensions)
    index.add(embeddings.astype("float32"))  # make sure dtype is float32 as required my numpy
    return index


def retrieve_chunks(query, index, chunks, model, top_k=3):
    # Encode the query
    query_emb = model.encode([query]).astype("float32")

    # Search in FAISS index
    distances, indices = index.search(query_emb, top_k)


    print("Distances:", distances)
    print("Indices:", indices)

    # Retrieve actual chunk text
    results = [chunks[i] for i in indices[0]]

    return results


chunks = pdf_chucnker("fridge-owners-manual.pdf",1000,500)
embeddings,  model = create_embedding(chunks)
index = faiss_search(embeddings)

query = "How do I adjust the fridge temperature?"
top_chunks = retrieve_chunks(query, index, chunks, model, top_k=3)

for i, chunk in enumerate(top_chunks):
    print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}")  # first 500 chars