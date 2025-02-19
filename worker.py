import os
import torch
import pdfplumber
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize global variables
qa_pipeline = None
embeddings_model = None
text_chunks = []
tokenizer = None  

# Initialize the optimized model (Phi-2)
def init_llm():
    global qa_pipeline, embeddings_model, tokenizer
    
    model_name = "microsoft/phi-2"  # Open-source model
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Load a sentence transformer model for embeddings
    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to process a PDF document
def process_document(document_path):
    global text_chunks
    print(f"Processing document: {document_path}")
    try:
        if not os.path.exists(document_path):
            raise ValueError(f"File not found: {document_path}")

        # Load the PDF document using pdfplumber
        full_text = ""
        with pdfplumber.open(document_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        
        if not full_text.strip():
            raise ValueError("No text could be extracted from the PDF.")

        # Split the document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks = text_splitter.split_text(full_text)
        
        print(f"Split document into {len(text_chunks)} chunks.")
        
        print("Document processed successfully.")
    except Exception as e:
        print(f"Error processing document: {e}")
        raise ValueError("Failed to process the document.")

# Function to process a user prompt
def process_prompt(prompt):
    global qa_pipeline, text_chunks, embeddings_model, tokenizer

    try:
        if not text_chunks:
            return "Please upload a document first."

        # Find the most relevant chunks using embeddings
        prompt_embedding = embeddings_model.encode(prompt)
        chunk_embeddings = embeddings_model.encode(text_chunks)
        similarities = cosine_similarity([prompt_embedding], chunk_embeddings)[0]
        most_relevant_chunk_index = np.argmax(similarities)
        context = text_chunks[most_relevant_chunk_index]

        print(f"Selected context (chunk {most_relevant_chunk_index}): {context[:200]}...")

        # Truncate the context to fit within the model's token limit
        max_context_length = 400
        truncated_context = context[:max_context_length]

        # Use Phi-2 to answer the question
        input_text = f"""
        You are a helpful AI assistant. Please provide a concise answer to the following question based on the context provided below.

        Context: {truncated_context}

        Question: {prompt}

        Answer:
        """.strip()

        result = qa_pipeline(
            input_text,
            max_new_tokens=100,
            truncation=True,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = result[0]['generated_text']

        # Extract only the answer part
        answer_start = generated_text.find("Answer:")
        if answer_start != -1:
            answer = generated_text[answer_start + len("Answer:"):].strip()
        else:
            answer = generated_text.strip()

        # Clean up any unnecessary content that might still be included
        return answer.split('\n')[0].strip()  # Return only the first line as answer
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return "Sorry, I encountered an error while processing your question."

# Initialize the language model
init_llm()