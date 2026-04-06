import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.memory.embeddings import get_embedding
from app.memory.vector_store import init_vector_store

# Load the processed knowledge base
knowledge_base_file = os.path.join('app', 'data', 'knowledge_base', 'knowledge_base.txt')
with open(knowledge_base_file, 'r', encoding='utf-8') as f:
    docs = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(docs)} documents from the knowledge base.")

# Add a few specific examples to ensure they are present
docs.extend([
    "Dengue: high fever, low platelets, severe body pain",
    "Influenza: fever, headache, muscle pain, fatigue",
    "COVID-19: fever, cough, fatigue, loss of taste",
    "Typhoid: prolonged fever, weakness, abdominal pain",
    "Cardiac Arrest: sudden collapse, loss of consciousness (unresponsiveness), no pulse, and no breathing or only gasping"
])

print("Generating embeddings for all documents...")
embeddings = [get_embedding(doc) for doc in docs]

print("Initializing vector store...")
init_vector_store(embeddings, docs)

print("Vector DB saved to disk with expanded knowledge base.")