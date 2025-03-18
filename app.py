import os
import faiss
import shutil
import numpy as np
from flask import Flask, request
from flask_restx import Api, Resource, fields
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.vectorstores.utils import DistanceStrategy
from sentence_transformers import SentenceTransformer

import time
from openai import OpenAI
from openai import RateLimitError

from config import REPO_DIR_PATH, GIT_PATH, EMBEDDING_MODEL_NAME, TRAINING_DATA_PATH, FAISS_INDEX_PATH, SUPPORTED_FILE_TYPES
vector_db = None

if not os.path.exists(REPO_DIR_PATH):
    Repo.clone_from(GIT_PATH, REPO_DIR_PATH)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Flask and Swagger setup
app = Flask(__name__)
api = Api(app, version="1.0", title="Git Repo QA API",
          description="Index and query local Git repository files",
          doc="/swagger/")

ns = api.namespace('api', description='Operations')

# Swagger Models
query_model = api.model('Query', {
    'query': fields.String(required=True, description='Query about the repo')
})

result_model = api.model('Result', {
    'content': fields.String(description='Snippet of file content'),
    'source': fields.String(description='File path')
})

index_response = api.model('IndexResponse', {
    'message': fields.String(description='Status message')
})

query_response = api.model('QueryResponse', {
    'results': fields.List(fields.Nested(result_model))
})

# Function to extract files
def extract_files():
    docs = []
    for root, _, files in os.walk(TRAINING_DATA_PATH):
        for file in files:
            if file.endswith(SUPPORTED_FILE_TYPES):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        docs.append(Document(page_content=content, metadata={"source": file_path}))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return docs

# Create FAISS index
def create_index(docs):
    texts = [doc.page_content for doc in docs]
    metadata = [doc.metadata for doc in docs]

    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    docstore_dict = {i: doc for i, doc in enumerate(docs)}
    docstore = InMemoryDocstore(docstore_dict)

    return FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id={i: i for i in range(len(docs))},
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )



# Load or create FAISS index
def load_or_create_index():
    # Remove invalid file named faiss_index
    if os.path.exists(FAISS_INDEX_PATH) and not os.path.isdir(FAISS_INDEX_PATH):
        print(f"Removing invalid file at {FAISS_INDEX_PATH}")
        os.remove(FAISS_INDEX_PATH)

    # Trying to load existing index if directory exists
    if os.path.isdir(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        files = os.listdir(FAISS_INDEX_PATH)
        print(f"Found files: {files}")
        if "index.faiss" not in files or "index.pkl" not in files:
            print("Missing FAISS index files, creating new index.")
        else:
            return FAISS.load_local(
                FAISS_INDEX_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )

    # Create new index
    print("Creating new FAISS index...")
    docs = extract_files()
    print(f"[DEBUG] Extracted {len(docs)} documents.")
    
    if not docs:
        raise ValueError("No valid files found in the repository.")

    vector_db = create_index(docs)
    print("[DEBUG] FAISS index created in memory.")

    # Force directory creation, then save
    if os.path.exists(FAISS_INDEX_PATH):
        print("[DEBUG] Cleaning up old index directory...")
        shutil.rmtree(FAISS_INDEX_PATH)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    print(f"[DEBUG] Saving FAISS index to {FAISS_INDEX_PATH}...")
    vector_db.save_local(FAISS_INDEX_PATH)
    
    # Step 5: Confirm files written
    files_written = os.listdir(FAISS_INDEX_PATH)
    print(f"[DEBUG] Files in index directory: {files_written}")

    if "index.faiss" not in files_written or "index.pkl" not in files_written:
        raise RuntimeError("FAISS save_local failed to write expected files.")

    return vector_db

client = OpenAI()

def is_question_in_scope(question):
    prompt = (
        "Determine if the following question is related to the content "
        "of the repository about [insert repo domain]. Answer with 'Yes' or 'No'.\n\nQuestion: "
        f"{question}"
    )
    
    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI that detects if a question is in scope of the given project."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            result = response.choices[0].message.content.strip().lower()
            return result.startswith("yes")

        except RateLimitError as e:
            print(f"Rate limit hit. Attempt {attempt + 1}/{retries}. Retrying in 5 seconds...")
            time.sleep(5)
    
    # After retries fail
    raise Exception("Rate limit exceeded. Please try again later or check your OpenAI quota.")



# Swagger Endpoints
@ns.route('/index')
class IndexRepo(Resource):
    @api.marshal_with(index_response)
    def post(self):
        """Indexes the local repository files"""
        global vector_db
        try:
            vector_db = load_or_create_index()
            return {"message": "Repository indexed successfully!"}
        except Exception as e:
            api.abort(500, str(e))

@ns.route('/query')
class QueryRepo(Resource):
    @api.expect(query_model)
    @api.marshal_with(query_response)
    def post(self):
        """Queries the indexed repository"""
        global vector_db
        if vector_db is None:
            api.abort(400, "No FAISS index loaded. Please index the repository first.")

        query_data = request.json
        query = query_data.get("query")
        if not query:
            api.abort(400, "Query parameter is required.")

        # Out-of-scope detection
        if not is_question_in_scope(query):
            return {"results": [{"content": "This question appears to be out of scope for this repository.", "source": "N/A"}]}

        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        docs_and_scores = vector_db.similarity_search_by_vector(query_embedding[0], k=5)

        results = [{"content": doc.page_content[:500], "source": doc.metadata.get("source")} for doc in docs_and_scores]
        return {"results": results}

if __name__ == "__main__":
    app.run(debug=True)
