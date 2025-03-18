# config.py
import os

# git source path
GIT_PATH = "https://github.com/vanna-ai/vanna.git"
# Paths
REPO_DIR_PATH = os.path.expanduser("~/vanna")
TRAINING_DATA_PATH = f"{REPO_DIR_PATH}/training_data"
FAISS_INDEX_PATH = "/Users/Sangeetha_Yemisetty/git_qa_system/faiss_index"

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Other Constants
SUPPORTED_FILE_TYPES = (".md", ".py", ".toml", ".cfg", ".json", ".txt")
