# Git Repository Question Answering System 🚀

This project is a **REST API** that allows users to ask questions about a given **GitHub repository** and get answers based on the repo’s contents. It uses **OpenAI's language models** and **FAISS** (Facebook AI Similarity Search) for efficient document retrieval.

---

## ✨ Features

- Indexes GitHub repository files.
- Answers natural language questions about the codebase.
- Uses **FAISS** for fast vector-based search over file contents.
- Detects **out-of-scope** questions.
- Handles **OpenAI rate limits** with friendly errors.

---

## 🛠️ Technologies Used

- **Python 3.11+**
- **Flask + Flask-RESTx** for API
- **OpenAI API** for language model
- **FAISS** for vector similarity search

---

## 📦 Installation

1. **Clone the Repository**
   git clone [https://github.com/SangeethaY-GenAI/GIT-QA-BOT.git]
   cd git-qa-system

2.  **Create Virtual Environment**
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3.  **Install Dependencies**
    pip install -r requirements.txt

4.  **Set Up Environment Variables**
    Create a .env file:
    OPENAI_API_KEY=your_openai_api_key
    
**Using FAISS for Vector Search**
When a GitHub repo is indexed, each file’s contents are embedded into vector space using OpenAI’s embeddings, and stored using FAISS for similarity-based retrieval.

**Run the API Server**
python app.py
Default: http://127.0.0.1:5000/

**Open Swagger**
Default: http://127.0.0.1:5000/swagger/

Here first run the index api then go for query api any number of times

**Rate Limit Handling**
When OpenAI API quota is exceeded:
Rate limit exceeded. Please try again later or check your OpenAI quota.



