# Customer Support RAG Assistant (Experiment 8)

A lightweight full-stack RAG web application built with **FastAPI**, **ChromaDB**, **GroqCloud (Qwen 3.8 27B)**, and a **Material Design 3 rounded frontend** with client-side validation.

---

## 📁 Project Structure

```text
Exp8/
├── app.py              # FastAPI server serving API endpoints & static frontend
├── support_rag.py      # LangChain RAG pipeline & GroqCloud LLM integration
├── requirements.txt    # Python dependencies
├── .env                # API keys and environment variables
├── .env.example        # Environment template
├── .gitignore          # Git exclusion rules
├── data/               # Knowledge base documents
│   ├── faq.docx
│   ├── medication_safety.pdf
│   └── wellness.txt
└── static/
    └── index.html      # Material You 3 frontend (with client-side validation)
```

---

## ⚡ Local Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Groq API key in .env
cp .env.example .env
# Edit .env and insert your GROQ_API_KEY

# 4. Run the app
python app.py
```

Access the frontend at: **`http://127.0.0.1:8000`**  
API documentation at: **`http://127.0.0.1:8000/docs`**
