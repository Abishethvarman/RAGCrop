##  AG AgriExpert RAG

#### Grownius - ZeroBug

The Agri Related ChatBot wich integrates RAG Pipeline to give domain specific Knowledge

# 🧠 Grownius - ZeroBug: RAG-based Agri Chatbot Architecture

                                +--------------------------+
                                |      👨‍🌾   User          |
                                +-----------+--------------+
                                            |
                                            v
                               +------------+-------------+
                               |     🌐 Frontend (React)   |
                               |  - User Interface         |
                               |  - Sends queries to API   |
                               +------------+-------------+
                                            |
                                            v
                               +------------+-------------+
                               |     🐍 Backend (Python)    |
                               |  - API using Flask/FastAPI|
                               |  - Handles business logic |
                               |  - Connects to RAG module |
                               +------------+-------------+
                                            |
                                            v
                       +--------------------+---------------------+
                       |             🧠 RAG Pipeline              |
                       |  +----------------+   +----------------+ |
                       |  | Retrieval Module |   |  LLM (e.g. GPT) | |
                       |  | (Vector DB)       |   |  Generates answer| |
                       |  +----------------+   +----------------+ |
                       +--------------------+---------------------+
                                            |
                                            v
                          +-----------------+------------------+
                          |  📘 Agricultural Knowledge Base     |
                          |  - Curated Domain Documents         |
                          |  - Used in Retrieval Step           |
                          +-------------------------------------+



You're welcome! Here's a reshaped, professional version of your **"How to Run"** section for the `README.md`, formatted with clean headings, bullet points, and code blocks:

---

## 🏃 How to Run the Project

### 🔧 Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd Backend
   ```

2. Install all required Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:

   ```bash
   python Grownius.py
   ```



### 💻 Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install all necessary Node modules:

   ```bash
   npm install
   ```

3. Start the frontend development server:

   ```bash
   npm run dev
   ```


