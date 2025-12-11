# AmbedkarGPT - RAG-based Q&A System

A local Retrieval-Augmented Generation (RAG) system that answers questions based on B.R. Ambedkar's speech using LangChain, ChromaDB, Ollama, and HuggingFace embeddings.


## Requirements

### Software Prerequisites
- Python 3.11
- Ollama 

## Setup Instructions

### Step 1: Install Ollama

**Windows:**
1. Download Ollama from: https://ollama.ai/download
2. Install and run the application

### Step 2: Pull Mistral Model

Open a terminal and run:
```bash
ollama pull mistral
```

### Step 3: Clone the Repository

```bash
git clone https://github.com/YourUsername/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Step 4: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Add Your Speech Text

Create a file named `speech.txt` in the project root and paste B.R. Ambedkar's speech content.


## Running the Application

### Start Ollama Service

Make sure Ollama is running:

**Windows**: Ollama app should be running in system tray

### Run the Q&A System

```bash
python main.py
```

## Project Structure

```
AmbedkarGPT-Intern-Task/
├── main.py            
├── requirements.txt     
├── README.md           
├── speech.txt          
├── chroma_db/          
└── .gitignore          
```

## Technical Details

### Architecture
- **Document Loading**: TextLoader from LangChain
- **Text Splitting**: RecursiveCharacterTextSplitter with 500-char chunks
- **Embeddings**: HuggingFace all-MiniLM-L6-v2 
- **Vector Store**: ChromaDB 
- **LLM**: Ollama Mistral 7B 
- **Chain**: RetrievalQA with custom prompt template
---
