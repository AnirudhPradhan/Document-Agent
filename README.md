# 📄 PDF_RAG

Welcome to the **PDF_RAG** project! This repository provides tools and utilities for working with PDF files, enabling efficient processing, analysis, and manipulation.

> **What is RAG (Retrieval-Augmented Generation)?**  
> RAG is a framework that combines information retrieval with generative AI models. Unlike a normal LLM (Large Language Model) that generates responses solely based on its pre-trained knowledge, RAG retrieves relevant context from an external knowledge base (e.g., vector database) and augments the input to the generative model. This improves the quality and relevance of generated responses, making it particularly useful for tasks like question answering, summarization, and document analysis.

## ✨ Features

### 1. 📜 PDF Parsing
- Extract text and metadata from PDF files.
- Supports multi-page PDF documents.

### 2. 🔍 Content Analysis
- Perform keyword searches within PDF content.
- Summarize text from large PDF documents.

### 3. 🔄 File Conversion
- Convert PDFs to other formats (e.g., text, images).
- Batch processing for multiple files.

### 4. ✏️ Customization
- Annotate and edit PDF files.
- Merge or split PDF documents.

## 🛠️ Pipeline Process

1. **📥 Load Document**: Import PDF files into the system.
2. **🧠 Generate Embeddings**: Process the content to create vector embeddings for efficient search and analysis. Embeddings are numerical representations of text that capture semantic meaning, enabling similarity-based searches. Supported embeddings include:
    - Google embeddings

3. **📊 Store in Vector Database**: Save the embeddings in a vector database for quick retrieval. Vector databases allow for fast similarity searches and are optimized for handling high-dimensional data. Supported vector databases include:
    - ChromaDB

4. **🔎 Query and Analyze**: Perform searches, extract insights, or manipulate the content as needed. This step leverages the embeddings and vector database to retrieve relevant information efficiently.


## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AnirudhPradhan/Document-Agent.git
   cd PDF_RAG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Follow the instructions in the terminal to load and process your PDF files.

## 🤝 How to Contribute

We welcome contributions to improve **PDF_RAG**! To contribute:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push your branch to your fork:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request on the main repository.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## 📜 License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software in accordance with the license terms.
