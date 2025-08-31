# 📄 PDF_RAG

Welcome to the **PDF_RAG** project! This repository demonstrates the use of Retrieval-Augmented Generation (RAG) to answer queries specific to any document. By combining information retrieval with generative AI, **PDF_RAG** enables precise and context-aware responses tailored to the content of your uploaded documents.

> **What is RAG (Retrieval-Augmented Generation)?**  
> RAG is a framework that combines information retrieval with generative AI models. Unlike a normal LLM (Large Language Model) that generates responses solely based on its pre-trained knowledge, RAG retrieves relevant context from an external knowledge base (e.g., vector database) and augments the input to the generative model. This improves the quality and relevance of generated responses, making it particularly useful for tasks like question answering, summarization, and document analysis.

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
