# config.py

# Path to the PDF document you want to process
PDF_PATH = r"documents/LBR_SpeechSynthesisWorkshop-NaijaTTS.pdf"

# Path to the Chroma vector database directory
VECTOR_DB_PATH = "./chroma_db"

# Google Generative AI model names
LLM_MODEL_NAME = "gemini-1.5-flash"
EMBEDDINGS_MODEL_NAME = "models/embedding-001"

# Text splitter configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200