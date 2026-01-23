# Project Documentation: Local RAG with Ollama (Text + Media)

Backend file: `backend.py`
UI file: `ui2.py`  

---

## 0. Prerequisites and Setup

This project runs entirely locally. Before running the Streamlit UI, you must install system dependencies, Python packages, and Ollama models.

### 0.1 Required software

- Python 3.10 or newer
- pip (Python package installer)
- Ollama (local model runtime)
- Git (optional, for cloning the project)

### 0.2 System dependencies (recommended)

If you plan to ingest videos, OpenCV must be able to decode common codecs.

On Windows:

- Install a recent Python build and OpenCV via pip

On Linux:

- Install a recent Python build and OpenCV via pip

On macOS:

- Install a recent Python build and OpenCV via pip

### 0.3 Install Ollama and download models

1. Install Ollama from the official installer.
2. Pull the required models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

Notes:

- The default configuration uses:
  - llm_model = "llama3"
  - embedding_model = "nomic-embed-text"
- If you change model names in the Streamlit sidebar, you must pull those models too.

### 0.4 Create a virtual environment (recommended)

From the project folder:

```bash
python -m venv .venv
```

Activate it:

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 0.5 Install Python dependencies

Install required packages:

```bash
pip install streamlit chromadb requests beautifulsoup4 pillow opencv-python sentence-transformers langchain langchain-community langchain-core langchain-text-splitters langchain-ollama pypdf
```

If you want a single requirements file, you can create `requirements.txt` with the packages above and run:

```bash
pip install -r requirements.txt
```

### 0.6 Run the application

From the project folder:

```bash
streamlit run ui2.py
```

Streamlit will open the UI in your browser.

### 0.7 Folder structure

Important paths in DEFAULT_CONFIG:

- `./data`
  - where uploaded files are saved
  - where bulk ingestion reads from
- `./chroma_db`
  - persistent ChromaDB storage folder

After running ingestion, you should expect these folders to exist locally.

---

## 1. Project Overview

This project is a local Retrieval-Augmented Generation (RAG) application that allows users to upload their own files (documents, images, and videos) and then ask questions about them through a chat interface.

The system works in five stages:

1. Ingest content (files or web links)
2. Convert content into embeddings (vectors)
3. Store embeddings in a persistent vector database (ChromaDB)
4. Retrieve the most relevant content at question time
5. Generate an answer using a local LLM via Ollama, constrained to retrieved context

This design reduces hallucinations because the model is instructed to answer only using the retrieved document context.

---

## 2. Key Features

### Supported ingestion inputs

Text:

- PDF (.pdf)
- Text files (.txt)
- Markdown files (.md)
- Web pages via URL (https://...)

Media:

- Images: .png, .jpg, .jpeg
- Videos: .mp4, .mov, .mkv (stored as embedded frames)

### Supported output

- Local answers generated with Ollama
- Source tracking for retrieved chunks
- Retrieval transparency (chunk text and distance scores)
- Related image/video frame retrieval

---

## 3. High-Level Architecture

### Data flow

Upload/Link → Parse → Chunk → Embed → Store → Query → Retrieve → Prompt → Answer

### Vector storage design

The system uses two ChromaDB collections:

- Text collection (embedded using Ollama text embeddings)
- Media collection (embedded using CLIP embeddings)

The collections are separate because text and images require different embedding models.

---

## Backend Documentation (backend.py)

### 4. Configuration

#### DEFAULT_CONFIG

DEFAULT_CONFIG contains all system parameters:

- data_folder: folder used to store ingested files
- persist_directory: folder used to persist ChromaDB data
- embedding_model: Ollama embedding model used for text embeddings
- llm_model: Ollama model used for generating answers
- top_k_retrieval: number of items retrieved for each query
- chunk_size: maximum characters per text chunk
- chunk_overlap: overlapping characters between consecutive chunks
- text_collection: ChromaDB collection name for text chunks
- media_collection: ChromaDB collection name for media items
- allowed_extensions: allowed file types for ingestion
- video_frame_step: sampling interval for video frames
- video_max_frames: limit on how many frames to extract per video

---

## 5. Prompting

### RAG_PROMPT_TEMPLATE

The prompt template forces the model to answer strictly using the retrieved context.

It includes rules:

- Do not use outside knowledge
- If the answer is not present in the context, return exactly:
  "I don't have enough information to answer this question."
- Be concise and accurate

prompt_template is created using LangChain’s PromptTemplate and is later used in RAGSystem.answer().

---

## 6. Utility Functions

### ensure_folder(path: str) -> None

Purpose:
Ensures a directory exists before writing files.

Implementation details:
Uses os.makedirs(path, exist_ok=True). If the directory already exists, no error is raised.

Where it is used:

- Creating the data folder
- Ensuring output directories exist before writing uploads

---

### safe_filename(name: str) -> str

Purpose:
Sanitizes a filename by removing directory separators.

Implementation details:
Replaces “/” and “\” with underscores, preventing directory traversal and broken paths.

---

### extract_text_from_url(url: str) -> str

Purpose:
Downloads and extracts visible readable text from a web page.

Implementation details:

1. Downloads HTML using requests.get() with a timeout and user-agent header
2. Raises an error if the request fails
3. Parses HTML using BeautifulSoup
4. Removes script, style, and noscript tags
5. Returns cleaned text from the page

Why it matters:
This allows the system to ingest web sources as searchable content without needing manual copy/paste.

---

### load_text_from_file(path: str) -> str

Purpose:
Loads text content from local supported file types.

Supported formats:

- PDF: PyPDFLoader is used to extract per-page text
- TXT/MD: TextLoader is used to load the file

Return behavior:
Returns a combined string containing the entire file’s content. If the file type is unsupported, returns an empty string.

---

### chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]

Purpose:
Splits large text into smaller overlapping chunks.

Implementation details:
Uses RecursiveCharacterTextSplitter from LangChain.

Why chunking matters:

- Improves semantic retrieval by making content more granular
- Reduces prompt size during answering
- Overlap helps preserve meaning across boundaries

---

### embed_clip_image(pil_img: Image.Image) -> List[float]

Purpose:
Generates an embedding vector for an image using the CLIP model.

Implementation details:
clip_model.encode(pil_img) produces an embedding, which is converted into a Python list for storage.

---

### extract_video_frames(video_path: str, frame_step: int = 30, max_frames: int = 120) -> List[Dict]

Purpose:
Extracts sampled frames from a video for embedding and retrieval.

Implementation details:

1. Opens video using cv2.VideoCapture
2. Iterates through frames
3. Every frame_step frames, the frame is converted from BGR to RGB and then to a PIL Image
4. Stops when max_frames frames have been saved or the video ends

Return format:
A list of dictionaries with:

- frame_index
- image (PIL Image)

Why this design is used:
Video files are searched by visual similarity using embedded frames rather than by raw file metadata.

---

## 7. Class: TextVectorDB

TextVectorDB wraps a persistent ChromaDB collection for text chunks.

### __init__(persist_directory: str, embedding_model: str, collection_name: str)

Purpose:
Initializes the persistent ChromaDB client and the text embedding model.

Key components:

- chromadb.Client(Settings(persist_directory=...))
- collection = get_or_create_collection(name=...)
- embedder = OllamaEmbeddings(model=...)

Important detail:
The embedding function is not passed directly into ChromaDB. Embeddings are generated in code and then stored manually.

---

### add_texts(texts: List[str], metadatas: List[Dict]) -> int

Purpose:
Adds new text chunks into the vector database.

Steps:

1. Generates a unique UUID for each chunk
2. Generates embeddings using embed_documents(texts)
3. Stores ids, documents, embeddings, and metadatas into the ChromaDB collection

Returns:
Number of chunks added.

---

### search(query: str, top_k: int = 5) -> List[Dict]

Purpose:
Performs semantic search over stored text chunks.

Steps:

1. Generates a query embedding using embed_query(query)
2. Calls ChromaDB query() for nearest neighbors
3. Extracts documents, ids, metadata, and distances
4. Returns a list of result dictionaries with:
   - id
   - text
   - metadata
   - distance

Distance:
Distance values come from ChromaDB similarity search and represent closeness in embedding space.

---

### count() -> int

Purpose:
Returns the number of stored text chunks.

---

### list_sources() -> List[str]

Purpose:
Returns a sorted list of unique “source” values from chunk metadata.

Usage:
Supports the UI source management panel for displaying what files and links exist in the database.

---

### delete_source(source: str) -> int

Purpose:
Deletes all stored chunks that belong to a given source.

Steps:

1. Retrieves ids and metadata
2. Filters ids matching metadata["source"] == source
3. Deletes matching ids
4. Returns number deleted

---

## 8. Class: MediaVectorDB

MediaVectorDB stores and retrieves embeddings for images and video frames.

### __init__(persist_directory: str, collection_name: str)

Purpose:
Initializes persistent ChromaDB client and collection for media items.

---

### add_media_embeddings(documents: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> int

Purpose:
Adds new media embeddings to the database.

Inputs:

- documents: strings such as image paths or “video_path | frame=X”
- embeddings: CLIP embedding vectors
- metadatas: dictionaries including source and type

Returns:
Number of media documents added.

---

### count(), list_sources(), delete_source()

Purpose:
These methods mirror TextVectorDB but apply to media storage.

---

## 9. Class: RAGSystem

RAGSystem orchestrates ingestion, retrieval, and answering.

### __init__(config: Dict)

Purpose:
Initializes the full RAG system.

Creates:

- TextVectorDB instance
- MediaVectorDB instance
- ChatOllama model for responses

temperature is set to 0 for consistent results.

---

### ingest_text_source(source: str) -> int

Purpose:
Ingests text from either a URL or a local file and stores it as embedded chunks.

Steps:

1. Detects URL vs file path
2. Extracts raw text
3. Rejects sources with fewer than 10 characters
4. Chunks text using chunk_text()
5. Creates metadata for each chunk (source, type="text")
6. Adds chunks to TextVectorDB

Returns:
Number of text chunks added.

---

### ingest_image(image_path: str) -> int

Purpose:
Ingests an image and stores its CLIP embedding.

Steps:

1. Loads image using PIL and converts to RGB
2. Embeds with embed_clip_image()
3. Stores the embedding into MediaVectorDB with metadata type="image"

Returns:
Number of stored media items added (typically 1).

---

### ingest_video(video_path: str) -> int

Purpose:
Ingests a video by extracting frames and storing frame embeddings.

Steps:

1. Extract frames using extract_video_frames()
2. For each frame:
   - create a document label "video_path | frame=frame_index"
   - embed image using CLIP
   - store metadata including frame_index
3. Adds all frames to MediaVectorDB

Returns:
Number of stored video frames added.

---

### ingest_file(path: str) -> Dict

Purpose:
Ingests a file automatically based on extension.

Behavior:

- PDF/TXT/MD → ingest_text_source()
- PNG/JPG/JPEG → ingest_image()
- MP4/MOV/MKV → ingest_video()
- otherwise → unsupported

Return format:
{ "kind": "...", "chunks_added": N }

---

### ingest_data_folder() -> Dict

Purpose:
Bulk-ingests all supported files already present inside the data folder.

Steps:

1. Ensures the data folder exists
2. Walks through all files recursively
3. Filters by allowed_extensions
4. Calls ingest_file() for each file
5. Aggregates totals for text chunks and media items

Returns:
{ "text_chunks": ..., "media_items": ..., "files_seen": ... }

---

### retrieve_text(question: str) -> List[Dict]

Purpose:
Retrieves top-k relevant text chunks for a question.

Implementation:
Delegates to TextVectorDB.search().

---

### retrieve_media(question: str) -> List[Dict]

Purpose:
Retrieves top-k related media results for a question.

Implementation:
Delegates to MediaVectorDB.search().

---

### format_text_context(retrieved_chunks: List[Dict]) -> str

Purpose:
Converts retrieved chunks into a structured context string for the LLM prompt.

Output format:
Source i (source_name):
chunk text

This improves traceability and makes source boundaries explicit for the model.

---

### answer(question: str) -> Dict

Purpose:
Generates a final RAG answer for a question.

Steps:

1. Retrieve relevant text chunks
2. Retrieve related media results
3. If no text chunks were found, return the refusal sentence exactly
4. Build context using format_text_context()
5. Format the prompt using prompt_template
6. Invoke the Ollama LLM with ChatOllama.invoke()
7. Collect unique source names from retrieved chunk metadata
8. Return a structured output dictionary containing:
   - answer
   - sources
   - retrieved_chunks
   - retrieved_media

Return format:
{
  "answer": "...",
  "sources": [...],
  "retrieved_chunks": [...],
  "retrieved_media": [...]
}

---

### list_all_sources() -> Dict

Purpose:
Returns the list of stored sources for both text and media collections.

Used by the UI to show the source management list.

---

### delete_source_everywhere(source: str) -> Dict

Purpose:
Deletes a specific source from both databases.

Returns:
{ "deleted_text": ..., "deleted_media": ... }

---

### clear_all() -> None

Purpose:
Deletes the entire persistent ChromaDB directory and recreates empty collections.

This resets the database completely.

---

### save_uploaded_file_to_data(uploaded_file) -> str

Purpose:
Saves a Streamlit uploaded file into the system data folder.

Steps:

1. Ensures the data folder exists
2. Sanitizes the file name
3. Writes file bytes to disk
4. Returns the saved path

---

## 10. Evaluation Utilities

### normalize_text(s: str) -> str

Purpose:
Normalizes text for consistent comparison:

- lowercase
- whitespace normalization
- remove non-alphanumeric characters

Used for overlap scoring.

---

### token_overlap_score(answer: str, context: str) -> float

Purpose:
Computes an approximate grounding score by measuring token overlap between answer and retrieved context.

Steps:

1. Normalize and tokenize answer and context
2. Count token frequency
3. Compute overlap ratio:
   overlap_tokens / answer_tokens

Returns:
A float score between 0.0 and 1.0.

---

### evaluate_rag(system: RAGSystem, eval_questions: List[Dict]) -> List[Dict]

Purpose:
Runs multiple evaluation questions and collects metrics.

For each question:

- generates an answer
- checks whether the system refused
- computes overlap grounding score
- stores chunk and media retrieval counts

This supports debugging and testing retrieval quality.

---

## UI Documentation (ui2.py)

## 11. Streamlit UI Overview

The Streamlit UI provides:

- file upload and ingestion
- link ingestion
- source deletion and management
- a chat interface
- a separate panel showing sources, retrieved chunks, and media hits

---

## 12. UI Functions

### apply_blue_theme()

Purpose:
Injects custom CSS to style the Streamlit app.

This affects:

- background color and gradients
- sidebar styling
- buttons and hover effects
- metric card styling

---

### init_state()

Purpose:
Initializes Streamlit session state variables.

Creates:

- a persistent RAGSystem instance stored as st.session_state.system
- a messages list to store chat history
- state toggles for showing sources, chunks, and media
- a storage slot for the last backend answer output

---

### sidebar_settings(system: RAGSystem)

Purpose:
Displays the configuration panel in the sidebar.

Allows changing:

- LLM model name
- embedding model name
- top-k retrieval
- chunk size
- chunk overlap

Includes two actions:

- Clear chat: resets chat messages only
- Clear DB: deletes all stored embeddings and resets the database

---

### render_header(system: RAGSystem)

Purpose:
Displays the application title, subtitle, and live metrics.

Metrics displayed:

- number of stored text chunks
- number of stored media items
- current top-k value

---

### upload_panel(system: RAGSystem)

Purpose:
Handles file uploads and ingestion from the UI.

Steps:

1. User selects files via st.file_uploader()
2. When the ingestion button is pressed:
   - each file is saved locally using save_uploaded_file_to_data()
   - ingested using ingest_file()
3. Shows per-file success or failure messages
4. Shows a total number of items added

Also includes a button to bulk-ingest all existing files in the data folder.

---

### link_panel(system: RAGSystem)

Purpose:
Allows ingestion of web pages by URL.

Steps:

1. User pastes links, one per line
2. On button press, each link is ingested via ingest_text_source()
3. Shows total chunks added from URLs

---

### sources_panel(system: RAGSystem)

Purpose:
Allows users to delete a specific source from the vector database.

Steps:

1. Collects all text and media sources
2. Shows a select box for choosing a source
3. Deletes it from both databases
4. Resets last_answer and refreshes the UI

---

### chat_feed()

Purpose:
Renders the full chat history.

Special behavior:
If an assistant message is "__THINKING__", it is displayed as a loading response.

---

### chat_input_bar()

Purpose:
Creates the user input area and sends messages into chat history.

Key behavior:
When the user sends a message:

1. Adds the user message to history
2. Adds an assistant placeholder "__THINKING__"
3. Calls st.rerun() so the backend processing can run during the next rerender

---

### process_thinking(system: RAGSystem)

Purpose:
Detects when the assistant is in the "__THINKING__" state and generates the answer.

Steps:

1. Confirms the last chat entry is the thinking token
2. Finds the latest user question
3. Displays a status indicator while the backend runs
4. Calls system.answer(question)
5. Stores backend output in last_answer
6. Replaces the thinking token with the real answer
7. Refreshes the UI

---

### clean_refusal(answer: str) -> str

Purpose:
Ensures the refusal message matches the required exact string.

This supports consistent refusal formatting.

---

### answer_details_panel()

Purpose:
Displays explainability details for the last answer, depending on user toggles.

Supported views:

- sources list
- retrieved chunks with distance values
- related media items with distance and metadata

---

### main()

Purpose:
Entry point for the Streamlit application.

High-level flow:

1. Configure Streamlit page settings
2. Apply CSS theme
3. Initialize session state
4. Build page layout into three columns:
   - ingestion and source management
   - chat interface
   - answer details
5. Runs processing loop via process_thinking()

---

## 13. Conclusion

This project implements a complete local RAG workflow supporting both text and media retrieval. It uses ChromaDB for persistent vector search, Ollama for local embeddings and generation, and CLIP for cross-modal media embedding. The Streamlit UI provides an interactive ingestion and chat experience, with optional retrieval transparency for debugging and explainability.
