import os
import re
import uuid
import shutil
from typing import List, Dict
from collections import Counter

import requests
from bs4 import BeautifulSoup

import chromadb
from chromadb.config import Settings

from PIL import Image
import cv2

from sentence_transformers import SentenceTransformer

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader, TextLoader


from langchain_ollama import ChatOllama


DEFAULT_CONFIG = {
    "data_folder": "./data",
    "persist_directory": "./chroma_db",
    "embedding_model": "nomic-embed-text",
    "llm_model": "llama3",
    "top_k_retrieval": 5,
    "chunk_size": 500,
    "chunk_overlap": 50,
    "text_collection": "rag_text",
    "media_collection": "rag_media",
    "allowed_extensions": [
        ".pdf", ".txt", ".md",
        ".png", ".jpg", ".jpeg",
        ".mp4", ".mov", ".mkv",
    ],
    "video_frame_step": 30,
    "video_max_frames": 120,
}


RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant.
Answer the question using ONLY the text context below.

Text context:
{context}

Question:
{question}

Rules:
- Do NOT use outside knowledge
- If the answer is not in the context, reply with EXACTLY this sentence and nothing else:
  "I don't have enough information to answer this question."
- Be concise and accurate

Answer:
"""

prompt_template = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

clip_model = SentenceTransformer("clip-ViT-B-32")


def ensure_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def extract_text_from_url(url: str) -> str:
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    return soup.get_text(separator=" ", strip=True)


def load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
        return "\n".join([d.page_content for d in docs]).strip()

    if ext in [".txt", ".md"]:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        return "\n".join([d.page_content for d in docs]).strip()

    return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def embed_clip_image(pil_img: Image.Image) -> List[float]:
    emb = clip_model.encode(pil_img)
    return emb.tolist()


def extract_video_frames(video_path: str, frame_step: int = 30, max_frames: int = 120) -> List[Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_index = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            frames.append(
                {
                    "frame_index": frame_index,
                    "image": img,
                }
            )
            saved += 1
            if saved >= max_frames:
                break

        frame_index += 1

    cap.release()
    return frames

class TextVectorDB:
    def __init__(self, persist_directory: str, embedding_model: str, collection_name: str):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self.client = chromadb.Client(Settings(persist_directory=persist_directory))

        # IMPORTANT: do NOT pass embedding_function here
        self.collection = self.client.get_or_create_collection(name=collection_name)

        
        self.embedder = OllamaEmbeddings(model=embedding_model)

    def add_texts(self, texts: List[str], metadatas: List[Dict]) -> int:
        if not texts:
            return 0

        ids = [str(uuid.uuid4()) for _ in texts]

        
        embeddings = self.embedder.embed_documents(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_emb = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
        )

        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            }
            for i in range(len(docs))
        ]

    def count(self) -> int:
        return int(self.collection.count())

    def list_sources(self) -> List[str]:
        n = self.collection.count()
        if n == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])

        sources = []
        for m in metas:
            sources.append((m or {}).get("source", "unknown"))

        return sorted(list(set(sources)))

    def delete_source(self, source: str) -> int:
        data = self.collection.get(include=["metadatas", "ids"])
        ids = data.get("ids", [])
        metas = data.get("metadatas", [])

        to_delete = []
        for _id, m in zip(ids, metas):
            if (m or {}).get("source") == source:
                to_delete.append(_id)

        if not to_delete:
            return 0

        self.collection.delete(ids=to_delete)
        return len(to_delete)


class MediaVectorDB:
    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_media_embeddings(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ) -> int:
        if not documents:
            return 0

        ids = [str(uuid.uuid4()) for _ in documents]
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = clip_model.encode(query).tolist()

        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k)

        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [
            {"id": ids[i], "document": docs[i], "metadata": metas[i], "distance": dists[i]}
            for i in range(len(docs))
        ]

    def count(self) -> int:
        return int(self.collection.count())

    def list_sources(self) -> List[str]:
        n = self.collection.count()
        if n == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])

        sources = []
        for m in metas:
            sources.append((m or {}).get("source", "unknown"))

        return sorted(list(set(sources)))

    def delete_source(self, source: str) -> int:
        data = self.collection.get(include=["metadatas", "ids"])
        ids = data.get("ids", [])
        metas = data.get("metadatas", [])

        to_delete = []
        for _id, m in zip(ids, metas):
            if (m or {}).get("source") == source:
                to_delete.append(_id)

        if not to_delete:
            return 0

        self.collection.delete(ids=to_delete)
        return len(to_delete)


class RAGSystem:
    def __init__(self, config: Dict):
        self.config = config

        self.text_db = TextVectorDB(
            persist_directory=config["persist_directory"],
            embedding_model=config["embedding_model"],
            collection_name=config["text_collection"],
        )

        self.media_db = MediaVectorDB(
            persist_directory=config["persist_directory"],
            collection_name=config["media_collection"],
        )

        self.llm = ChatOllama(model=config["llm_model"], temperature=0)

    def ingest_text_source(self, source: str) -> int:
        raw = ""
        if source.startswith("http://") or source.startswith("https://"):
            raw = extract_text_from_url(source)
        else:
            raw = load_text_from_file(source)

        if not raw or len(raw.strip()) < 10:
            return 0

        chunks = chunk_text(raw, self.config["chunk_size"], self.config["chunk_overlap"])
        metas = [{"source": source, "type": "text"} for _ in chunks]
        return self.text_db.add_texts(chunks, metas)

    def ingest_image(self, image_path: str) -> int:
        img = Image.open(image_path).convert("RGB")
        emb = embed_clip_image(img)

        return self.media_db.add_media_embeddings(
            documents=[image_path],
            embeddings=[emb],
            metadatas=[{"source": image_path, "type": "image"}],
        )

    def ingest_video(self, video_path: str) -> int:
        frames = extract_video_frames(
            video_path,
            frame_step=self.config["video_frame_step"],
            max_frames=self.config["video_max_frames"],
        )

        if not frames:
            return 0

        documents = []
        embeddings = []
        metadatas = []

        for f in frames:
            frame_index = f["frame_index"]
            img = f["image"]

            documents.append(f"{video_path} | frame={frame_index}")
            embeddings.append(embed_clip_image(img))
            metadatas.append(
                {
                    "source": video_path,
                    "type": "video_frame",
                    "frame_index": frame_index,
                }
            )

        return self.media_db.add_media_embeddings(documents, embeddings, metadatas)

    def ingest_file(self, path: str) -> Dict:
        ext = os.path.splitext(path)[-1].lower()

        if ext in [".pdf", ".txt", ".md"]:
            added = self.ingest_text_source(path)
            return {"kind": "text", "chunks_added": added}

        if ext in [".png", ".jpg", ".jpeg"]:
            added = self.ingest_image(path)
            return {"kind": "image", "chunks_added": added}

        if ext in [".mp4", ".mov", ".mkv"]:
            added = self.ingest_video(path)
            return {"kind": "video", "chunks_added": added}

        return {"kind": "unsupported", "chunks_added": 0}

    def ingest_data_folder(self) -> Dict:
        ensure_folder(self.config["data_folder"])

        paths = []
        for root, _, files in os.walk(self.config["data_folder"]):
            for f in files:
                ext = os.path.splitext(f)[-1].lower()
                if ext in self.config["allowed_extensions"]:
                    paths.append(os.path.join(root, f))

        paths = sorted(paths)

        total_text = 0
        total_media = 0

        for p in paths:
            result = self.ingest_file(p)
            if result["kind"] == "text":
                total_text += result["chunks_added"]
            elif result["kind"] in ["image", "video"]:
                total_media += result["chunks_added"]

        return {"text_chunks": total_text, "media_items": total_media, "files_seen": len(paths)}

    def retrieve_text(self, question: str) -> List[Dict]:
        return self.text_db.search(question, top_k=self.config["top_k_retrieval"])

    def retrieve_media(self, question: str) -> List[Dict]:
        return self.media_db.search(question, top_k=self.config["top_k_retrieval"])

    def format_text_context(self, retrieved_chunks: List[Dict]) -> str:
        blocks = []
        for i, ch in enumerate(retrieved_chunks, start=1):
            src = ch.get("metadata", {}).get("source", "unknown")
            txt = ch.get("text", "")
            blocks.append(f"Source {i} ({src}):\n{txt}")
        return "\n\n".join(blocks)

    def answer(self, question: str) -> Dict:
        retrieved_text = self.retrieve_text(question)
        retrieved_media = self.retrieve_media(question)

        if not retrieved_text:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "retrieved_chunks": [],
                "retrieved_media": retrieved_media,
            }

        context = self.format_text_context(retrieved_text)

        prompt = prompt_template.format(context=context, question=question)
        answer = self.llm.invoke(prompt)


        sources = list({c.get("metadata", {}).get("source", "unknown") for c in retrieved_text})

        return {
            "answer": answer.content.strip(),
            "sources": sources,
            "retrieved_chunks": retrieved_text,
            "retrieved_media": retrieved_media,
        }

    def list_all_sources(self) -> Dict:
        return {
            "text_sources": self.text_db.list_sources(),
            "media_sources": self.media_db.list_sources(),
        }

    def delete_source_everywhere(self, source: str) -> Dict:
        deleted_text = self.text_db.delete_source(source)
        deleted_media = self.media_db.delete_source(source)
        return {"deleted_text": deleted_text, "deleted_media": deleted_media}

    def clear_all(self) -> None:
        if os.path.exists(self.config["persist_directory"]):
            shutil.rmtree(self.config["persist_directory"])

        self.text_db = TextVectorDB(
            persist_directory=self.config["persist_directory"],
            embedding_model=self.config["embedding_model"],
            collection_name=self.config["text_collection"],
        )
        self.media_db = MediaVectorDB(
            persist_directory=self.config["persist_directory"],
            collection_name=self.config["media_collection"],
        )

    def save_uploaded_file_to_data(self, uploaded_file) -> str:
        ensure_folder(self.config["data_folder"])
        name = safe_filename(uploaded_file.name)
        save_path = os.path.join(self.config["data_folder"], name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return save_path


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def token_overlap_score(answer: str, context: str) -> float:
    a = normalize_text(answer).split()
    c = normalize_text(context).split()

    if not a or not c:
        return 0.0

    a_counts = Counter(a)
    c_counts = Counter(c)

    overlap = 0
    for t in a_counts:
        overlap += min(a_counts[t], c_counts.get(t, 0))

    return overlap / max(1, len(a))


def evaluate_rag(system: RAGSystem, eval_questions: List[Dict]) -> List[Dict]:
    results = []

    for item in eval_questions:
        q = item["question"]
        must_refuse = item.get("must_refuse", False)

        out = system.answer(q)
        answer = out["answer"]

        refused = "i don't have enough information" in answer.lower()

        context = system.format_text_context(out["retrieved_chunks"])
        overlap = token_overlap_score(answer, context)

        results.append(
            {
                "question": q,
                "answer": answer,
                "refused": refused,
                "must_refuse": must_refuse,
                "grounding_overlap": overlap,
                "text_chunks": len(out["retrieved_chunks"]),
                "media_hits": len(out["retrieved_media"]),
            }
        )

    return results
