import json
import os
import time
import re
import pdfplumber
import hashlib
import numpy as np
import pytesseract
from io import BytesIO
from st_copy import copy_button
import streamlit.components.v1 as components
from datetime import datetime
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black
from pdf2image import convert_from_bytes
from google.genai.errors import ServerError, ClientError
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from streamlit_agraph import (
    agraph,
    Node,
    Edge,
    Config
)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepInFrame
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import (
    TA_LEFT
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PDF_FOLDER = os.path.join(
    BASE_DIR,
    "uploaded_pdfs"
)

os.makedirs(
    PDF_FOLDER,
    exist_ok=True
)



FONT_PATH = os.path.join(
    "fonts",
    "NotoSansDevanagari-Regular.ttf"
)

pdfmetrics.registerFont(
    TTFont(
        "NotoHindi",
        FONT_PATH
    )
)

def show_error(message):

    st.error(
        f"⚠️ {message}"
    )


st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()
RESPONSE_CACHE_FILE = "response_cache.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "Gemini API key is not configured. "
        "Please add GEMINI_API_KEY to your .env file."
    )
    st.stop()

# ============================================
# GEMINI MODEL CONFIGURATION
# ============================================

PDF_TRANSLATION_MODEL = (
    "gemini-3.6-flash"
)

CHAT_MODEL = (
    "gemini-3.6-flash"
)

MAX_AI_RETRIES = 3

PDF_BATCH_MAX_PAGES = 2

PDF_BATCH_MAX_CHARS = 10000

# ==============================================
# CACHED RESOURCES
# ==============================================


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_chroma_client():

    return chromadb.PersistentClient(
        path="chroma_db"
    )


@st.cache_resource
def get_collection():

    client = get_chroma_client()

    return client.get_or_create_collection(
        name="pdf_documents"
    )


@st.cache_resource
def get_gemini_client():

    return genai.Client(
        api_key=GEMINI_API_KEY
    )

gemini_client = get_gemini_client()
embedding_model = load_embedding_model()
chroma_client = get_chroma_client()
collection = get_collection()




# ==================================
# PARSE CONCEPT MAP
# ==================================
def parse_concept_map(response):

    try:

        data = json.loads(
            response
        )

        if not isinstance(
            data,
            dict
        ):
            return None

        concepts = data.get(
            "concepts",
            []
        )

        if not isinstance(
            concepts,
            list
        ):
            return None

        valid_concepts = []

        names = set()

        # --------------------------------------
        # VALIDATE CONCEPTS
        # --------------------------------------

        for concept in concepts:

            if not isinstance(
                concept,
                dict
            ):
                continue

            name = str(
                concept.get(
                    "name",
                    ""
                )
            ).strip()

            parent = concept.get(
                "parent"
            )

            if not name:
                continue

            if name in names:
                continue

            if parent is not None:

                parent = str(
                    parent
                ).strip()

                if not parent:
                    parent = None

            # Concept cannot be its own parent
            if parent == name:
                continue

            valid_concepts.append(
                {
                    "name": name,
                    "parent": parent
                }
            )

            names.add(
                name
            )

        # --------------------------------------
        # BASIC VALIDATION
        # --------------------------------------

        if not valid_concepts:
            return None

        # --------------------------------------
        # EXACTLY ONE ROOT
        # --------------------------------------

        roots = [
            concept
            for concept in valid_concepts
            if concept["parent"] is None
        ]

        if len(roots) != 1:
            return None

        # --------------------------------------
        # CHECK PARENTS EXIST
        # --------------------------------------

        for concept in valid_concepts:

            parent = concept["parent"]

            if (
                parent is not None
                and parent not in names
            ):

                return None

        # --------------------------------------
        # RETURN VALID TREE
        # --------------------------------------

        return valid_concepts

    except Exception:

        return None



# =================================
# build concept levels
# =================================
def build_concept_levels(concepts):

    levels = {}

    roots = [
        concept
        for concept in concepts
        if concept.get("parent") is None
    ]

    if not roots:
        return {}

    root = roots[0]["name"]

    levels[root] = 0

    changed = True

    while changed:

        changed = False

        for concept in concepts:

            name = concept.get(
                "name",
                ""
            )

            parent = concept.get(
                "parent"
            )

            if (
                parent in levels
                and name not in levels
            ):

                levels[name] = (
                    levels[parent] + 1
                )

                changed = True

    return levels


# ==================================================
# TEXT CHUNKING
# ==================================================

def split_text(
    text,
    chunk_size=1000,
    overlap=200
):

    text = " ".join(
        text.split()
    )

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        # Try to end the chunk at a natural boundary
        if end < text_length:

            boundary_positions = [
                text.rfind(".", start, end),
                text.rfind("?", start, end),
                text.rfind("!", start, end),
                text.rfind("\n", start, end)
            ]

            best_boundary = max(
                boundary_positions
            )

            if best_boundary > start + 400:

                end = best_boundary + 1

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        next_start = end - overlap

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# ==================================================
# PDF EXTRACTION WITH PAGE NUMBERS
# ==================================================

def extract_pdf_pages(uploaded_file):

    pdf_reader = PdfReader(
        uploaded_file
    )

    pages = []

    for page_number, page in enumerate(
        pdf_reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages



# ==========================================
# LAYOUT-AWARE PDF EXTRACTION
# ==========================================

def extract_pdf_layout(pdf_file):

    if pdf_file is None:
        return []

    try:

        if hasattr(pdf_file, "read"):
            pdf_file.seek(0)
            pdf_source = pdf_file
        else:
            pdf_source = pdf_file

        pages = []

        with pdfplumber.open(pdf_source) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                page_data = {
                    "page": page_number,
                    "text": "",
                    "tables": []
                }

                found_tables = page.find_tables()

                table_bboxes = []

                for table_obj in found_tables:

                    bbox = table_obj.bbox
                    table_bboxes.append(bbox)

                    table = table_obj.extract()

                    clean_table = []

                    for row in table:

                        if not row:
                            continue

                        clean_row = []

                        for cell in row:

                            cell = str(
                                cell or ""
                            ).strip()

                            clean_row.append(cell)

                        clean_table.append(
                            clean_row
                        )

                    if clean_table:
                        page_data[
                            "tables"
                        ].append(
                            clean_table
                        )

                words = page.extract_words()

                normal_words = []

                for word in words:

                    x0 = word["x0"]
                    top = word["top"]
                    x1 = word["x1"]
                    bottom = word["bottom"]

                    inside_table = False

                    for bbox in table_bboxes:

                        tx0, ttop, tx1, tbottom = bbox

                        if (
                            x0 >= tx0
                            and x1 <= tx1
                            and top >= ttop
                            and bottom <= tbottom
                        ):
                            inside_table = True
                            break

                    if not inside_table:
                        normal_words.append(word)

                lines = {}

                for word in normal_words:

                    line_key = round(
                        word["top"],
                        1
                    )

                    if line_key not in lines:
                        lines[line_key] = []

                    lines[line_key].append(word)

                text_lines = []

                for line_words in sorted(
                    lines.values(),
                    key=lambda x: x[0]["top"]
                ):

                    line_words.sort(
                        key=lambda x: x["x0"]
                    )

                    line = " ".join(
                        word["text"]
                        for word in line_words
                    )

                    if line.strip():
                        text_lines.append(
                            line.strip()
                        )

                page_data["text"] = "\n".join(
                    text_lines
                )

                pages.append(page_data)

        return pages

    except Exception as e:

        st.error(
            f"PDF layout extraction failed: {e}"
        )

        return []


# ===============================================
# PREPARE LAYOUT FOR TRANSLATE
# ===============================================
def prepare_layout_for_translation(layout_pages):

    structured_pages = []

    for page_data in layout_pages:

        page_number = page_data["page"]
        text = page_data.get("text", "")
        tables = page_data.get("tables", [])

        structured_pages.append({
            "page": page_number,
            "text": text,
            "tables": tables
        })

    return structured_pages

# ========================================
# extract_pdf_with_ocr
# =========================================

def extract_pdf_with_ocr(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    images = convert_from_bytes(
        pdf_bytes,
        dpi=200,
        poppler_path=r"C:\Users\malvi\OneDrive\Pictures\Release-26.02.0-0\poppler-26.02.0\Library\bin"
)

    pages = []

    for page_number, image in enumerate(
        images,
        start=1
    ):

        text = pytesseract.image_to_string(
            image
        )

        if text and text.strip():

            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


# ==================================================
# CREATE CHUNKS WITH METADATA
# ==================================================

def create_chunks_with_metadata(pages, filename):

    chunks = []
    metadata = []

    current_section = "General"

    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        lines = text.split("\n")

        current_text = ""

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------
            # Detect possible heading
            # ------------------------------------------

            words = line.split()

            is_heading = (
                len(words) <= 12
                and len(line) <= 100
                and (
                    line.isupper()
                    or line.istitle()
                    or line.endswith(":")
                )
            )

            if is_heading:

                # Save previous text first
                if current_text.strip():

                    page_chunks = split_text(
                        current_text.strip(),
                        chunk_size=1000,
                        overlap=200
                    )

                    for chunk in page_chunks:

                        chunks.append(chunk)

                        metadata.append({
                            "filename": filename,
                            "page": page_number,
                            "section": current_section
                        })

                current_section = line

                current_text = ""

            else:

                current_text += " " + line

        # ------------------------------------------
        # Save remaining page text
        # ------------------------------------------

        if current_text.strip():

            page_chunks = split_text(
                current_text.strip(),
                chunk_size=1000,
                overlap=200
            )

            for chunk in page_chunks:

                chunks.append(chunk)

                metadata.append({
                    "filename": filename,
                    "page": page_number,
                    "section": current_section
                })

    return chunks, metadata

# ===============================================
# Check existing pdf 
# ==============================================
def pdf_already_exists(filename):

    existing_pdfs = get_available_pdfs()

    return filename in existing_pdfs

# ==================================================
# STORE DATA IN CHROMADB
# ==================================================

def store_pdf(
    chunks,
    metadata,
    filename
):

    embeddings = embedding_model.encode(
        chunks
    )

    ids = [
        f"{filename}_{i}"
        for i in range(len(chunks))
    ]

    collection.upsert(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata,
        ids=ids
    )

    return len(chunks)


# ==================================================
# CALCULATE PDF SIMILARITY
# ==================================================

def calculate_pdf_similarity(
    filename1,
    filename2
):

    try:

        result1 = collection.get(
            where={
                "filename": filename1
            },
            include=[
                "embeddings"
            ]
        )

        result2 = collection.get(
            where={
                "filename": filename2
            },
            include=[
                "embeddings"
            ]
        )

        embeddings1 = result1.get(
            "embeddings"
        )

        embeddings2 = result2.get(
            "embeddings"
        )

        if (
            embeddings1 is None
            or embeddings2 is None
        ):

            return None

        if (
            len(embeddings1) == 0
            or len(embeddings2) == 0
        ):

            return None

        # ------------------------------------------
        # CREATE DOCUMENT REPRESENTATIONS
        # ------------------------------------------

        vector1 = np.mean(
            np.array(embeddings1),
            axis=0
        )

        vector2 = np.mean(
            np.array(embeddings2),
            axis=0
        )

        # ------------------------------------------
        # COSINE SIMILARITY
        # ------------------------------------------

        similarity = (
            np.dot(vector1, vector2)
            /
            (
                np.linalg.norm(vector1)
                *
                np.linalg.norm(vector2)
            )
        )

        similarity = max(
            0,
            min(
                1,
                float(similarity)
            )
        )

        return round(
            similarity * 100,
            1
        )

    except Exception:

        return None


# ==================================================
# GET AVAILABLE PDF NAMES
# ==================================================

def get_available_pdfs():

    try:

        results = collection.get(
            include=["metadatas"]
        )

        filenames = set()

        for metadata in results["metadatas"]:

            if metadata:

                filenames.add(
                    metadata["filename"]
                )

        return sorted(filenames)

    except Exception:

        return []
    


# ===================================
# GET PDF INFO
# ===================================
def get_pdf_info(filename):

    try:

        results = collection.get(
            where={
                "filename": filename
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = results.get(
            "documents",
            []
        )

        if not documents:

            return None

        chunk_count = len(
            documents
        )

        character_count = sum(
            len(document)
            for document in documents
            if document
        )

        pages = set()

        for metadata in results.get(
            "metadatas",
            []
        ):

            if metadata:

                page = metadata.get(
                    "page"
                )

                if page is not None:

                    pages.add(
                        str(page)
                    )

        return {
            "filename": filename,
            "pages": len(pages),
            "chunks": chunk_count,
            "characters": character_count,
            "status": "Processed"
        }

    except Exception:

        return None

# ====================================
#  CALCULATED  KEYWORD SCORE
# ====================================
def calculate_keyword_score(
    question,
    document
):

    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            question.lower()
        )
    )

    document_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            document.lower()
        )
    )

    if not question_words:
        return 0.0

    matched_words = (
        question_words.intersection(
            document_words
        )
    )

    return (
        len(matched_words)
        / len(question_words)
    )


#===========================================
#  Reranking function
# ==========================================

def rerank_documents(
    question,
    documents,
    metadatas,
    distances,
    top_k=5
):

    if not documents:

        return (
            [],
            [],
            []
        )

    ranked_results = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        # Keyword score
        keyword_score = calculate_keyword_score(
            question,
            document
        )

        # Convert distance into similarity score
        vector_score = 1 / (
            1 + max(distance, 0)
        )

        # Final score
        final_score = (
            0.7 * vector_score
            + 0.3 * keyword_score
        )

        ranked_results.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "score": final_score
            }
        )

    # Highest score first
    ranked_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Select top results
    ranked_results = ranked_results[
        :min(top_k, len(ranked_results))
    ]

    final_documents = [
        item["document"]
        for item in ranked_results
    ]

    final_metadatas = [
        item["metadata"]
        for item in ranked_results
    ]

    final_distances = [
        item["distance"]
        for item in ranked_results
    ]

    return (
        final_documents,
        final_metadatas,
        final_distances
    )


# ==================================================
# RETRIEVE RELEVANT CHUNKS
# ==================================================

def retrieve_relevant_chunks(
    question,
    filename,
    n_results=8
):

    question_embedding = (
        embedding_model.encode(
            [question]
        )[0]
    )

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=n_results,
        where={
            "filename": filename
        },
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = (
        results.get("documents", [[]])[0]
    )

    metadatas = (
        results.get("metadatas", [[]])[0]
    )

    distances = (
        results.get("distances", [[]])[0]
    )

    if not documents:

        return (
            [],
            [],
            []
        )

    # ==========================================
    # RERANK RESULTS
    # ==========================================

    (
        documents,
        metadatas,
        distances
    ) = rerank_documents(
        question,
        documents,
        metadatas,
        distances,
        top_k=5
    )

    return (
        documents,
        metadatas,
        distances
    )
    
    
    
#====================================
#  retrieve_from_multiple_pdfs
# ================================= 
def retrieve_from_multiple_pdfs(
    question,
    filenames,
    n_results=8
):

    all_documents = []
    all_metadatas = []
    all_distances = []

    for filename in filenames:

        try:

            documents, metadatas, distances = (
                retrieve_relevant_chunks(
                    question,
                    filename,
                    n_results
                )
            )

            all_documents.extend(
                documents
            )

            all_metadatas.extend(
                metadatas
            )

            all_distances.extend(
                distances
            )

        except Exception:

            continue

    combined = list(
        zip(
            all_documents,
            all_metadatas,
            all_distances
        )
    )

    combined.sort(
        key=lambda x: x[2]
    )

    combined = combined[:n_results]

    if not combined:

        return [], [], []

    documents = [
        item[0]
        for item in combined
    ]

    metadatas = [
        item[1]
        for item in combined
    ]

    distances = [
        item[2]
        for item in combined
    ]

    return (
        documents,
        metadatas,
        distances
    )

# =======================================
#  GET ALL PDF CHUNKS
# =====================================
def get_all_pdf_chunks(filename):

    results = collection.get(
        where={
            "filename": filename
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    return (
        results["documents"],
        results["metadatas"]
    )
    
    
# ==========================================
# GROUP PDF CHUNKS BY PAGE
# ==========================================

def get_pdf_pages_for_translation(
    filename
):

    documents, metadatas = (
        get_all_pdf_chunks(
            filename
        )
    )

    if not documents:
        return []

    pages = {}

    for document, metadata in zip(
        documents,
        metadatas
    ):

        page = metadata.get(
            "page",
            1
        )

        try:
            page = int(page)
        except Exception:
            page = 1

        if page not in pages:

            pages[page] = []

        pages[page].append(
            document
        )

    ordered_pages = []

    for page_number in sorted(
        pages.keys()
    ):

        page_text = "\n\n".join(
            pages[page_number]
        )

        ordered_pages.append(
            {
                "page": page_number,
                "text": page_text
            }
        )

    return ordered_pages   




# ==========================================
# TRANSLATION COST ESTIMATOR
# ==========================================

def estimate_translation_cost(
    pages,
    translation_cache
):

    total_pages = len(pages)

    cached_pages = 0
    new_pages = 0
    total_chars = 0

    for page_data in pages:

        page_text = str(
            page_data.get(
                "text",
                ""
            )
        ).strip()

        if not page_text:
            continue

        total_chars += len(
            page_text
        )

        cache_key = page_text

        if cache_key in translation_cache:

            cached_pages += 1

        else:

            new_pages += 1

    # Rough estimate only.
    # Actual Gemini quota depends on
    # tokens, model limits and API quota.

    estimated_requests = new_pages

    return {
        "total_pages": total_pages,
        "cached_pages": cached_pages,
        "new_pages": new_pages,
        "total_chars": total_chars,
        "estimated_requests":
            estimated_requests
    }
    



# ==========================================
# SMART TRANSLATION BATCH BUILDER
# ==========================================

def create_translation_batches(
    pages,
    translation_cache,
    translated_pages=None,
    max_chars=12000,
    max_pages=2
):

    if translated_pages is None:
        translated_pages = {}

    batches = []
    current_batch = []
    current_chars = 0

    for page_data in pages:

        page_number = page_data.get("page")

        page_text = str(
            page_data.get(
                "text",
                ""
            )
        ).strip()

        page_tables = page_data.get(
            "tables",
            []
        )

        # ----------------------------------
        # SKIP COMPLETELY EMPTY PAGE
        # ----------------------------------

        if not page_text and not page_tables:
            continue

        page_key = str(page_number)

        # ----------------------------------
        # CURRENT PAGE FINGERPRINT
        # ----------------------------------

        current_fingerprint = (
            get_page_fingerprint(
                page_data
            )
        )

        saved_page = translated_pages.get(
            page_key
        )

        page_complete = False

        # ----------------------------------
        # VALIDATE CHECKPOINT
        # ----------------------------------

        if isinstance(
            saved_page,
            dict
        ):

            saved_fingerprint = (
                saved_page.get(
                    "fingerprint"
                )
            )

            saved_translation = str(
                saved_page.get(
                    "translation",
                    ""
                )
            ).strip()

            saved_tables = (
                saved_page.get(
                    "translated_tables",
                    []
                )
            )

            # Fingerprint must match
            if (
                saved_fingerprint
                == current_fingerprint
            ):

                page_complete = True

                # ----------------------------------
                # TEXT VALIDATION
                # ----------------------------------

                if page_text and not saved_translation:
                    page_complete = False

                # ----------------------------------
                # TABLE VALIDATION
                # ----------------------------------

                if page_complete and page_tables:

                    if not isinstance(
                        saved_tables,
                        list
                    ):
                        page_complete = False

                    elif len(saved_tables) != len(
                        page_tables
                    ):
                        page_complete = False

                    else:

                        for (
                            original_table,
                            translated_table
                        ) in zip(
                            page_tables,
                            saved_tables
                        ):

                            if not isinstance(
                                translated_table,
                                list
                            ):
                                page_complete = False
                                break

                            if len(
                                original_table
                            ) != len(
                                translated_table
                            ):
                                page_complete = False
                                break

                            for (
                                original_row,
                                translated_row
                            ) in zip(
                                original_table,
                                translated_table
                            ):

                                if not isinstance(
                                    translated_row,
                                    list
                                ):
                                    page_complete = False
                                    break

                                if len(
                                    original_row
                                ) != len(
                                    translated_row
                                ):
                                    page_complete = False
                                    break

                                # ----------------------------------
                                # CELL CONTENT VALIDATION
                                # ----------------------------------

                                for (
                                    original_cell,
                                    translated_cell
                                ) in zip(
                                    original_row,
                                    translated_row
                                ):

                                    original_cell = str(
                                        original_cell
                                    ).strip()

                                    translated_cell = str(
                                        translated_cell
                                    ).strip()

                                    if (
                                        original_cell
                                        and not translated_cell
                                    ):
                                        page_complete = False
                                        break

                                if not page_complete:
                                    break

                            if not page_complete:
                                break

        # ----------------------------------
        # PAGE ALREADY COMPLETE
        # ----------------------------------

        if page_complete:
            continue

        # ----------------------------------
        # CREATE PAGE BLOCK
        # ----------------------------------

        page_block = (
            f"<<<PAGE {page_number}>>>\n"
            f"{page_text}\n"
            f"<<<END PAGE {page_number}>>>"
        )

        page_size = len(
            page_block
        )

        # ----------------------------------
        # START NEW BATCH
        # ----------------------------------

        if (
            current_batch
            and (
                current_chars + page_size
                > max_chars
                or len(current_batch)
                >= max_pages
            )
        ):

            batches.append(
                current_batch
            )

            current_batch = []
            current_chars = 0

        # ----------------------------------
        # ADD PAGE
        # ----------------------------------

        current_batch.append({
            "page":
                page_number,

            "text":
                page_text,

            "tables":
                page_tables,

            "block":
                page_block
        })

        current_chars += page_size

    # ----------------------------------
    # LAST BATCH
    # ----------------------------------

    if current_batch:

        batches.append(
            current_batch
        )

    return batches

# ===============================================
# TRANSLATE TABLE
# ================================================
def translate_table(
    table,
    target_language="Hindi"
):

    if not table:
        return []

    original_rows = []

    for row in table:

        original_rows.append([
            str(cell or "").strip()
            for cell in row
        ])

    row_count = len(original_rows)

    column_counts = [
        len(row)
        for row in original_rows
    ]

    if not column_counts:
        return []

    expected_columns = max(
        set(column_counts),
        key=column_counts.count
    )

    prompt = f"""
You are a professional technical PDF translator.

Translate the table from English into natural, clear and professional Hindi.

The table structure MUST remain exactly unchanged.

Original table:
- Rows: {row_count}
- Expected columns: {expected_columns}

STRICT STRUCTURE RULES:

1. Return ONLY valid JSON.
2. Return exactly {row_count} rows.
3. Preserve the exact order of rows.
4. Each row MUST contain exactly the same number of cells as its
   corresponding original row.
5. Do NOT add rows.
6. Do NOT remove rows.
7. Do NOT add columns.
8. Do NOT remove columns.
9. Do NOT merge cells.
10. Do NOT split cells.
11. Do NOT reorder cells.

TRANSLATION RULES:

12. Translate normal explanatory English text into natural Hindi.
13. Preserve the exact meaning.
14. Do not summarize.
15. Do not add information.
16. Do not remove information.
17. Avoid literal or awkward word-for-word translation.
18. Use simple, professional Hindi suitable for computer science
    students.

TECHNICAL TERMS:

Keep technical terms in English when translating them would make
the technical meaning less clear.

Keep terms such as:

Linux
command
terminal
shell
Bash
filesystem
directory
file
process
job
PID
CPU
RAM
memory
storage
disk
partition
package
service
kernel
environment variable
script
function
argument
parameter
output
input
permission
user
group
root
server
network
socket
port
URL
API
DNS
SSH
HTTP
HTTPS
TLS
compiler
debugging
pipeline
redirection
cron
OpenSSL
Git
Python
Node.js
JavaScript
Makefile
ACL

COMMANDS AND CODE:

19. Keep all Linux commands EXACTLY unchanged.
20. Keep command options EXACTLY unchanged.
21. Keep file paths EXACTLY unchanged.
22. Keep URLs EXACTLY unchanged.
23. Keep numbers EXACTLY unchanged.
24. Keep code EXACTLY unchanged.
25. Keep programming syntax EXACTLY unchanged.
26. Keep Linux operators EXACTLY unchanged.

For example, these MUST remain unchanged:

pwd
ls -la
dmesg | tail
ps aux | grep nginx
echo hello > file.txt
command > all.log 2>&1
mkdir app && cd app

IMPORTANT:

A "|" character inside a cell is part of the cell content.
It MUST NOT be treated as a column separator.

Do not modify such commands.

Return ONLY a JSON array.

Example:

[
  ["pwd", "वर्तमान working directory दिखाता है।", "अपनी स्थिति जानने के लिए उपयोग करें।", "pwd"],
  ["ls", "फ़ाइलों और directories की सूची दिखाता है।", "वर्तमान directory की सामग्री देखने के लिए उपयोग करें।", "ls -la"]
]

ORIGINAL TABLE:

{json.dumps(original_rows, ensure_ascii=False)}
"""

    response = safe_generate_content(
        prompt,
        feature="PDF Table Translation"
    )

    response = response.strip()

    # ----------------------------------
    # REMOVE MARKDOWN CODE FENCES
    # ----------------------------------

    if response.startswith("```"):

        response = response.replace(
            "```json",
            "",
            1
        )

        response = response.replace(
            "```",
            ""
        )

        response = response.strip()

    # ----------------------------------
    # PARSE JSON
    # ----------------------------------

    try:

        translated_table = json.loads(
            response
        )

    except Exception as e:

        raise RuntimeError(
            "Invalid JSON returned for "
            "table translation: "
            + str(e)
        )

    # ----------------------------------
    # BASIC VALIDATION
    # ----------------------------------

    if not isinstance(
        translated_table,
        list
    ):

        raise RuntimeError(
            "Table translation did not "
            "return a JSON list."
        )

    if len(translated_table) != row_count:

        raise RuntimeError(
            "Table row count changed. "
            f"Expected {row_count}, "
            f"got {len(translated_table)}."
        )

    # ----------------------------------
    # COLUMN VALIDATION
    # ----------------------------------

    for index, row in enumerate(
        translated_table
    ):

        if not isinstance(
            row,
            list
        ):

            raise RuntimeError(
                f"Table row {index + 1} "
                "is not a JSON list."
            )

        original_column_count = len(
            original_rows[index]
        )

        if len(row) != original_column_count:

            raise RuntimeError(
                f"Table column count changed "
                f"on row {index + 1}. "
                f"Expected "
                f"{original_column_count}, "
                f"got {len(row)}."
            )

    # ----------------------------------
    # CLEAN CELLS
    # ----------------------------------

    cleaned_table = []

    for row in translated_table:

        cleaned_row = []

        for cell in row:

            if cell is None:
                cell = ""

            cleaned_row.append(
                str(cell).strip()
            )

        cleaned_table.append(
            cleaned_row
        )

    return cleaned_table 
    

# =====================================
#  remove_duplicate_chunks
# =====================================

def remove_duplicate_chunks(
    documents
):

    unique_documents = []
    seen = set()

    for document in documents:

        document = document.strip()

        if not document:
            continue

        # Normalize whitespace
        normalized = " ".join(
            document.split()
        ).lower()

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        unique_documents.append(
            document
        )

    return unique_documents


# =================================
#  GET ADAPTIVE RESULT COUNT
#=================================
def get_adaptive_result_count(
    question
):

    words = question.split()

    length = len(words)

    if length <= 6:

        return 4

    elif length <= 15:

        return 6

    else:

        return 8
    
# ========================================
def get_document_context(
    filename,
    question=None,
    n_results=None
):

    try:

        if question:
            
            if n_results is None:
                n_results = (
                    get_adaptive_result_count(
                        question
                    )
                )

            query_embedding = (
                embedding_model.encode(
                    [question]
                )[0]
            )

            results = collection.query(
                query_embeddings=[
                    query_embedding.tolist()
                ],
                n_results=n_results,
                where={
                    "filename": filename
                },
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

            documents = results.get(
                "documents",
                [[]]
            )[0]
            
            distances = results.get(
                "distances",
                [[]]
            )[0]

            filtered_documents = []

            for document, distance in zip(
                documents,
                distances
            ):

                if distance <= 0.75:

                    filtered_documents.append(
                        document
                    )

            documents = filtered_documents
            

            documents = remove_duplicate_chunks(
                documents
            )

            return documents

        documents, metadatas = (
            get_all_pdf_chunks(
                filename
            )
        )

        if not documents:
            return []

        return documents[:n_results]

    except Exception:

        return []

# ======================================
#  CREATE CHUNKS GROUP
# ==================================

def create_chunk_groups(
    documents,
    max_chars=8000
):

    groups = []
    current_group = []
    current_size = 0

    for document in documents:

        document_size = len(document)

        if (
            current_size + document_size
            > max_chars
            and current_group
        ):

            groups.append(
                "\n\n".join(current_group)
            )

            current_group = []
            current_size = 0

        current_group.append(document)

        current_size += document_size

    if current_group:

        groups.append(
            "\n\n".join(current_group)
        )

    return groups


# ==========================================
# TRANSLATION CHUNK GROUPS
# ==========================================

def create_translation_groups(
    documents,
    max_chars=10000
):

    groups = []

    current_group = []
    current_size = 0

    for document in documents:

        document = str(
            document
        ).strip()

        if not document:
            continue

        document_size = len(
            document
        )

        # --------------------------------------
        # LARGE DOCUMENT
        # --------------------------------------

        if (
            document_size > max_chars
            and not current_group
        ):

            start = 0

            while start < document_size:

                part = document[
                    start:start + max_chars
                ]

                groups.append(
                    part
                )

                start += max_chars

            continue

        # --------------------------------------
        # START NEW GROUP
        # --------------------------------------

        if (
            current_size + document_size
            > max_chars
            and current_group
        ):

            groups.append(
                "\n\n".join(
                    current_group
                )
            )

            current_group = []
            current_size = 0

        current_group.append(
            document
        )

        current_size += (
            document_size
        )

    # --------------------------------------
    # LAST GROUP
    # --------------------------------------

    if current_group:

        groups.append(
            "\n\n".join(
                current_group
            )
        )

    return groups


# ==================================
# SUMMERIZE CHUNKS GROUP
# =================================
def summarize_chunk_group(
    context
):

    prompt = f"""
You are analyzing part of a PDF.

Extract the important information from
this section.

CONTENT:
-------------------------
{context}
-------------------------

Instructions:

- Use only the provided content.
- Preserve important definitions,
  concepts and facts.
- Do not add outside information.
- Keep the result concise.
"""

    return safe_generate_content(
        prompt
    )


# ================================
#  COMBINE SUMMARIZE
# ================================

def combine_summaries(
    summaries
):

    combined = "\n\n".join(
        summaries
    )

    prompt = f"""
You are creating the final summary
of a PDF.

Below are summaries of different
sections of the document.

SECTION SUMMARIES:
-------------------------
{combined}
-------------------------

Create one clear final summary.

Instructions:

- Combine related information.
- Remove repetition.
- Preserve important concepts.
- Use headings and bullet points.
- Do not add outside information.
- Make it useful for study and revision.

Return only the final summary.
"""

    return safe_generate_content(
        prompt
    )


# ====================================
#  GENERATE LARGE DOCUMENTS SUMMARY
# =======================================
def generate_large_document_summary(
    filename
):

    documents = get_document_context(
        filename
    )

    if not documents:

        return None

    groups = create_chunk_groups(
        documents
    )

    summaries = []

    for group in groups:

        summary = summarize_chunk_group(
            group
        )

        summaries.append(
            summary
        )

    return combine_summaries(
        summaries
    )

# ==================================
#  GENERATE QUIZ
# ==================================
def generate_quiz(context):

    prompt = f"""
You are an AI exam preparation assistant.

Create exactly 5 multiple-choice questions
from the provided PDF content.

PDF CONTENT:
-------------------------
{context}
-------------------------

Return ONLY valid JSON.

Use exactly this structure:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": 0,
    "explanation": "Short explanation"
  }}
]

Rules:

- answer must be 0, 1, 2, or 3.
- 0 means first option.
- 1 means second option.
- 2 means third option.
- 3 means fourth option.
- Use only information from the PDF.
- Do not add outside information.
- Questions should test important concepts.
- Avoid ambiguous questions.
- Mix easy and medium difficulty.
"""

    return safe_generate_content(prompt)


# ====================================
# GENERATE SUGGESTED QUESTIONS
# ===================================

def generate_suggested_questions(
    context
):

    prompt = f"""
You are an AI document assistant.

Based only on the following PDF content,
generate 4 useful questions that a user
could ask about this document.

PDF CONTENT:
-------------------------
{context}
-------------------------

Rules:

- Use only information from the PDF.
- Questions should cover important topics.
- Keep questions short.
- Avoid duplicate questions.
- Return ONLY a JSON array of strings.

Example:

[
    "What is OOP?",
    "What are the four pillars of OOP?",
    "Explain inheritance",
    "What is polymorphism?"
]
"""

    return safe_generate_content(
        prompt
    )

# ===================================
#  parse quiz
# ==================================
def parse_quiz(quiz_text):

    try:

        return json.loads(
            quiz_text
        )

    except json.JSONDecodeError:

        start = quiz_text.find("[")

        end = quiz_text.rfind("]")

        if start != -1 and end != -1:

            return json.loads(
                quiz_text[start:end + 1]
            )

        return []
    
  
# ==============================
# PARSE SUGGESTED QUESTIONS
# ============================== 
def parse_suggested_questions(
    response_text
):

    try:

        return json.loads(
            response_text
        )

    except json.JSONDecodeError:

        start = response_text.find("[")

        end = response_text.rfind("]")

        if start != -1 and end != -1:

            try:

                return json.loads(
                    response_text[
                        start:end + 1
                    ]
                )

            except json.JSONDecodeError:

                return []

        return []


# ===================================
# DELETE PDF
# ==================================
def delete_pdf(filename):

    # ------------------------------------------
    # DELETE CHROMADB CHUNKS
    # ------------------------------------------

    collection.delete(
        where={
            "filename": filename
        }
    )

    # ------------------------------------------
    # UPDATE CHAT HISTORIES
    # ------------------------------------------

    if "chat_histories" in st.session_state:

        old_histories = (
            st.session_state.chat_histories
        )

        new_histories = {}

        for chat_key, history in old_histories.items():

            pdfs = chat_key.split("|")

            # Deleted PDF is not part of this chat
            if filename not in pdfs:

                new_histories[chat_key] = history

                continue

            # Remove deleted PDF
            remaining_pdfs = [
                pdf
                for pdf in pdfs
                if pdf != filename
            ]

            # If no PDF remains, discard this history
            if not remaining_pdfs:

                continue

            # Create new key for remaining PDFs
            new_key = "|".join(
                sorted(remaining_pdfs)
            )

            # Preserve existing history
            if new_key in new_histories:

                new_histories[new_key].extend(
                    history
                )

            else:

                new_histories[new_key] = (
                    history.copy()
                )

        st.session_state.chat_histories = (
            new_histories
        )


# ==================================
#  CLEAR CURRENT CHAT
# ==================================
def clear_current_chat():

    if "chat_histories" not in st.session_state:
        return

    selected_pdfs = st.session_state.get(
        "selected_pdfs",
        []
    )

    if not selected_pdfs:
        return

    chat_key = "|".join(
        sorted(selected_pdfs)
    )

    st.session_state.chat_histories.pop(
        chat_key,
        None
    )

# ================================
#  is follow up
# =====================================

def is_follow_up(question):

    follow_up_words = [
        "it",
        "its",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "his",
        "her",
        "above",
        "previous",
        "earlier"
    ]

    question_lower = question.lower()

    for word in follow_up_words:

        if f" {word} " in f" {question_lower} ":
            return True

    return False


# ==========================================
# CAN MAKE AI REQUEST
# ==========================================

def can_make_ai_request(
    cooldown=2
):

    current_time = time.time()

    elapsed = (
        current_time
        - st.session_state.last_ai_request_time
    )

    if elapsed < cooldown:

        return False

    st.session_state.last_ai_request_time = (
        current_time
    )

    return True

# ===============safe generate content==============

def safe_generate_content(
    prompt,
    feature="Unknown"
):

    # ==========================================
    # LOCAL QUOTA PROTECTION
    # ==========================================

    if check_quota_protection():

        raise RuntimeError(
            "⚠️ AI quota protection is active. "
            "Please wait before making more AI requests."
        )

    # ==========================================
    # MODEL ROUTING
    # ==========================================

    if feature in (
        "PDF Batch Translation",
        "PDF Table Translation",
        "PDF Translation Repair"
    ):

        model_name = (
            PDF_TRANSLATION_MODEL
        )

    else:

        model_name = (
            CHAT_MODEL
        )

    # ==========================================
    # RETRIES
    # ==========================================

    for attempt in range(
        MAX_AI_RETRIES
    ):

        try:

            response = (
                gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
            )

            # ==================================
            # RESPONSE VALIDATION
            # ==================================

            if response is None:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            response_text = getattr(
                response,
                "text",
                None
            )

            if not response_text:

                raise RuntimeError(
                    "Gemini returned empty text."
                )

            # ==================================
            # COUNT SUCCESSFUL CALL
            # ==================================

            track_ai_usage(
                "gemini_call",
                feature
            )

            return response_text.strip()

        except Exception as e:

            error_text = str(e)

            # ==================================
            # FAILED REQUEST
            # ==================================

            track_ai_usage(
                "failed",
                feature
            )

            # ==================================
            # QUOTA / RATE LIMIT
            # ==================================

            if (
                "429" in error_text
                or
                "RESOURCE_EXHAUSTED"
                in error_text
                or
                "quota"
                in error_text.lower()
            ):

                raise RuntimeError(
                    "Gemini API quota or rate limit "
                    "was exceeded. "
                    "Please wait for the quota reset."
                )

            # ==================================
            # TEMPORARY SERVER ERROR
            # ==================================

            if (
                "503" in error_text
                or
                "UNAVAILABLE"
                in error_text
                or
                "timeout"
                in error_text.lower()
                or
                "timed out"
                in error_text.lower()
            ):

                if (
                    attempt
                    < MAX_AI_RETRIES - 1
                ):

                    wait_time = (
                        2 ** attempt
                    )

                    time.sleep(
                        wait_time
                    )

                    continue

                raise RuntimeError(
                    "Gemini service is temporarily "
                    "unavailable after multiple retries."
                )

            # ==================================
            # MODEL ERROR
            # ==================================

            if (
                "404" in error_text
                or
                "NOT_FOUND"
                in error_text
            ):

                raise RuntimeError(
                    f"Gemini model "
                    f"'{model_name}' is unavailable."
                )

            # ==================================
            # OTHER ERROR
            # ==================================

            raise RuntimeError(
                "Gemini request failed. "
                "Please try again later."
            )

    raise RuntimeError(
        "Gemini request failed after retries."
    )

# ====================================
# GET CACHE KEY
# ====================================
def get_cache_key(
    question,
    context="",
    response_language="English",
    mode=""
):

    data = (
        f"{question}|"
        f"{context}|"
        f"{response_language}|"
        f"{mode}"
    )

    return hashlib.sha256(
        data.encode("utf-8")
    ).hexdigest()
    


# ====================================
# COMPRESS CONTEXT
# ====================================
def compress_context(
    context,
    max_chars=12000
):

    if not context:

        return ""

    context = context.strip()

    if len(context) <= max_chars:

        return context

    parts = context.split(
        "\n\n"
    )

    compressed = []
    current_length = 0

    for part in parts:

        part = part.strip()

        if not part:
            continue

        part_length = len(part)

        if (
            current_length
            + part_length
            + 2
            > max_chars
        ):
            break

        compressed.append(
            part
        )

        current_length += (
            part_length + 2
        )

    return "\n\n".join(
        compressed
    )

# ===============================
# GET CONTENT BUDGET
# ==============================

def get_context_budget(
    question,
    base_limit=8000,
    max_limit=16000
):

    word_count = len(
        question.split()
    )

    if word_count <= 6:

        return base_limit

    if word_count <= 15:

        return 12000

    return max_limit

# .===========================
# LOAD RESPONSE CACHE
# =============================
def load_response_cache():

    if not os.path.exists(
        RESPONSE_CACHE_FILE
    ):

        return {}

    try:

        with open(
            RESPONSE_CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


# ===============================
#  SAVE TO RESPONSE CACHE
# ===============================
def save_to_response_cache(
    cache_key,
    response,
    max_items=100
):

    cache = st.session_state.response_cache

    cache[cache_key] = response

    while len(cache) > max_items:

        oldest_key = next(
            iter(cache)
        )

        del cache[oldest_key]

    try:

        with open(
            RESPONSE_CACHE_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                cache,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception:

        pass


# ==========================
# TRACK AI USAGE
# ==========================
def track_ai_usage(
    usage_type,
    feature="Chat"
):

    usage = st.session_state.ai_usage

    if usage_type == "request":

        usage["total_requests"] += 1

    elif usage_type == "cache_hit":

        usage["cache_hits"] += 1

    elif usage_type == "gemini_call":

        usage["gemini_calls"] += 1

    elif usage_type == "failed":

        usage["failed_requests"] += 1

    # --------------------------------------
    # FEATURE-WISE TRACKING
    # --------------------------------------

    if feature not in usage["features"]:

        usage["features"][feature] = {
            "requests": 0,
            "cache_hits": 0,
            "gemini_calls": 0,
            "failed": 0
        }

    feature_usage = (
        usage["features"][feature]
    )

    if usage_type == "request":

        feature_usage["requests"] += 1

    elif usage_type == "cache_hit":

        feature_usage["cache_hits"] += 1

    elif usage_type == "gemini_call":

        feature_usage["gemini_calls"] += 1

    elif usage_type == "failed":

        feature_usage["failed"] += 1




# =============================
#  CHECK QUOTA PROTECTIONS
# =============================
def check_quota_protection(
    limit=80
):

    gemini_calls = (
        st.session_state.ai_usage.get(
            "gemini_calls",
            0
        )
    )

    return gemini_calls >= limit


# ===============================
#  get ai usage status
# ===============================
def get_ai_usage_status():

    gemini_calls = (
        st.session_state.ai_usage.get(
            "gemini_calls",
            0
        )
    )

    if gemini_calls >= 80:

        return (
            "🔴 Protection Mode",
            "AI requests are temporarily protected."
        )

    elif gemini_calls >= 60:

        return (
            "🟡 Save AI Mode",
            "Using aggressive caching and retrieval optimization."
        )

    else:

        return (
            "🟢 Normal Mode",
            "AI features are running normally."
        )

# =====================================
#  get_ai_optimization_settings 
# ====================================
def get_ai_optimization_settings():

    status, _ = get_ai_usage_status()

    if "Protection" in status:

        return {
            "max_chunks": 3,
            "context_limit": 6000
        }

    elif "Save AI" in status:

        return {
            "max_chunks": 4,
            "context_limit": 8000
        }

    return {
        "max_chunks": 6,
        "context_limit": 12000
    }


# =================================
#  GENERATE AI USAGE INSIGHTS
# ==================================
def generate_ai_usage_insights():

    usage = st.session_state.ai_usage

    total = usage.get(
        "total_requests",
        0
    )

    cache_hits = usage.get(
        "cache_hits",
        0
    )

    gemini_calls = usage.get(
        "gemini_calls",
        0
    )

    failed = usage.get(
        "failed_requests",
        0
    )

    features = usage.get(
        "features",
        {}
    )

    insights = []

    # ------------------------------------------
    # CACHE INSIGHT
    # ------------------------------------------

    if total > 0:

        cache_rate = (
            cache_hits / total
        ) * 100

        if cache_rate >= 50:

            insights.append(
                f"✓ {cache_rate:.0f}% of requests "
                "were served from cache."
            )

        elif cache_rate > 0:

            insights.append(
                f"💾 Cache is currently serving "
                f"{cache_rate:.0f}% of requests."
            )

    # ------------------------------------------
    # MOST USED FEATURE
    # ------------------------------------------

    if features:

        most_used = max(
            features.items(),
            key=lambda item: item[1].get(
                "requests",
                0
            )
        )

        feature_name = most_used[0]

        feature_requests = (
            most_used[1].get(
                "requests",
                0
            )
        )

        insights.append(
            f"📌 Most used AI feature: "
            f"{feature_name} "
            f"({feature_requests} requests)."
        )

    # ------------------------------------------
    # GEMINI USAGE
    # ------------------------------------------

    if gemini_calls > 0:

        insights.append(
            f"🤖 Gemini was called "
            f"{gemini_calls} times."
        )

    # ------------------------------------------
    # FAILED REQUESTS
    # ------------------------------------------

    if failed > 0:

        insights.append(
            f"⚠️ {failed} AI requests failed."
        )

    # ------------------------------------------
    # NO DATA
    # ------------------------------------------

    if not insights:

        insights.append(
            "💡 Start using AI features to "
            "generate usage insights."
        )

    return insights


# ===============================
# SHOW AI USAGE DASHBOARD
# =================================
def show_ai_usage_dashboard():

    usage = st.session_state.ai_usage
    
    status, status_text = (
        get_ai_usage_status()
    )

    st.markdown(
        f"### {status}"
    )

    st.caption(
        status_text
    )

    total = usage.get(
        "total_requests",
        0
    )

    cache_hits = usage.get(
        "cache_hits",
        0
    )

    gemini_calls = usage.get(
        "gemini_calls",
        0
    )

    failed = usage.get(
        "failed_requests",
        0
    )

    # --------------------------------------
    # CACHE HIT RATE
    # --------------------------------------

    if total > 0:

        cache_rate = (
            cache_hits / total
        ) * 100

    else:

        cache_rate = 0

    st.markdown(
        "## 📊 AI Usage Dashboard"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Requests",
            total
        )

    with col2:

        st.metric(
            "Cache Hits",
            cache_hits
        )

    with col3:

        st.metric(
            "Gemini Calls",
            gemini_calls
        )

    with col4:

        st.metric(
            "Failed",
            failed
        )

    st.progress(
        min(
            cache_rate / 100,
            1.0
        )
    )

    st.caption(
        f"💾 Cache Hit Rate: "
        f"{cache_rate:.1f}%"
    )

    # --------------------------------------
    # FEATURE USAGE
    # --------------------------------------

    features = usage.get(
        "features",
        {}
    )

    if features:

        st.markdown(
            "### 🔎 Feature Usage"
        )

        rows = []

        for feature, data in features.items():

            rows.append(
                {
                    "Feature": feature,
                    "Requests": data.get(
                        "requests",
                        0
                    ),
                    "Cache Hits": data.get(
                        "cache_hits",
                        0
                    ),
                    "Gemini Calls": data.get(
                        "gemini_calls",
                        0
                    ),
                    "Failed": data.get(
                        "failed",
                        0
                    )
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )
        
        
    # --------------------------------------
    # AI INSIGHTS
    # --------------------------------------

    st.markdown(
        "### 💡 AI Usage Insights"
    )

    insights = (
        generate_ai_usage_insights()
    )

    for insight in insights:

        st.write(
            insight
        )

# ================================
# IS REQUEST PROCEESING
# ================================
def is_request_processing(
    cache_key
):

    return (
        cache_key
        in st.session_state.processing_requests
    )


# =========== rewrite question===============

def rewrite_question(question, chat_history):

    history_text = ""

    for message in chat_history[-6:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are a question rewriting assistant.

Your job is to convert the user's latest question
into a standalone question that can be understood
without previous conversation.

Conversation History:
-------------------------
{history_text}
-------------------------

Latest Question:
{question}

Rewrite the latest question so that all references
such as "it", "its", "they", "this", "that", etc.
are replaced with the actual topic from the conversation.

If the question is already standalone,
return it unchanged.

Return ONLY the rewritten question.
Do not provide an answer.
"""

    response = safe_generate_content(prompt)

    return response.strip()



# ==================================================
# LOCAL SENTENCE RELEVANCE
# ==================================================

def extract_relevant_sentences(
    question,
    documents,
    max_sentences=6
):

    if not documents:
        return []

    # ------------------------------------------
    # QUESTION KEYWORDS
    # ------------------------------------------

    question_words = set(
        word.lower().strip(
            ".,!?;:()[]{}\"'"
        )
        for word in question.split()
        if len(word) > 2
    )

    scored_sentences = []

    # ------------------------------------------
    # PROCESS DOCUMENTS
    # ------------------------------------------

    for document_index, document in enumerate(
        documents
    ):

        # Basic sentence splitting
        sentences = re.split(
            r"(?<=[.!?])\s+",
            document
        )

        for sentence_index, sentence in enumerate(
            sentences
        ):

            sentence = sentence.strip()

            if len(sentence) < 20:
                continue

            sentence_words = set(
                word.lower().strip(
                    ".,!?;:()[]{}\"'"
                )
                for word in sentence.split()
            )

            # ----------------------------------
            # KEYWORD SCORE
            # ----------------------------------

            matched_words = (
                question_words
                & sentence_words
            )

            score = len(
                matched_words
            )

            if score == 0:
                continue

            scored_sentences.append(
                (
                    score,
                    document_index,
                    sentence_index,
                    sentence
                )
            )

    # ------------------------------------------
    # SORT BY RELEVANCE
    # ------------------------------------------

    scored_sentences.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2]
        )
    )

    # ------------------------------------------
    # REMOVE DUPLICATES
    # ------------------------------------------

    selected = []
    seen = set()

    for (
        score,
        document_index,
        sentence_index,
        sentence
    ) in scored_sentences:

        normalized = " ".join(
            sentence.lower().split()
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        selected.append(
            sentence
        )

        if len(selected) >= max_sentences:
            break

    return selected 

# ==================================================
# LOCAL PDF FALLBACK
# ==================================================

def local_pdf_fallback(
    question,
    documents,
    metadatas,
    distances,
    response_language="English"
):

    if not documents:

        return (
            "I could not find relevant information "
            "in the selected PDF."
        )

    # ------------------------------------------
    # SELECT BEST CHUNKS
    # ------------------------------------------

    max_chunks = min(
        3,
        len(documents)
    )

    selected_documents = documents[
        :max_chunks
    ]

    relevant_sentences = (
        extract_relevant_sentences(
            question,
            selected_documents,
            max_sentences=6
        )
    )
    
    selected_metadatas = metadatas[
        :max_chunks
    ]

    # ------------------------------------------
    # LANGUAGE LABEL
    # ------------------------------------------

    if response_language == "Hindi":

        title = "📄 PDF से संबंधित जानकारी"

    elif response_language == "Hinglish":

        title = "📄 PDF se related information"

    else:

        title = "📄 Relevant PDF Information"

    # ------------------------------------------
    # BUILD FALLBACK RESPONSE
    # ------------------------------------------

    response_parts = []

    response_parts.append(
        "⚠️ AI service is temporarily unavailable."
    )

    response_parts.append(
        title
    )

    response_parts.append(
        "The following relevant information "
        "was found in your PDF:"
    )

    # ------------------------------------------
    # ADD RELEVANT CONTENT
    # ------------------------------------------

    if relevant_sentences:
        for i, sentence in enumerate(
            relevant_sentences
        ):

            response_parts.append(
                f"• {sentence}"
            )

    else:

        for i, document in enumerate(
            selected_documents
        ):

            document = document.strip()

            if not document:
                continue

            response_parts.append(
                f"**{i + 1}.** {document}"
            )

    # ------------------------------------------
    # SOURCES
    # ------------------------------------------

    response_parts.append(
        "📚 **Sources**"
    )

    shown_sources = set()

    for metadata in selected_metadatas:

        filename = metadata.get(
            "filename",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        section = metadata.get(
            "section",
            "General"
        )

        source = (
            filename,
            page,
            section
        )

        if source in shown_sources:
            continue

        shown_sources.add(
            source
        )

        response_parts.append(
            f"- **{filename}** | "
            f"Page: {page} | "
            f"Section: {section}"
        )

    return "\n\n".join(
        response_parts
    )

# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(
    question,
    context,
    chat_history,
    response_language="English"
):

    track_ai_usage(
        "request",
        "Chat"
    )
    history_text = ""

    for message in chat_history[-4:]:

        history_text += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )
    
    optimization = (
        get_ai_optimization_settings()
    )

    context_limit = min(
        get_context_budget(question),
        optimization["context_limit"]
    )

    context = compress_context(
        context,
        max_chars=context_limit
    )
    
    cache_key = get_cache_key(
        question,
        context,
        response_language,
        "chat"
    )

    if cache_key in st.session_state.response_cache:
        track_ai_usage(
            "cache_hit",
            "Chat"
        )

        return st.session_state.response_cache[
            cache_key
        ]
    
    if is_request_processing(
        cache_key
    ):

        return (
            "⏳ This request is already being processed. "
            "Please wait for the current response."
        )

    st.session_state.processing_requests.add(
        cache_key
    )

    prompt = f"""
You are an AI PDF Assistant.

Your job is to answer the user's current question
using ONLY the information provided in the PDF context.

LANGUAGE RULE:
Selected response language: {response_language}

Rules:
- English: Answer in clear English.
- Hindi: Answer in Hindi using Devanagari script.
- Hinglish: Answer in natural Hindi-English using Roman/English script.
- Do NOT mention the selected language.
- Do NOT say that you understood the language.
- Do NOT greet the user.
- Do NOT say "How can I help you?"
- Directly answer the user's question.
- Do not make up information that is not supported by the PDF.

CONVERSATION HISTORY:
-------------------------
{history_text}
-------------------------

PDF CONTEXT:
-------------------------
{context}
-------------------------

CURRENT QUESTION:
-------------------------
{question}
-------------------------

Now answer the CURRENT QUESTION directly.
"""

    try:

        response = safe_generate_content(
            prompt
        )

        response = response.strip()

        save_to_response_cache(
            cache_key,
            response
        )

        return response

    except Exception:

        track_ai_usage(
            "failed",
            "Chat"
        )

        raise

    finally:

        st.session_state.processing_requests.discard(
            cache_key
        )



# ======================================
#  GENERATE NOTES
# ======================================
def generate_notes(
    context,
    response_language="English"
):

    prompt = f"""
You are an AI study notes generator.

Create structured study notes using ONLY
the provided PDF context.

Response language:
{response_language}

Rules:
- English: Use clear English.
- Hindi: Use Hindi in Devanagari script.
- Hinglish: Use natural Hindi-English in Roman script.
- Do not mention the language.
- Do not add information outside the PDF.

Use this structure:

# Study Notes

## Important Definitions
- Definition points

## Main Concepts
- Important concepts

## Key Points
- Important points

## Examples
- Examples from the PDF

## Important Terms
- Important terms

PDF CONTEXT:
-------------------------
{context}
-------------------------

Generate the study notes now.
"""

    response = safe_generate_content(
        prompt
    )

    return response.strip()


# ==============================
# GENERATE CONCEPT MAP
# =============================
def generate_concept_map(
    context,
    learning_purpose="General Learning"
):

    prompt = f"""
You are an expert knowledge-map designer.

Analyze ONLY the provided PDF content.

Create a TREE-STYLE concept map.

Learning purpose:
{learning_purpose}

Return ONLY valid JSON.

FORMAT:

{{
  "concepts": [
    {{
      "name": "Main Topic",
      "parent": null
    }},
    {{
      "name": "Major Concept 1",
      "parent": "Main Topic"
    }},
    {{
      "name": "Major Concept 2",
      "parent": "Main Topic"
    }},
    {{
      "name": "Sub Concept",
      "parent": "Major Concept 1"
    }}
  ]
}}

STRICT RULES:

1. Create exactly ONE root/main concept.
2. The root must have parent = null.
3. Create 3 to 6 major concepts directly under
   the root.
4. Create sub-concepts under major concepts
   where appropriate.
5. Do NOT create a long chain.
6. Do NOT make one concept the parent of every
   other concept.
7. Maximum hierarchy depth is 3 levels:

   Level 1 = Main Topic
   Level 2 = Major Concepts
   Level 3 = Sub Concepts

8. A Level 3 concept must NOT have children.
9. Every concept must have exactly one parent,
   except the root.
10. Use concepts actually present in the PDF.
11. Do not invent concepts.
12. Do not add explanations.
13. Do not use Markdown.
14. Return JSON only.

BAD STRUCTURE:

A → B → C → D → E → F → G

GOOD STRUCTURE:

             A
       ┌─────┼─────┐
       ↓     ↓     ↓
       B     C     D
      / \\    │    / \\
     E   F   G    H   I

PDF CONTENT:
-------------------------
{context}
-------------------------

Generate the tree concept map now.
"""

    response = safe_generate_content(
        prompt
    )

    return response.strip()



# =================================
#  GENERATE PDF COMPARISION
# =================================
def generate_pdf_comparison(
    pdf1,
    context1,
    pdf2,
    context2
):

    prompt = f"""
You are an AI document comparison assistant.

Compare these two PDF documents using ONLY
their provided content.

PDF 1:
{pdf1}

PDF 2:
{pdf2}

Return a clear Markdown comparison with
these sections:

## 📚 Common Topics
Topics covered by both PDFs.

## 📄 Only in PDF 1
Important topics found only in PDF 1.

## 📄 Only in PDF 2
Important topics found only in PDF 2.

## 🔍 Key Differences
Important differences in the way the
topics are covered or explained.

## 📊 Coverage
Explain which PDF gives more detail
where the content supports this conclusion.

## ⚠️ Missing Topics
Mention important topics present in one PDF
but absent from the other.

Rules:

- Use ONLY the provided PDF content.
- Do not use outside knowledge.
- Do not invent information.
- If a section cannot be determined,
  clearly say so.
- Keep the comparison factual.
- Do not mention these instructions.

PDF 1 CONTENT:
-------------------------
{context1}
-------------------------

PDF 2 CONTENT:
-------------------------
{context2}
-------------------------

Generate the comparison.
"""

    response = safe_generate_content(
        prompt
    )

    return response.strip()


# ================================
#  GENERATE IMPORTANT QUESTIONS 
# ===============================


def generate_important_questions(
    context,
    response_language="English"
):

    prompt = f"""
You are an AI exam preparation assistant.

Generate 10 important exam questions from
ONLY the provided PDF context.

Response language:
{response_language}

Rules:
- English: Use clear English.
- Hindi: Use Hindi in Devanagari script.
- Hinglish: Use natural Hindi-English in Roman script.
- Do not mention the language.
- Do not add information outside the PDF.
- Include a mixture of short-answer and
  descriptive questions.
- Focus on concepts that are important for exams.
- Do not provide answers.
- Number every question from 1 to 10.

PDF CONTEXT:
-------------------------
{context}
-------------------------

Generate the important questions now.
"""

    response = safe_generate_content(
        prompt
    )

    return response.strip()

# ===============================
# GENERATE FLASHCARD
# ===============================
def generate_flashcards(
    context,
    response_language="English"
):

    prompt = f"""
You are an AI flashcard generator.

Create 8 useful study flashcards from ONLY
the provided PDF context.

Response language:
{response_language}

Rules:
- English: Use clear English.
- Hindi: Use Hindi in Devanagari script.
- Hinglish: Use natural Hindi-English in Roman script.
- Do not mention the language.
- Do not add information outside the PDF.
- Keep questions short and useful for revision.
- Keep answers concise.

Return ONLY valid JSON in this format:

[
  {{
    "question": "Question here",
    "answer": "Answer here"
  }}
]

PDF CONTEXT:
-------------------------
{context}
-------------------------
"""

    response = safe_generate_content(
        prompt
    )

    return response.strip()


# =======================================
# TRANSLATE PDF TEXT
# =======================================

def translate_pdf_text(
    text,
    target_language="Hindi"
):
    if not text or not text.strip():
        return ""

    prompt = f"""
You are a professional PDF translator.

Translate the following PDF content into {target_language}.

RULES:
1. Translate ALL content.
2. Do NOT summarize.
3. Do NOT add information.
4. Do NOT remove information.
5. Preserve the original order.
6. Preserve headings and paragraphs.
7. Preserve line breaks as much as possible.
8. Preserve bullet points and numbered lists.
9. Keep commands, code, numbers, dates and IDs unchanged.
10. Return ONLY the translated content.
11. Do not add explanations.

PDF CONTENT:
-------------------------
{text}
-------------------------

Return the complete Hindi translation now.
"""

    response = safe_generate_content(prompt)

    if not response:
        raise RuntimeError(
            "Gemini returned an empty translation."
        )

    return response.strip()


# ==================================================
# TRANSLATION CHECKPOINT
# ==================================================

def get_translation_checkpoint(
    filename,
    language
):

    safe_name = (
        f"{os.path.splitext(filename)[0]}"
        f"_{language}_translation.json"
    )

    return safe_name


def load_translation_checkpoint(
    filename,
    language
):

    checkpoint_file = (
        get_translation_checkpoint(
            filename,
            language
        )
    )

    if not os.path.exists(
        checkpoint_file
    ):

        return []

    try:

        with open(
            checkpoint_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(
            data,
            list
        ):

            return data

        return []

    except Exception:

        return []

# ===================================
#  SAVE TRANSLATE CHECKPOINT
# ===================================
def save_translation_checkpoint(
    filename,
    language,
    translated_parts
):

    checkpoint_file = (
        get_translation_checkpoint(
            filename,
            language
        )
    )

    try:

        with open(
            checkpoint_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                translated_parts,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception:

        pass
    
    
    
    
# ==========================================
# PAGE TRANSLATION CHECKPOINT
# ==========================================

def get_page_translation_checkpoint(
    filename,
    language
):

    return (
        f"{os.path.splitext(filename)[0]}"
        f"_{language}_page_translation.json"
    )


def load_page_translation_checkpoint(
    filename,
    language
):

    checkpoint_file = (
        get_page_translation_checkpoint(
            filename,
            language
        )
    )

    # --------------------------------------
    # CURRENT TRANSLATION VERSION
    # --------------------------------------

    CURRENT_TRANSLATION_VERSION = "hindi_v3_quality_guard"

    # --------------------------------------
    # NO CHECKPOINT
    # --------------------------------------

    if not os.path.exists(
        checkpoint_file
    ):
        return {
            "version": CURRENT_TRANSLATION_VERSION,
            "pages": {},
            "failures": {}
        }

    try:

        with open(
            checkpoint_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        # --------------------------------------
        # VALID NEW FORMAT
        # --------------------------------------

        if (
            isinstance(data, dict)
            and data.get("version")
            == CURRENT_TRANSLATION_VERSION
            and "pages" in data
            and "failures" in data
        ):

            return {
                "version":
                    CURRENT_TRANSLATION_VERSION,

                "pages":
                    data.get(
                        "pages",
                        {}
                    ),

                "failures":
                    data.get(
                        "failures",
                        {}
                    )
            }

        # --------------------------------------
        # OLD / OUTDATED CHECKPOINT
        # --------------------------------------
        #
        # Prompt or translation architecture
        # has changed.
        #
        # Do NOT reuse old translations.
        # --------------------------------------

        return {
            "version":
                CURRENT_TRANSLATION_VERSION,

            "pages": {},

            "failures": {}
        }

    except Exception:

        # --------------------------------------
        # CORRUPTED CHECKPOINT
        # --------------------------------------

        return {
            "version":
                CURRENT_TRANSLATION_VERSION,

            "pages": {},

            "failures": {}
        }


def save_page_translation_checkpoint(
    filename,
    language,
    translated_pages,
    failures=None
):

    checkpoint_file = (
        get_page_translation_checkpoint(
            filename,
            language
        )
    )

    if failures is None:
        failures = {}

    # --------------------------------------
    # CURRENT TRANSLATION VERSION
    # --------------------------------------

    CURRENT_TRANSLATION_VERSION = (
        "hindi_v2_quality_guard"
    )

    # --------------------------------------
    # CHECKPOINT DATA
    # --------------------------------------

    data = {
        "version": CURRENT_TRANSLATION_VERSION,
        "pages": translated_pages,
        "failures": failures
    }

    try:

        # ----------------------------------
        # Ensure checkpoint directory exists
        # ----------------------------------

        checkpoint_dir = os.path.dirname(
            checkpoint_file
        )

        if checkpoint_dir:
            os.makedirs(
                checkpoint_dir,
                exist_ok=True
            )

        # ----------------------------------
        # Save checkpoint
        # ----------------------------------

        with open(
            checkpoint_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:

        # ----------------------------------
        # Checkpoint failure should NOT
        # crash the PDF translation.
        # ----------------------------------

        print(
            f"Warning: Could not save "
            f"translation checkpoint: {e}"
        )
    
    
# ==========================================
# TRANSLATION FAILURE TRACKER
# ==========================================

def save_translation_failure(
    filename,
    language,
    page_number,
    error_message
):

    checkpoint_file = (
        get_page_translation_checkpoint(
            filename,
            language
        )
    )

    data = load_page_translation_checkpoint(
        filename,
        language
    )

    failures = data.get(
        "_failures",
        {}
    )

    failures[str(page_number)] = {
        "error": str(error_message)
    }

    data["_failures"] = failures

    save_page_translation_checkpoint(
        filename,
        language,
        data
    )
    
    
# ==========================================
# PAGE CONTENT FINGERPRINT
# ==========================================

def get_page_fingerprint(page_data):

    if isinstance(page_data, dict):

        payload = {
            "text": str(
                page_data.get(
                    "text",
                    ""
                )
            ).strip(),

            "tables": page_data.get(
                "tables",
                []
            )
        }

    else:

        payload = {
            "text": str(
                page_data or ""
            ).strip(),

            "tables": []
        }

    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()
    
# ==================================================
# TRANSLATION CACHE
# ==================================================

def get_translation_cache_file(
    language
):

    return (
        f"translation_cache_{language}.json"
    )


def load_translation_cache(
    language
):

    cache_file = (
        get_translation_cache_file(
            language
        )
    )

    if not os.path.exists(
        cache_file
    ):

        return {}

    try:

        with open(
            cache_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(
            data,
            dict
        ):

            return data

        return {}

    except Exception:

        return {}


def save_translation_cache(
    language,
    cache
):

    cache_file = (
        get_translation_cache_file(
            language
        )
    )

    try:

        with open(
            cache_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                cache,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception:

        pass
    

# ==================================================
# TABLE DETECTION
# ==================================================

def is_table_line(line):

    line = line.strip()

    if not line:
        return False

    # Markdown table
    if "|" in line:

        parts = [
            part.strip()
            for part in line.split("|")
            if part.strip()
        ]

        if len(parts) >= 2:
            return True

    # Tab-separated data
    if "\t" in line:

        parts = [
            part.strip()
            for part in line.split("\t")
            if part.strip()
        ]

        if len(parts) >= 2:
            return True

    return False


# ==================================================
# EXTRACT TABLE BLOCKS
# ==================================================

def extract_table_blocks(text):

    lines = text.splitlines()

    blocks = []
    current_table = []

    for line in lines:

        if is_table_line(line):

            current_table.append(
                line
            )

        else:

            if current_table:

                blocks.append(
                    current_table
                )

                current_table = []

    if current_table:

        blocks.append(
            current_table
        )

    return blocks


# ====================================
#  PARSE TABLE LINES
# ====================================
def parse_table_lines(
    table_lines
):

    rows = []

    for line in table_lines:

        line = line.strip()

        # Markdown table
        if "|" in line:

            cells = [
                cell.strip()
                for cell in line.split("|")
            ]

            cells = [
                cell
                for cell in cells
                if cell
            ]

            # Skip markdown separator
            if cells and all(
                set(cell) <= set("-: ")
                for cell in cells
            ):
                continue

            if len(cells) >= 2:

                rows.append(
                    cells
                )

        # Tab-separated table
        elif "\t" in line:

            cells = [
                cell.strip()
                for cell in line.split("\t")
            ]

            if len(cells) >= 2:

                rows.append(
                    cells
                )

    return rows



# ==========================================
# PARSE BATCH TRANSLATION
# ==========================================

# def parse_translation_batch_response(
#     response,
#     batch
# ):

#     if not response:
#         raise RuntimeError(
#             "Gemini returned an empty translation response."
#         )

#     translated_pages = {}
#     missing_pages = []

#     for item in batch:

#         page_number = str(
#             item["page"]
#         )

#         start_marker = (
#             f"<<<PAGE {page_number}>>>"
#         )

#         end_marker = (
#             f"<<<END PAGE {page_number}>>>"
#         )

#         start_index = response.find(
#             start_marker
#         )

#         if start_index == -1:

#             missing_pages.append(
#                 page_number
#             )

#             continue

#         start_index += len(
#             start_marker
#         )

#         end_index = response.find(
#             end_marker,
#             start_index
#         )

#         if end_index == -1:

#             missing_pages.append(
#                 page_number
#             )

#             continue

#         translated_text = (
#             response[
#                 start_index:end_index
#             ]
#             .strip()
#         )

#         if not translated_text:

#             missing_pages.append(
#                 page_number
#             )

#             continue

#         translated_pages[
#             page_number
#         ] = translated_text

#     # ==========================================
#     # STRICT PAGE VALIDATION
#     # ==========================================

#     if missing_pages:

#         raise RuntimeError(
#             "Gemini did not return a complete "
#             "translation for page(s): "
#             + ", ".join(missing_pages)
#         )

#     # ==========================================
#     # EXTRA SAFETY CHECK
#     # ==========================================

#     expected_pages = {
#         str(item["page"])
#         for item in batch
#     }

#     returned_pages = set(
#         translated_pages.keys()
#     )

#     if returned_pages != expected_pages:

#         missing = (
#             expected_pages
#             - returned_pages
#         )

#         raise RuntimeError(
#             "Translation response is incomplete. "
#             "Missing page(s): "
#             + ", ".join(
#                 sorted(missing)
#             )
#         )

#     return translated_pages


# ==========================================
# TRANSLATE PDF BATCH
# ==========================================

def translate_pdf_batch(
    batch,
    target_language="Hindi"
):
    """
    Production-level PDF batch translation.

    One Gemini API call translates:
        - page text
        - all tables

    Response format:
    {
        "pages": [
            {
                "page": "1",
                "translation": "...",
                "translated_tables": [
                    [
                        ["cell 1", "cell 2"],
                        ["cell 3", "cell 4"]
                    ]
                ]
            }
        ]
    }
    """

    if not batch:
        return {}

    # --------------------------------------------------
    # 1. Prepare structured source data
    # --------------------------------------------------

    source_pages = []

    for item in batch:

        page_number = str(item["page"])

        source_pages.append({
            "page": page_number,
            "text": str(
                item.get("text", "")
            ).strip(),
            "tables": item.get(
                "tables",
                []
            )
        })

    source_json = json.dumps(
        source_pages,
        ensure_ascii=False,
        indent=2
    )

    # --------------------------------------------------
    # 2. Production translation prompt
    # --------------------------------------------------

    prompt = f"""
You are a professional technical PDF translator.

Translate the provided English Linux documentation into
natural, clear and professional {target_language}.

The input contains multiple PDF pages.

Your job is to translate BOTH:

1. Normal page text
2. Table cell text

Do NOT summarize the content.

Do NOT remove information.

Do NOT add new information.

Do NOT change the structure.

==================================================
TRANSLATION QUALITY
==================================================

1. Translate all explanatory English sentences.

2. Preserve the original meaning exactly.

3. Use natural and grammatically correct Hindi.

4. Do not perform awkward word-for-word translation.

5. Use simple, professional Hindi that a
   Hindi-speaking computer science student can
   easily understand.

6. Keep technical English terminology when
   translating it into Hindi would make the
   meaning less clear.

Examples of terms that should normally remain
in English:

Linux
Linux command
command
terminal
shell
Bash
filesystem
directory
file
process
job
PID
CPU
RAM
memory
storage
disk
partition
package
service
system
kernel
environment variable
variable
script
function
argument
parameter
output
input
permission
user
group
root
server
client
network
socket
port
URL
API
DNS
SSH
HTTP
HTTPS
TLS
certificate
encryption
compiler
debugging
system call
signal
pipeline
redirection
cron
OpenSSL
Git
Python
Node.js
JavaScript
Makefile
ACL

==================================================
COMMAND / CODE PRESERVATION
==================================================

NEVER translate or modify:

- Linux command names
- command options
- shell commands
- code
- file paths
- URLs
- variable names
- programming syntax
- Linux operators
- numbers
- technical identifiers

These examples MUST remain EXACTLY unchanged:

pwd

ls -la

cd /var/log

dmesg | tail

ps aux | grep nginx

echo hello > file.txt

command > all.log 2>&1

mkdir app && cd app

IMPORTANT:

The character "|" may appear inside a table cell.

It is NOT a table-column separator.

Treat "|" as normal cell content unless the
JSON structure itself indicates a different column.

==================================================
HEADINGS / STRUCTURE
==================================================

Preserve:

- page numbers
- headings
- heading numbering
- paragraphs
- bullet points
- examples
- content order
- table order
- table row order
- table column order

Do not combine unrelated sections.

Do not move content between pages.

Do not split one page into another page.

==================================================
HINDI QUALITY
==================================================

Use natural technical Hindi.

BAD:

"printf आउटपुट को अनुमानित रूप से फ़ॉर्मेट करता है।"

GOOD:

"printf आउटपुट को सटीक रूप से फ़ॉर्मेट करता है।"

BAD:

"कमांड उच्च विशेषाधिकारों के साथ चलता है।"

GOOD:

"कमांड को उच्च विशेषाधिकारों के साथ चलाता है।"

Avoid literal machine-style translations.

==================================================
TABLE RULES
==================================================

For every table:

1. Preserve the exact number of rows.

2. Preserve the exact number of columns.

3. Translate only translatable text inside cells.

4. Keep commands unchanged.

5. Keep code unchanged.

6. Keep paths unchanged.

7. Keep URLs unchanged.

8. Keep numbers unchanged.

9. Keep Linux operators unchanged.

10. Never create additional columns.

11. Never remove columns.

12. Never merge rows.

13. Never split rows.

14. Never reorder rows or columns.

15. "|" inside a cell MUST remain inside that cell.



# --------------------------------------------------
# HINDI TRANSLATION QUALITY GUARD
# --------------------------------------------------

HINDI QUALITY RULES:

1. Translate the meaning naturally, not word-by-word.

2. The final text must read like professional technical
   documentation written in natural Hindi.

3. Use Hindi for normal explanatory sentences.

4. Keep technical terms in English when they are commonly
   used in Linux/programming documentation.

   Examples:
   Linux, command, terminal, file, directory, process,
   filesystem, package, service, kernel, script, shell,
   variable, function, argument, output, input, memory,
   CPU, RAM, API, URL, SSH, DNS, HTTP, HTTPS, Git,
   Python, Node.js, compiler, permissions, ACL, etc.

5. Do NOT forcefully translate technical English terms
   into unnatural Hindi.

6. Do NOT create awkward Hindi-English combinations.

   BAD:
   "interactively रूप से processes को प्रबंधित करें।"

   GOOD:
   "processes को interactive तरीके से प्रबंधित करें।"

7. Every explanatory sentence must be grammatically complete.

   BAD:
   "कई results पर एक command लागू करने के लिए।"

   GOOD:
   "कई results पर एक ही command लागू करने के लिए इसका उपयोग करें।"

   BAD:
   "GNU C compiler। Linux पर C programs को compile करने के लिए।"

   GOOD:
   "GNU C compiler है। Linux पर C programs को compile करने के लिए
   इसका उपयोग किया जाता है।"

8. Avoid unnecessary literal translations.

   BAD:
   "पूर्व स्थिति में लाएँ"

   Prefer natural wording such as:
   "पहली स्थिति में वापस लाएँ"
   or another context-appropriate Hindi expression.

9. Avoid unnatural words such as:

   "अनुमानित रूप से"
   when the intended meaning is "accurately" or "precisely".

   Prefer:
   "सटीक रूप से"

10. Prefer natural technical Hindi sentence patterns.

   BAD:
   "Processes को interactive तरीके से प्रबंधित और निरीक्षण करें।"

   GOOD:
   "Processes को interactive तरीके से प्रबंधित करें और उनकी निगरानी करें।"

11. Do not translate commands, command options, flags,
    operators, code, file paths, URLs, usernames, variables,
    package names, programming syntax, or examples.

12. Preserve these exactly whenever they appear:

    |
    >
    >>
    <
    2>
    2>&1
    &&
    ||
    ;
    $HOME
    $1
    -r
    -f
    -n
    -p
    -c

13. A "|" character inside a table cell is content.
    Never treat it as a table separator.

14. Preserve numbers, version numbers, filenames,
    extensions, commands and code exactly.

15. Do not add information that is not present in the
    original document.

16. Do not remove information from the original document.

17. For short technical table cells, concise Hindi is preferred.

18. For "Why use it" descriptions, use a complete,
    natural purpose sentence.

    Prefer:
    "Files की शुरुआती lines देखने के लिए।"

    If the sentence is grammatically dependent on the
    previous text, rewrite it naturally rather than leaving
    an incomplete phrase.

19. Avoid excessive English when a natural Hindi expression
    is available.

20. However, technical English is allowed and should NOT
    be replaced merely to increase the amount of Hindi.

21. IMPORTANT:
    Before returning the JSON, internally review every
    translated sentence and every translated table cell.

    Check:

    - Is the Hindi natural?
    - Is the sentence complete?
    - Is the meaning preserved?
    - Are technical terms used appropriately?
    - Were commands/code preserved exactly?
    - Were table rows and columns preserved?
    - Is any English present unnecessarily?
    - Does any sentence sound like machine translation?

22. If a sentence sounds like direct machine translation,
    rewrite it naturally before returning the final JSON.

23. Final output should be suitable for a professional
    Hindi Linux command reference document.
    
    
    
# --------------------------------------------------
# FINAL INTERNAL REVIEW
# --------------------------------------------------

Before producing the final JSON, perform one silent
internal proofreading pass.

Do NOT describe the proofreading.
Do NOT return comments.
Do NOT return explanations.

Return only the required JSON object.

Every page and every table cell must be checked for:

- natural Hindi
- complete sentences
- correct technical terminology
- preserved commands and syntax
- preserved table structure
- preserved meaning
- no unnecessary literal translation

==================================================
CRITICAL OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Do NOT return:

- Markdown
- ```json
- explanations
- comments
- notes
- additional text

Use EXACTLY this structure:

{{
  "pages": [
    {{
      "page": "1",
      "translation": "translated page text",
      "translated_tables": [
        [
          ["translated cell", "translated cell"],
          ["translated cell", "translated cell"]
        ]
      ]
    }}
  ]
}}

IMPORTANT:

- Every input page MUST appear exactly once.
- Page numbers MUST remain unchanged.
- The order of pages MUST remain unchanged.
- Every input table MUST have exactly one output table.
- Every input row MUST have exactly one output row.
- Every input column MUST have exactly one output column.
- Do not omit empty cells.
- Do not create additional cells.
- Do not change the JSON structure.

==================================================
SOURCE PDF DATA
==================================================

{source_json}
"""

    # --------------------------------------------------
    # 3. ONE Gemini API call
    # --------------------------------------------------

    response = safe_generate_content(
        prompt,
        feature="PDF Batch Translation"
    )

    if not response:
        raise RuntimeError(
            "Gemini returned an empty translation response."
        )

    # --------------------------------------------------
    # 4. Clean accidental markdown fences
    # --------------------------------------------------

    response = response.strip()

    if response.startswith("```"):
        response = response.replace(
            "```json",
            "",
            1
        )

        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()

    # --------------------------------------------------
    # 5. Parse JSON
    # --------------------------------------------------

    try:
        parsed = json.loads(response)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            "Gemini returned invalid JSON for PDF "
            f"translation: {e}"
        )

    # --------------------------------------------------
    # 6. Validate top-level structure
    # --------------------------------------------------

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Translation response must be a JSON object."
        )

    returned_pages = parsed.get("pages")

    if not isinstance(returned_pages, list):
        raise RuntimeError(
            "Translation response does not contain "
            "a valid 'pages' list."
        )

    # --------------------------------------------------
    # 7. Validate page count
    # --------------------------------------------------

    if len(returned_pages) != len(batch):
        raise RuntimeError(
            "Page count changed in translation response. "
            f"Expected {len(batch)}, "
            f"got {len(returned_pages)}."
        )

    # --------------------------------------------------
    # 8. Build result after validation
    # --------------------------------------------------

    results = {}

    expected_page_numbers = [
        str(item["page"])
        for item in batch
    ]

    returned_page_numbers = []

    for page_result in returned_pages:

        if not isinstance(page_result, dict):
            raise RuntimeError(
                "Invalid page object in translation response."
            )

        page_number = str(
            page_result.get("page", "")
        ).strip()

        returned_page_numbers.append(
            page_number
        )

    # --------------------------------------------------
    # 9. Validate page numbers
    # --------------------------------------------------

    if returned_page_numbers != expected_page_numbers:
        raise RuntimeError(
            "Page order or page numbers changed. "
            f"Expected {expected_page_numbers}, "
            f"got {returned_page_numbers}."
        )

    # --------------------------------------------------
    # 10. Validate every page
    # --------------------------------------------------

    for item, page_result in zip(
        batch,
        returned_pages
    ):

        page_number = str(
            item["page"]
        )

        translated_text = str(
            page_result.get(
                "translation",
                ""
            )
        ).strip()
        
        
        # --------------------------------------
        # HINDI QUALITY VALIDATION + AUTO REPAIR
        # --------------------------------------

        original_text = str(
            item.get(
                "text",
                ""
            )
        ).strip()

        if original_text and translated_text:

            translated_text = (
                quality_check_and_repair_page(
                    original_text,
                    translated_text,
                    max_repairs=1
                )
            )

        translated_tables = page_result.get(
            "translated_tables",
            []
        )

        original_tables = item.get(
            "tables",
            []
        )
        
        # --------------------------------------
        # TABLE HINDI QUALITY VALIDATION
        # + AUTO REPAIR
        # --------------------------------------

        if original_tables and translated_tables:

            if len(original_tables) != len(
                translated_tables
            ):
                raise RuntimeError(
                    f"Page {page_number}: "
                    "table count mismatch before "
                    "quality validation."
                )

            repaired_tables = []

            for (
                original_table,
                translated_table
            ) in zip(
                original_tables,
                translated_tables
            ):

                repaired_table = (
                    quality_check_and_repair_table(
                        original_table,
                        translated_table,
                        max_repairs=1
                    )
                )

                repaired_tables.append(
                    repaired_table
                )

            translated_tables = repaired_tables

        # ----------------------------------------------
        # Text / content validation
        # ----------------------------------------------

        if (
            not translated_text
            and not original_tables
        ):
            raise RuntimeError(
                f"Empty translation for page "
                f"{page_number}."
            )

        # ----------------------------------------------
        # Table list validation
        # ----------------------------------------------

        if not isinstance(
            translated_tables,
            list
        ):
            raise RuntimeError(
                f"Invalid table structure on "
                f"page {page_number}."
            )

        if len(translated_tables) != len(
            original_tables
        ):
            raise RuntimeError(
                f"Table count changed on page "
                f"{page_number}. "
                f"Expected {len(original_tables)}, "
                f"got {len(translated_tables)}."
            )

        # ----------------------------------------------
        # Validate every table
        # ----------------------------------------------

        validated_tables = []

        for table_index, (
            original_table,
            translated_table
        ) in enumerate(
            zip(
                original_tables,
                translated_tables
            ),
            start=1
        ):

            if not isinstance(
                translated_table,
                list
            ):
                raise RuntimeError(
                    f"Invalid table structure on "
                    f"page {page_number}, "
                    f"table {table_index}."
                )

            # ------------------------------------------
            # Row count
            # ------------------------------------------

            if len(translated_table) != len(
                original_table
            ):
                raise RuntimeError(
                    f"Table row count changed on "
                    f"page {page_number}, "
                    f"table {table_index}. "
                    f"Expected {len(original_table)}, "
                    f"got {len(translated_table)}."
                )

            validated_table = []

            # ------------------------------------------
            # Validate every row
            # ------------------------------------------

            for row_index, (
                original_row,
                translated_row
            ) in enumerate(
                zip(
                    original_table,
                    translated_table
                ),
                start=1
            ):

                if not isinstance(
                    translated_row,
                    list
                ):
                    raise RuntimeError(
                        f"Invalid row structure on "
                        f"page {page_number}, "
                        f"table {table_index}, "
                        f"row {row_index}."
                    )

                # --------------------------------------
                # Column count
                # --------------------------------------

                if len(translated_row) != len(
                    original_row
                ):
                    raise RuntimeError(
                        f"Table column count changed "
                        f"on page {page_number}, "
                        f"table {table_index}, "
                        f"row {row_index}. "
                        f"Expected {len(original_row)}, "
                        f"got {len(translated_row)}."
                    )

                cleaned_row = []

                for cell in translated_row:

                    if cell is None:
                        cell = ""

                    if not isinstance(
                        cell,
                        str
                    ):
                        cell = str(cell)

                    cleaned_row.append(
                        cell.strip()
                    )

                validated_table.append(
                    cleaned_row
                )

            validated_tables.append(
                validated_table
            )

        # ----------------------------------------------
        # Store fully validated page
        # ----------------------------------------------

        results[page_number] = {
            "translation": translated_text,
            "translated_tables": validated_tables
        }

    # --------------------------------------------------
    # 11. Final page-set validation
    # --------------------------------------------------

    if set(results.keys()) != set(
        expected_page_numbers
    ):
        missing_pages = (
            set(expected_page_numbers)
            - set(results.keys())
        )

        raise RuntimeError(
            "Translation response is incomplete. "
            "Missing page(s): "
            + ", ".join(
                sorted(missing_pages)
            )
        )

    return results


# # ============================
# # DEBUG
# st.write(
#     "DEBUG translated pages:",
#     len(translated_pages)
# )

# for page_number, data in translated_pages.items():
#     st.write(
#         f"Page {page_number}:",
#         str(data.get("translation", ""))[:150]
#     )

# =================================
# CREATE HINDI PDF
# =================================

def create_hindi_pdf(layout_pages):

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25,
        leftMargin=25,
        topMargin=25,
        bottomMargin=25
    )

    styles = getSampleStyleSheet()

    normal_style = ParagraphStyle(
        "HindiNormal",
        parent=styles["Normal"],
        fontName="NotoHindi",
        fontSize=8.5,
        leading=10,
        spaceAfter=2
    )

    heading_style = ParagraphStyle(
        "HindiHeading",
        parent=styles["Heading2"],
        fontName="NotoHindi",
        fontSize=11,
        leading=13,
        spaceAfter=4
    )

    table_style = ParagraphStyle(
        "HindiTable",
        parent=normal_style,
        fontName="NotoHindi",
        fontSize=6.2,
        leading=7.5,
        spaceAfter=0
    )

    story = []

    # ==================================================
    # HELPER: ESCAPE PDF TEXT
    # ==================================================

    def safe_pdf_text(value):
        return (
            str(value or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

    # ==================================================
    # HELPER: NORMALIZE TABLE
    # ==================================================

    def normalize_table(table_data):

        if not table_data:
            return []

        cleaned_rows = []

        # ----------------------------------------------
        # Remove completely empty rows
        # ----------------------------------------------

        for row in table_data:

            if not isinstance(row, list):
                continue

            cleaned_row = [
                str(cell or "").strip()
                for cell in row
            ]

            if not any(cleaned_row):
                continue

            cleaned_rows.append(
                cleaned_row
            )

        if not cleaned_rows:
            return []

        # ----------------------------------------------
        # Determine dominant column count
        # ----------------------------------------------

        row_lengths = [
            len(row)
            for row in cleaned_rows
        ]

        column_count = max(
            set(row_lengths),
            key=row_lengths.count
        )

        if column_count <= 0:
            return []

        # ----------------------------------------------
        # Normalize every row
        # ----------------------------------------------

        normalized_rows = []

        for row in cleaned_rows:

            if len(row) < column_count:

                row = row + (
                    [""] *
                    (
                        column_count
                        - len(row)
                    )
                )

            elif len(row) > column_count:

                # Unexpected extra cells are merged
                # into the final valid column.

                extra_cells = row[
                    column_count - 1:
                ]

                merged_last_cell = " ".join(
                    cell
                    for cell in extra_cells
                    if cell
                )

                row = (
                    row[:column_count - 1]
                    + [merged_last_cell]
                )

            normalized_rows.append(
                row
            )

        return normalized_rows

    # ==================================================
    # ONE ORIGINAL PAGE = ONE PDF PAGE
    # ==================================================

    for page_index, page_data in enumerate(
        layout_pages
    ):

        page_number = page_data.get(
            "page",
            page_index + 1
        )

        page_story = []

        # ==============================================
        # PAGE TITLE
        # ==============================================

        page_story.append(
            Paragraph(
                safe_pdf_text(
                    f"Page {page_number}"
                ),
                heading_style
            )
        )

        # ==============================================
        # NORMAL TEXT
        # ==============================================

        page_text = str(
            page_data.get(
                "translated_text",
                ""
            )
        ).strip()

        if page_text:

            paragraphs = page_text.split(
                "\n"
            )

            for paragraph in paragraphs:

                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                page_story.append(
                    Paragraph(
                        safe_pdf_text(
                            paragraph
                        ),
                        normal_style
                    )
                )

        # ==============================================
        # TABLES
        # ==============================================

        translated_tables = page_data.get(
            "translated_tables",
            []
        )

        for table_data in translated_tables:

            normalized_rows = normalize_table(
                table_data
            )

            if not normalized_rows:
                continue

            # ------------------------------------------
            # FORMAT CELLS
            # ------------------------------------------

            formatted_table = []

            for row in normalized_rows:

                formatted_row = []

                for cell in row:

                    safe_cell = safe_pdf_text(
                        cell
                    )

                    formatted_row.append(
                        Paragraph(
                            safe_cell,
                            table_style
                        )
                    )

                formatted_table.append(
                    formatted_row
                )

            # ------------------------------------------
            # COLUMN COUNT
            # ------------------------------------------

            column_count = len(
                normalized_rows[0]
            )

            # ------------------------------------------
            # AVAILABLE WIDTH
            # ------------------------------------------

            available_width = (
                A4[0]
                - pdf.leftMargin
                - pdf.rightMargin
            )

            # ------------------------------------------
            # COLUMN WIDTHS
            # ------------------------------------------

            if column_count == 4:

                col_widths = [
                    available_width * 0.15,
                    available_width * 0.29,
                    available_width * 0.31,
                    available_width * 0.25
                ]

            elif column_count == 3:

                col_widths = [
                    available_width / 3
                ] * 3

            elif column_count == 2:

                col_widths = [
                    available_width / 2
                ] * 2

            else:

                col_widths = [
                    available_width / column_count
                ] * column_count

            # ------------------------------------------
            # CREATE TABLE
            # ------------------------------------------

            pdf_table = Table(
                formatted_table,
                colWidths=col_widths,
                repeatRows=1,
                hAlign="LEFT"
            )

            pdf_table.setStyle(
                TableStyle([
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        black
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, -1),
                        "NotoHindi"
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        2
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        2
                    )
                ])
            )

            page_story.append(
                Spacer(1, 5)
            )

            page_story.append(
                pdf_table
            )

            page_story.append(
                Spacer(1, 5)
            )

        # ==============================================
        # PAGE FITTING
        # ==============================================

        # Keep each source page together.
        #
        # If content is slightly too large, ReportLab
        # will shrink it rather than creating an
        # unexpected extra page.

        page_frame = KeepInFrame(
            A4[0] - pdf.leftMargin - pdf.rightMargin,
            A4[1] - pdf.topMargin - pdf.bottomMargin,
            page_story,
            mode="shrink"
        )

        story.append(
            page_frame
        )

        # ==============================================
        # NEXT PAGE
        # ==============================================

        if page_index < len(layout_pages) - 1:

            story.append(
                PageBreak()
            )

    # ==================================================
    # BUILD PDF
    # ==================================================

    pdf.build(
        story
    )

    buffer.seek(0)

    return buffer


# ===============================
#  VALIDATE HINDI TRANSLATE
# ===============================
def validate_hindi_translation(
    original_text,
    translated_text
):

    issues = []

    original_text = str(
        original_text or ""
    ).strip()

    translated_text = str(
        translated_text or ""
    ).strip()

    # --------------------------------------
    # BASIC EMPTY CHECK
    # --------------------------------------

    if original_text and not translated_text:

        issues.append(
            "Translated text is empty."
        )

        return {
            "valid": False,
            "issues": issues
        }

    if not translated_text:

        return {
            "valid": True,
            "issues": []
        }

    # --------------------------------------
    # KNOWN AWKWARD TRANSLATION PATTERNS
    # --------------------------------------

    suspicious_phrases = [

        "इंटरैक्टिवली रूप से",

        "interactively रूप से",

        "अनुमानित रूप से",

        "एक command लागू करने के लिए।",

        "प्रबंधित और निरीक्षण करें।",

        "बदल (replace कर)",

        "सुविधाजनक रूप से",

        "पूर्व स्थिति में लाएँ",

        "यह सुविधा देने वाले systems",

        "निदान करें; सामान्यतः",

    ]

    translated_lower = (
        translated_text.lower()
    )

    for phrase in suspicious_phrases:

        if phrase.lower() in translated_lower:

            issues.append(
                f"Potentially unnatural phrase: {phrase}"
            )

    # --------------------------------------
    # BROKEN SENTENCE CHECK
    # --------------------------------------

    broken_endings = [

        "के लिए।",

        "करने के लिए।",

        "जिसके लिए।",

        "जिससे।",

        "और।",

        "या।",

    ]

    lines = translated_text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        for ending in broken_endings:

            if line.endswith(ending):

                if len(line) > 25:

                    issues.append(
                        "Possible incomplete sentence: "
                        + line
                    )

                    break

    # --------------------------------------
    # HINDI PRESENCE CHECK
    # --------------------------------------
    #
    # Do NOT reject a translation merely
    # because it contains technical English.
    #
    # Linux documentation naturally contains:
    # commands, package names, paths,
    # programming terms, options, etc.
    #
    # Only reject a long translation if
    # Hindi is almost completely absent.
    # --------------------------------------

    hindi_chars = sum(
        1
        for char in translated_text
        if (
            "\u0900"
            <= char
            <= "\u097F"
        )
    )

    english_chars = sum(
        1
        for char in translated_text
        if (
            "a" <= char.lower() <= "z"
        )
    )

    total_language_chars = (
        hindi_chars
        + english_chars
    )

    if (
        total_language_chars > 250
        and hindi_chars < 20
    ):

        issues.append(
            "Translation contains very little Hindi "
            "and may not have been translated."
        )

    # --------------------------------------
    # FINAL RESULT
    # --------------------------------------

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }
    
    
# ================================
#  REPAIR HINDI TRANSLATE
# ================================   
def repair_hindi_translation(
    original_text,
    translated_text,
    issues,
    target_language="Hindi"
):

    if not translated_text:

        raise RuntimeError(
            "Cannot repair empty translation."
        )

    issue_text = "\n".join(
        f"- {issue}"
        for issue in issues
    )

    prompt = f"""
You are a professional technical editor
specializing in Hindi Computer Science documentation.

The following Hindi translation has quality problems.

Your task is to REWRITE the translation naturally
while preserving the exact technical meaning.

QUALITY PROBLEMS DETECTED:

{issue_text}

IMPORTANT RULES:

1. Write natural, professional Hindi.

2. Do not translate word-for-word.

3. Do not summarize.

4. Do not remove information.

5. Do not add information.

6. Preserve the exact meaning.

7. Keep standard technical terms in English
   when that is clearer.

8. Keep terms such as:
Linux, command, terminal, shell, Bash,
filesystem, directory, file, process, PID,
CPU, RAM, memory, storage, disk, partition,
package, service, kernel, script, function,
argument, parameter, output, input, permission,
user, group, root, server, client, network,
socket, port, URL, API, DNS, SSH, HTTP, HTTPS,
TLS, certificate, encryption, compiler,
debugging, system call, signal, pipeline,
redirection, cron, OpenSSL, Git, Python,
Node.js, JavaScript, Makefile, ACL

in English where appropriate.

9. Do NOT translate commands.

10. Do NOT translate command options.

11. Do NOT translate code.

12. Do NOT translate paths.

13. Do NOT translate URLs.

14. Do NOT translate variable names.

15. Do NOT translate programming syntax.

16. Do NOT translate Linux operators.

17. Preserve numbers exactly.

18. Preserve headings and numbering.

19. Preserve bullet points.

20. Preserve paragraph order.

21. Do not create incomplete sentences.

22. Do not use awkward Hindi-English combinations.

23. Technical English is allowed.
Natural Hindi is more important than forcing
every word into Hindi.

EXAMPLES:

BAD:
"interactively रूप से processes को प्रबंधित करें।"

GOOD:
"processes को interactive तरीके से प्रबंधित करें।"

BAD:
"कई results पर एक ही command लागू करने के लिए।"

GOOD:
"कई results पर एक ही command लागू करने के लिए इसका उपयोग करें।"

BAD:
"printf आउटपुट को अनुमानित रूप से फ़ॉर्मेट करता है।"

GOOD:
"printf आउटपुट को सटीक रूप से फ़ॉर्मेट करता है।"

ORIGINAL ENGLISH:

{original_text}

CURRENT HINDI:

{translated_text}

Return ONLY the corrected Hindi translation.
Do not explain the changes.

TABLE CELL RULE:
If the input is a short table cell, keep the translation concise.
Do not turn a short cell into a long explanation.
Preserve commands, options, paths, URLs, variables, operators,
numbers and technical terms exactly.
"""

    response = safe_generate_content(
        prompt,
        feature="PDF Translation Repair"
    )

    repaired_text = str(
        response or ""
    ).strip()

    if not repaired_text:

        raise RuntimeError(
            "Gemini returned an empty repaired translation."
        )

    return repaired_text




# =================================
#  QUALITY CHECK AND REPAIR PAGE
# =================================
def quality_check_and_repair_page(
    original_text,
    translated_text,
    max_repairs=1
):

    validation = validate_hindi_translation(
        original_text,
        translated_text
    )

    # --------------------------------------
    # ALREADY GOODss
    # --------------------------------------

    if validation["valid"]:

        return translated_text

    current_translation = translated_text

    # --------------------------------------
    # REPAIR
    # --------------------------------------

    for repair_attempt in range(
        max_repairs
    ):

        current_translation = (
            repair_hindi_translation(
                original_text,
                current_translation,
                validation["issues"],
                "Hindi"
            )
        )

        validation = (
            validate_hindi_translation(
                original_text,
                current_translation
            )
        )

        if validation["valid"]:

            return current_translation

    # --------------------------------------
    # REPAIR FAILED
    # --------------------------------------

    raise RuntimeError(
        "Hindi translation failed quality "
        "validation after automatic repair. "
        + " | ".join(
            validation["issues"]
        )
    )
    
    
# ===================================== 
#  quality check and repair table
# ======================================
def quality_check_and_repair_table(
    original_table,
    translated_table,
    max_repairs=1
):

    if not isinstance(
        original_table,
        list
    ):
        return translated_table

    if not isinstance(
        translated_table,
        list
    ):
        return translated_table

    repaired_table = []

    for original_row, translated_row in zip(
        original_table,
        translated_table
    ):

        repaired_row = []

        for original_cell, translated_cell in zip(
            original_row,
            translated_row
        ):

            original_cell = str(
                original_cell or ""
            ).strip()

            translated_cell = str(
                translated_cell or ""
            ).strip()

            # --------------------------------------
            # EMPTY CELL
            # --------------------------------------

            if not original_cell:

                repaired_row.append(
                    translated_cell
                )

                continue

            if not translated_cell:

                repaired_row.append(
                    translated_cell
                )

                continue

            # --------------------------------------
            # QUALITY CHECK
            # --------------------------------------

            result = validate_hindi_translation(
                original_cell,
                translated_cell
            )

            # --------------------------------------
            # ALREADY GOOD
            # --------------------------------------

            if result["valid"]:

                repaired_row.append(
                    translated_cell
                )

                continue

            # --------------------------------------
            # AUTO REPAIR
            # --------------------------------------

            repaired_cell = translated_cell

            for _ in range(max_repairs):

                repaired_cell = (
                    repair_hindi_translation(
                        original_cell,
                        repaired_cell,
                        result["issues"],
                        target_language="Hindi"
                    )
                )

                repaired_cell = str(
                    repaired_cell or ""
                ).strip()

                recheck = (
                    validate_hindi_translation(
                        original_cell,
                        repaired_cell
                    )
                )

                if recheck["valid"]:

                    break

                result = recheck

            # --------------------------------------
            # FINAL CELL RESULT
            # --------------------------------------

            final_check = (
                validate_hindi_translation(
                    original_cell,
                    repaired_cell
                )
            )

            if not final_check["valid"]:

                raise RuntimeError(
                    "Hindi table cell failed "
                    "quality validation after "
                    "automatic repair. "
                    + "; ".join(
                        final_check["issues"]
                    )
                )

            repaired_row.append(
                repaired_cell
            )

        repaired_table.append(
            repaired_row
        )

    return repaired_table


# ==============================================
#  GENERATE SUMMARY 
# ==============================================

def generate_summary(context):

    prompt = f"""
You are an AI document assistant.

Summarize the following PDF content.

PDF CONTENT:
-------------------------
{context}
-------------------------

Instructions:

- Give a clear and accurate summary.
- Use only the provided PDF content.
- Do not add outside information.
- Cover the main topics and important concepts.
- Use headings and bullet points.
- Keep the summary concise but useful for studying.

Return only the summary.
"""

    return safe_generate_content(prompt)


# =======================================
#  GENERATE KEY POINTS
# =======================================

def generate_key_points(context):

    prompt = f"""
You are an AI study assistant.

Extract the most important key points
from the following PDF content.

PDF CONTENT:
-------------------------
{context}
-------------------------

Instructions:

- Use only the provided PDF content.
- Do not add outside information.
- Focus on important concepts, definitions,
  classifications, formulas, and facts.
- Avoid unnecessary details.
- Use numbered points.
- Keep each point short and clear.
- Make it useful for exam preparation.

Return only the key points.
"""

    return safe_generate_content(prompt)


# ===================================
#  EXPLAIN SIMPLLY
# ====================================
def explain_simply(
    context,
    explain_mode="Beginner"
):

    mode_instructions = {

        "Beginner":
        """
        Explain the topic in very simple language.
        Assume the user is completely new to the topic.
        Use simple examples and avoid unnecessary
        technical terminology.
        """,

        "College Student":
        """
        Explain at a college-student level.
        Include important concepts, definitions,
        examples and relationships between concepts.
        """,

        "Interview Preparation":
        """
        Explain the topic for technical interviews.
        Focus on important concepts, practical points,
        common interview questions and key differences.
        """,

        "Exam Preparation":
        """
        Explain the topic for university exams.
        Focus on definitions, important points,
        examples, differences and concepts useful
        for writing exam answers.
        """,

        "Technical Deep Dive":
        """
        Give a detailed technical explanation.
        Cover concepts, internal working, relationships,
        examples, advantages, limitations and
        technical details.
        """
    }

    prompt = f"""
You are an AI PDF learning assistant.

Explain the provided PDF content according to
the selected explanation mode.

Selected Mode:
{explain_mode}

Instructions:
{mode_instructions.get(
    explain_mode,
    mode_instructions["Beginner"]
)}

Important rules:

- Use ONLY the provided PDF context.
- Do not add unsupported information.
- Do not mention these instructions.
- Do not say "I understand".
- Do not greet the user.
- Structure the explanation clearly.
- Use headings and bullet points where useful.
- Respect the selected response language.

Response language:
{st.session_state.get(
    "response_language",
    "English"
)}

PDF CONTEXT:
-------------------------
{context}
-------------------------

Generate the explanation now.
"""

    response = safe_generate_content(
        prompt
    )

    return response.strip()

# ==================================================
# PDF-SPECIFIC CHAT HISTORY
# ==================================================

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
    
if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
        
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
    
if "selected_pdfs" not in st.session_state:
    st.session_state.selected_pdfs = []

if "response_language" not in st.session_state:
    st.session_state.response_language = "English"
    
if "pdf_notes" not in st.session_state:
    st.session_state.pdf_notes = None

if "flashcards" not in st.session_state:
    st.session_state.flashcards = []

if "translated_pdf_text" not in st.session_state:
    st.session_state.translated_pdf_text = None

if "important_questions" not in st.session_state:
    st.session_state.important_questions = None

if "saved_answers" not in st.session_state:
    st.session_state.saved_answers = []

if "previous_pdf_selection" not in st.session_state:
    st.session_state.previous_pdf_selection = None
    
if "answer_feedback" not in st.session_state:
    st.session_state.answer_feedback = {}

if "confirm_clear_chat" not in st.session_state:
    st.session_state.confirm_clear_chat = False

if "explain_mode" not in st.session_state:
    st.session_state.explain_mode = "Beginner"

if "concept_map" not in st.session_state:
    st.session_state.concept_map = None

if "concept_levels" not in st.session_state:
    st.session_state.concept_levels = {}

if "selected_concept" not in st.session_state:
    st.session_state.selected_concept = None

if "compare_pdfs" not in st.session_state:
    st.session_state.compare_pdfs = False

if "response_cache" not in st.session_state:
    st.session_state.response_cache = (
        load_response_cache()
    )

if "processing_requests" not in st.session_state:
    st.session_state.processing_requests = set()

if "last_ai_request_time" not in st.session_state:
    st.session_state.last_ai_request_time = 0

if "ai_usage" not in st.session_state:
    st.session_state.ai_usage = {
        "total_requests": 0,
        "cache_hits": 0,
        "gemini_calls": 0,
        "failed_requests": 0,
        "features": {},
        "quota_protection": False
    }
    

# ==================================
# CLEAR PDF GENERATE CONTENT
# ==================================
def clear_pdf_generated_content():

    keys = [
        "pdf_summary",
        "key_points",
        "simple_explanation",
        "quiz",
        "quiz_result",
        "pdf_notes",
        "flashcards",
        "important_questions",
        "exam_questions",
        "translated_pdf_text",
        "suggested_questions",
        "pending_question",
        "confirm_delete"
    ]

    for key in keys:

        st.session_state.pop(
            key,
            None
        )



# ===================================
# GET CHAT HISTORY
# =====================================
def get_chat_history(filenames):

    if "chat_histories" not in st.session_state:

        st.session_state.chat_histories = {}

    if isinstance(filenames, str):

        filenames = [filenames]

    chat_key = "|".join(
        sorted(filenames)
    )

    if chat_key not in st.session_state.chat_histories:

        st.session_state.chat_histories[
            chat_key
        ] = []

    return st.session_state.chat_histories[
        chat_key
    ]
    

# =========================================
def get_multi_pdf_chat_key(
    filenames
):

    return " + ".join(
        sorted(filenames)
    )


# ==============================
#  CREATE CHAT EXPORT
# ==============================
def create_chat_export(
    messages,
    selected_pdf
):

    content = []

    content.append(
        f"AI PDF Assistant - {selected_pdf}"
    )

    content.append(
        "=" * 50
    )

    for message in messages:

        role = message["role"].upper()

        content.append(
            f"\n{role}\n"
        )

        content.append(
            message["content"]
        )

        content.append(
            "\n" + "-" * 50
        )

    return "\n".join(content)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    # ----------------------------------------------
    # APP TITLE
    # ----------------------------------------------

    st.title("🤖 AI PDF Assistant")

    st.caption(
        "Upload, manage and chat with your documents."
    )

    st.divider()


    # ----------------------------------------------
    # UPLOAD PDF
    # ----------------------------------------------

    st.subheader("📤 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        uploaded_file.seek(0)

        pdf_bytes = uploaded_file.getvalue()

        pdf_path = os.path.join(
            PDF_FOLDER,
            uploaded_file.name
        )

        # Always keep original PDF
        with open(pdf_path, "wb") as file:
            file.write(pdf_bytes)

        # Keep current upload in session too
        st.session_state["translation_pdf"] = pdf_bytes

        uploaded_file.seek(0)


        # for page_data in layout_pages:

        #     st.write(
        #         f"### Page {page_data['page']}"
        #     )

        #     st.write(
        #         f"📊 Tables found: "
        #         f"{len(page_data['tables'])}"
        #     )

        #     for table_index, table in enumerate(
        #         page_data["tables"],
        #         start=1
        #     ):

        #         st.write(
        #             f"**Table {table_index}**"
        #         )

        #         st.table(table)
    
        # ===================================
        # if structured_pages:

        #     first_page = structured_pages[0]

        #     if first_page["tables"]:

        #         st.subheader("🌐 Table Translation Test")

        #         original_table = first_page["tables"][0]

        #         translated_table = translate_table(
        #             original_table,
        #             "Hindi"
        #         )

        #         st.write("**Original Table:**")
        #         st.table(original_table)

        #         st.write("**Hindi Table:**")
        #         st.table(translated_table)

        #     else:
        #         st.warning(
        #             "No table found on the first page."
        #         )

    # ----------------------------------------------
    # PROCESS PDF
    # ----------------------------------------------

    if uploaded_file:
        
        MAX_FILE_SIZE = 50 * 1024 * 1024
        if uploaded_file.size > MAX_FILE_SIZE:

            st.error(
                "PDF is too large. "
                "Maximum allowed size is 50 MB."
            )
            st.stop()
        st.caption(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "⚙️ Process PDF",
            use_container_width=True
        ):
            # ------------------------------------------
            # DUPLICATE CHECK
            # ------------------------------------------

            if pdf_already_exists(
                uploaded_file.name
            ):

                st.warning(
                    f"'{uploaded_file.name}' "
                    "is already processed."
                )

            else:

                try:

                    # ======================================
                    # PROCESSING STATUS
                    # ======================================

                    with st.status(
                        "Processing PDF...",
                        expanded=True
                    ) as status:

                        # ----------------------------------
                        # STEP 1: EXTRACT TEXT
                        # ----------------------------------

                        st.write(
                            "📄 Reading PDF..."
                        )

                        pages = extract_pdf_pages(
                            uploaded_file
                        )


                        # ----------------------------------
                        # STEP 2: OCR FALLBACK
                        # ----------------------------------

                        if not pages:

                            st.write(
                                "🔍 No selectable text found."
                            )

                            st.write(
                                "🔠 Running OCR..."
                            )

                            pages = extract_pdf_with_ocr(
                                uploaded_file
                            )


                        if not pages:

                            status.update(
                                label="❌ PDF processing failed",
                                state="error"
                            )

                            st.error(
                                "Could not extract readable "
                                "text from this PDF."
                            )

                            st.stop()


                        st.write(
                            f"✓ Extracted "
                            f"{len(pages)} pages"
                        )


                        # ----------------------------------
                        # STEP 3: CREATE CHUNKS
                        # ----------------------------------

                        st.write(
                            "🧩 Creating text chunks..."
                        )

                        chunks, metadata = (
                            create_chunks_with_metadata(
                                pages,
                                uploaded_file.name
                            )
                        )


                        if not chunks:

                            status.update(
                                label="❌ Chunk creation failed",
                                state="error"
                            )

                            st.error(
                                "Could not create text chunks."
                            )

                            st.stop()


                        st.write(
                            f"✓ Created "
                            f"{len(chunks)} chunks"
                        )


                        # ----------------------------------
                        # STEP 4: EMBEDDINGS + CHROMADB
                        # ----------------------------------

                        st.write(
                            "🧠 Generating embeddings..."
                        )

                        st.write(
                            "💾 Storing data in ChromaDB..."
                        )

                        total_chunks = store_pdf(
                            chunks,
                            metadata,
                            uploaded_file.name
                        )


                        st.write(
                            f"✓ Stored "
                            f"{total_chunks} chunks"
                        )


                        # ----------------------------------
                        # SUCCESS
                        # ----------------------------------

                        status.update(
                            label="✅ PDF processed successfully",
                            state="complete",
                            expanded=False
                        )


                    st.success(
                        f"'{uploaded_file.name}' "
                        "is ready to chat."
                    )

                    st.rerun()


                except Exception as e:

                    st.error(
                        "Something went wrong while "
                        "processing the PDF."
                    )

                    st.exception(e)


    # ----------------------------------------------
    # RESPONSE LANGUAGE
    # ----------------------------------------------
    
    st.subheader(
        "🌐 Response Language"
    )

    response_language = st.selectbox(
        "Choose language",
        [
            "English",
            "Hindi",
            "Hinglish"
        ],
        index=[
            "English",
            "Hindi",
            "Hinglish"
        ].index(
            st.session_state.response_language
        )
    )

    st.session_state.response_language = (
        response_language
    )
    
    st.divider()
    
    
    # ----------------------------------------------
    # DOCUMENT LIBRARY
    # ----------------------------------------------

    st.subheader("📚 Your Documents")
    
    available_pdfs = get_available_pdfs()

    # ==================================================
    # PDF SEARCH & SELECTION
    # ==================================================


    if available_pdfs:

        search_pdf = st.text_input(
            "🔎 Search PDF",
            placeholder="Type PDF name..."
        )

        if search_pdf:

            filtered_pdfs = [
                pdf
                for pdf in available_pdfs
                if search_pdf.lower()
                in pdf.lower()
            ]

        else:

            filtered_pdfs = available_pdfs

        if filtered_pdfs:

            selected_pdfs = st.multiselect(
                "📚 Select PDF(s)",
                filtered_pdfs,
                default=(
                    [st.session_state.selected_pdf]
                    if st.session_state.get("selected_pdf")
                    in filtered_pdfs
                    else []
                )
            )
            
            
            # =============================
            
            current_selection = tuple(
                sorted(selected_pdfs)
            )

            previous_selection = st.session_state.get(
                "previous_pdf_selection"
            )

            if (
                previous_selection is not None
                and current_selection != previous_selection
            ):

                clear_pdf_generated_content()

            st.session_state.previous_pdf_selection = (
                current_selection
            )
            
            
            
            # =============================
            if selected_pdfs:

                selected_pdf = selected_pdfs[0]

                st.session_state.selected_pdf = (
                    selected_pdf
                )

                st.session_state.selected_pdfs = (
                    selected_pdfs
                )
                
                 # ==========================================
                # DELETE SELECTED PDF
                # ==========================================

                if st.button(
                    "🗑️ Delete Selected PDF",
                    use_container_width=True
                ):

                    delete_pdf(
                        selected_pdf
                    )

                    st.session_state.selected_pdfs = []

                    st.session_state.selected_pdf = None

                    clear_pdf_generated_content()

                    st.success(
                        f"'{selected_pdf}' deleted successfully."
                    )

                    st.rerun()
                    
                # ==========================================
                # SELECTED PDF DETAILS
                # ==========================================

                pdf_info = get_pdf_info(
                    selected_pdf
                )

                if pdf_info:

                    st.caption(
                        f"📄 {pdf_info['filename']}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "📑 Pages",
                            pdf_info.get(
                                "pages",
                                0
                            )
                        )

                    with col2:

                        st.metric(
                            "🧩 Chunks",
                            pdf_info.get(
                                "chunks",
                                0
                            )
                        )

            else:
                selected_pdf = None
                st.session_state.selected_pdf = None
                st.session_state.selected_pdfs = []
           
            

        else:

            selected_pdf = None

            st.warning(
                "No PDF found."
            )

    else:

        selected_pdf = None

        st.info(
            "Upload and process a PDF first."
        )
        
    # ==========================================
    # CHAT SEARCH
    # ==========================================

    st.subheader("🔎 Search Chat")

    chat_search = st.text_input(
        "Search previous questions...",
        key="chat_search",
        placeholder="e.g. inheritance"
    )
    
    
    # ==========================================
    # SAVED ANSWERS
    # ==========================================

    with st.expander(
        "📚 Saved Answers",
        expanded=False
    ):

        saved_answers = (
            st.session_state.get(
                "saved_answers",
                []
            )
        )

        if not saved_answers:

            st.info(
                "No saved answers yet."
            )

        else:

            saved_search = st.text_input(
                "Search saved answers...",
                key="saved_answer_search",
                placeholder="e.g. OOP"
            )

            search_term = (
                saved_search.strip().lower()
            )

            filtered_answers = []

            for item in saved_answers:

                question_text = item.get(
                    "question",
                    ""
                ).lower()

                answer_text = item.get(
                    "answer",
                    ""
                ).lower()

                if (
                    not search_term
                    or search_term in question_text
                    or search_term in answer_text
                ):

                    filtered_answers.append(
                        item
                    )

            if not filtered_answers:

                st.warning(
                    "No matching saved answers."
                )

            else:

                for index, item in enumerate(
                    filtered_answers
                ):

                    st.markdown(
                        f"**❓ {item.get('question', '')}**"
                    )

                    st.caption(
                        f"📄 {item.get('pdf', 'Unknown')}"
                    )

                    st.write(
                        item.get(
                            "answer",
                            ""
                        )
                    )
                    
                    

                    if st.button(
                        "🗑️",
                        key=f"delete_saved_{index}",
                        help="Remove saved answer"
                    ):

                        saved_id = item.get("id")

                        st.session_state.saved_answers = [
                            saved
                            for saved in st.session_state.saved_answers
                            if saved.get("id") != saved_id
                        ]

                        st.rerun()

                    st.divider()
    
    
        
    if selected_pdf:

        pdf_info = get_pdf_info(
            selected_pdf
        )

        with st.expander(
            "📄 PDF Information"
        ):

            st.write(
                f"**File:** "
                f"{pdf_info['filename']}"
            )

            st.write(
                "**Status:** ✅ Processed"
            )
        
        
        st.divider()
        st.subheader("🧠 Document Tools")
        if st.button(
            "📝 Summarize PDF",
            use_container_width=True
        ):
            with st.spinner(
                "Preparing PDF summary..."
            ):

                try:

                    summary = (
                        generate_large_document_summary(
                            selected_pdf
                        )
                    )

                    if not summary:

                        st.warning(
                            "No content found in this PDF."
                        )

                        st.stop()

                    st.session_state.pdf_summary = (
                        summary
                    )

                except RuntimeError as e:

                    st.error(
                        f"⚠️ {str(e)}"
                    )
        #  key button
        if st.button(
            "📌 Key Points",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting key points..."
            ):

                try:

                    documents, metadatas = (
                        get_all_pdf_chunks(
                            selected_pdf
                        )
                    )

                    if not documents:

                        st.warning(
                            "No content found in this PDF."
                        )

                        st.stop()

                    context = "\n\n".join(
                        documents
                    )

                    key_points = generate_key_points(
                        context
                    )

                    st.session_state.key_points = (
                        key_points
                    )

                except RuntimeError as e:

                    st.error(
                        f"⚠️ {str(e)}"
                    )
                
                
        # ==========================================
        # CONCEPT MAP
        # ==========================================

        st.markdown(
            "### 🗺️ Concept Map"
        )

        learning_purpose = st.selectbox(
            "Learning Purpose",
            [
                "General Learning",
                "Student",
                "Working Professional",
                "Interview Preparation",
                "Research / Academic"
            ],
            key="concept_learning_purpose"
        )

        if st.button(
            "🗺️ Build Concept Map",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing PDF concepts..."
            ):

                try:

                    documents = get_document_context(
                        selected_pdf
                    )

                    if not documents:

                        st.warning(
                            "No content found in this PDF."
                        )

                        st.stop()

                    groups = create_chunk_groups(
                        documents,
                        max_chars=10000
                    )

                    context = groups[0]

                    response = generate_concept_map(
                        context,
                        learning_purpose
                    )

                    concepts = parse_concept_map(
                        response
                    )
                    if concepts:
                        concept_levels = (
                            build_concept_levels(
                                concepts
                            )
                        )

                        st.session_state.concept_levels = (
                            concept_levels
                        )

                    if not concepts:

                        st.error(
                            "Could not create the concept map."
                        )

                        st.stop()

                    st.session_state.concept_map = (
                        concepts
                    )

                except RuntimeError as e:
                    st.error(
                        f"⚠️ {str(e)}"
                    )
        
        # ==========================================
        # EXPLAIN MODE
        # ==========================================

        st.markdown("### 🧠 Explain Mode")

        explain_mode = st.selectbox(
            "Choose explanation level",
            [
                "Beginner",
                "College Student",
                "Interview Preparation",
                "Exam Preparation",
                "Technical Deep Dive"
            ],
            index=[
                "Beginner",
                "College Student",
                "Interview Preparation",
                "Exam Preparation",
                "Technical Deep Dive"
            ].index(
                st.session_state.explain_mode
            ),
            key="explain_mode_select"
        )

        st.session_state.explain_mode = (
            explain_mode
        )

        if st.button(
            "🧠 Explain PDF",
            use_container_width=True
        ):

            with st.spinner(
                f"Preparing {explain_mode} explanation..."
            ):

                try:

                    documents, metadatas = (
                        get_all_pdf_chunks(
                            selected_pdf
                        )
                    )

                    if not documents:

                        st.warning(
                            "No content found in this PDF."
                        )

                        st.stop()

                    context = "\n\n".join(
                        documents
                    )

                    explanation = explain_simply(
                        context,
                        explain_mode
                    )

                    if not explanation:

                        st.error(
                            "Could not generate explanation."
                        )

                        st.stop()

                    st.session_state.simple_explanation = (
                        explanation
                    )

                except RuntimeError as e:
                    st.error(
                        f"⚠️ {str(e)}"
                    )
        
        #  quiz button 
        if st.button(
            "❓ Generate Quiz",
            use_container_width=True
        ):

            with st.spinner(
                "Generating quiz..."
            ):

                try:

                    documents = get_document_context(
                        selected_pdf
                    )

                    if not documents:

                        st.warning(
                            "No content found in this PDF."
                        )

                        st.stop()

                    groups = create_chunk_groups(
                        documents,
                        max_chars=10000
                    )

                    context = groups[0]

                    quiz_text = generate_quiz(
                        context
                    )

                    quiz = parse_quiz(
                        quiz_text
                    )

                    if not quiz:

                        st.error(
                            "Could not create the quiz."
                        )

                        st.stop()

                    st.session_state.quiz = quiz


                except RuntimeError as e:

                    st.error(
                        f"⚠️ {str(e)}"
                    )
                    
        # generate questions
        if st.button(
            "💡 Generate Questions",
            use_container_width=True
        ):

            with st.spinner(
                "Generating suggested questions..."
            ):

                try:

                    documents = get_document_context(
                        selected_pdf
                    )

                    if not documents:

                        st.warning(
                            "No content found in this PDF."
                        )

                        st.stop()

                    groups = create_chunk_groups(
                        documents,
                        max_chars=8000
                    )

                    response = (
                        generate_suggested_questions(
                            groups[0]
                        )
                    )

                    questions = (
                        parse_suggested_questions(
                            response
                        )
                    )

                    st.session_state.suggested_questions = (
                        questions
                    )

                except RuntimeError as e:

                    st.error(
                        f"⚠️ {str(e)}"
                    )              
                               
        # study notes
        if st.button(
            "📚 Generate Notes",
            use_container_width=True
        ):

            with st.spinner(
                "Generating study notes..."
            ):

                try:

                    documents = get_document_context(
                        selected_pdf
                    )

                    if not documents:

                        st.warning(
                            "No content found in this PDF."
                        )

                        st.stop()

                    groups = create_chunk_groups(
                        documents,
                        max_chars=10000
                    )

                    context = groups[0]

                    notes = generate_notes(
                        context,
                        st.session_state.response_language
                    )

                    if not notes:

                        st.error(
                            "Could not generate study notes."
                        )

                        st.stop()

                    st.session_state.pdf_notes = (
                        notes
                    )

                except RuntimeError as e:

                    st.error(
                        f"⚠️ {str(e)}"
                    )
                    
        
        # ==========================================
        # IMPORTANT QUESTIONS
        # ==========================================

        if st.button(
            "💡 Important Questions",
            use_container_width=True
        ):

            with st.spinner(
                "Generating important exam questions..."
            ):

                try:

                    documents = get_document_context(
                        selected_pdf,
                        question
                    )

                    if not documents:

                        st.warning(
                            "No content found in this PDF."
                        )

                        st.stop()

                    groups = create_chunk_groups(
                        documents,
                        max_chars=10000
                    )

                    context = groups[0]

                    questions = generate_important_questions(
                        context,
                        st.session_state.response_language
                    )

                    if not questions:

                        st.error(
                            "Could not generate important questions."
                        )

                        st.stop()

                    st.session_state.important_questions = (
                        questions
                    )

                except RuntimeError as e:

                    st.error(
                        f"⚠️ {str(e)}"
                    )
                    
        # flashcards

        if st.button(
            "🃏 Generate Flashcards",
            use_container_width=True
        ):

            with st.spinner(
                "Generating flashcards..."
            ):

                try:

                    documents = get_document_context(
                        selected_pdf
                    )

                    if not documents:

                        st.warning(
                            "No content found in this PDF."
                        )

                        st.stop()

                    groups = create_chunk_groups(
                        documents,
                        max_chars=10000
                    )

                    context = groups[0]

                    flashcards_text = generate_flashcards(
                        context,
                        st.session_state.response_language
                    )

                    try:

                        flashcards = json.loads(
                            flashcards_text
                        )

                    except json.JSONDecodeError:

                        st.error(
                            "Could not parse flashcards."
                        )

                        st.stop()

                    if not isinstance(
                        flashcards,
                        list
                    ):

                        st.error(
                            "Invalid flashcard format."
                        )

                        st.stop()

                    st.session_state.flashcards = (
                        flashcards
                    )

                except RuntimeError as e:

                    st.error(
                        f"⚠️ {str(e)}"
                    )
        
        
        
        # ==========================================
        # TRANSLATION ESTIMATE
        # ==========================================

        preview_pages = (
            get_pdf_pages_for_translation(
                selected_pdf
            )
        )

        preview_cache = (
            load_translation_cache(
                "Hindi"
            )
        )

        translation_estimate = (
            estimate_translation_cost(
                preview_pages,
                preview_cache
            )
        )

        with st.expander(
            "📊 Translation Estimate",
            expanded=False
        ):

            st.write(
                f"📄 Pages: "
                f"{translation_estimate['total_pages']}"
            )

            st.write(
                f"💾 Cached pages: "
                f"{translation_estimate['cached_pages']}"
            )

            st.write(
                f"🤖 New translation units: "
                f"{translation_estimate['new_pages']}"
            )

            st.write(
                f"🧮 Estimated API requests: "
                f"~{translation_estimate['estimated_requests']}"
            )

            st.caption(
                "This is an estimate. Actual API quota "
                "depends on token usage, model limits "
                "and your Gemini account quota."
            )
            
            
        # ==========================================
        # PDF TRANSLATION
        # ==========================================
        
        
        translation_batches = (
            create_translation_batches(
                preview_pages,
                preview_cache
            )
        )

        st.write(
            f"Translation batches: "
            f"{len(translation_batches)}"
        )


        if st.button(
            "🌐 Translate PDF to Hindi",
            use_container_width=True
        ):

            selected_pdf = st.session_state.get(
                "selected_pdf"
            )

            selected_pdfs = st.session_state.get(
                "selected_pdfs",
                []
            )

            if not selected_pdf and selected_pdfs:
                selected_pdf = selected_pdfs[0]

            if not selected_pdf:
                st.error(
                    "Please select a PDF first."
                )
                st.stop()

            pdf_path = os.path.join(
                PDF_FOLDER,
                selected_pdf
            )

            if not os.path.isfile(pdf_path):

                st.error(
                    f"Original PDF is not available for "
                    f"'{selected_pdf}'. Please upload this "
                    "PDF again once."
                )

                st.stop()

            with open(
                pdf_path,
                "rb"
            ) as file:

                pdf_bytes = file.read()

            with st.spinner(
                "Preparing PDF layout for translation..."
            ):

                try:

                    pdf_stream = BytesIO(
                        pdf_bytes
                    )

                    layout_pages = extract_pdf_layout(
                        pdf_stream
                    )

                except Exception as e:

                    st.error(
                        "Could not read the PDF."
                    )

                    st.exception(e)

                    st.stop()

            
            # ------------------------------------------
            # CHECK EXTRACTED PDF LAYOUT
            # ------------------------------------------

            if not layout_pages:

                st.error(
                    "Could not extract PDF layout."
                )

                st.stop()

            st.write(
                "📄 Pages detected:",
                len(layout_pages)
            )


            # ------------------------------------------
            # PREPARE STRUCTURED PAGES
            # ------------------------------------------

            structured_pages = (
                prepare_layout_for_translation(
                    layout_pages
                )
            )

            if not structured_pages:

                st.error(
                    "Could not prepare PDF pages for translation."
                )

                st.stop()

            st.write(
                "✅ Structured pages:",
                len(structured_pages)
            )


            # ------------------------------------------
            # LOAD CHECKPOINT
            # ------------------------------------------

            checkpoint = (
                load_page_translation_checkpoint(
                    selected_pdf,
                    "Hindi"
                )
            )

            translated_pages = checkpoint.get(
                "pages",
                {}
            )

            translation_failures = checkpoint.get(
                "failures",
                {}
            )

            # total_pages = len(
            #     structured_pages
            # )


            # ------------------------------------------
            # TRANSLATE PDF USING BATCHES
            # ------------------------------------------

            translation_batches = (
                create_translation_batches(
                    structured_pages,
                    {},
                    translated_pages=translated_pages,
                    max_chars=8000,
                    max_pages=1
                )
            )

            total_batches = len(
                translation_batches
            )

            batch_progress = st.progress(0)

            # =====================================
            for batch_index, batch in enumerate(translation_batches):
                if not batch:
                    continue

                batch_page_numbers = [
                    str(item["page"])
                    for item in batch
                ]

                try:
                    # -----------------------------------------
                    # 1. Translate complete batch
                    # -----------------------------------------
                    translated_batch = translate_pdf_batch(
                        batch,
                        "Hindi"
                    )

                    # -----------------------------------------
                    # 2. Validate complete batch BEFORE saving
                    # -----------------------------------------
                    validated_batch_results = {}

                    for item in batch:

                        page_number = str(item["page"])

                        result = translated_batch.get(page_number)

                        if not result:
                            raise RuntimeError(
                                f"Page {page_number} was not returned "
                                f"in batch translation."
                            )

                        translated_text = str(
                            result.get("translation", "")
                        ).strip()

                        translated_tables = result.get(
                            "translated_tables",
                            []
                        )

                        original_text = str(
                            item.get("text", "")
                        ).strip()

                        original_tables = item.get(
                            "tables",
                            []
                        )

                        # -----------------------------------------
                        # 3. Basic content validation
                        # -----------------------------------------
                        if not translated_text and not translated_tables:
                            raise RuntimeError(
                                f"Page {page_number} returned "
                                f"an empty translation."
                            )

                        # -----------------------------------------
                        # 4. Table count validation
                        # -----------------------------------------
                        if len(translated_tables) != len(original_tables):
                            raise RuntimeError(
                                f"Page {page_number}: table count changed. "
                                f"Expected {len(original_tables)}, "
                                f"got {len(translated_tables)}."
                            )

                        # -----------------------------------------
                        # 5. Row / column validation
                        # -----------------------------------------
                        for table_index, (
                            original_table,
                            translated_table
                        ) in enumerate(
                            zip(
                                original_tables,
                                translated_tables
                            ),
                            start=1
                        ):

                            if len(translated_table) != len(original_table):
                                raise RuntimeError(
                                    f"Page {page_number}, "
                                    f"table {table_index}: "
                                    f"row count changed. "
                                    f"Expected {len(original_table)}, "
                                    f"got {len(translated_table)}."
                                )

                            for row_index, (
                                original_row,
                                translated_row
                            ) in enumerate(
                                zip(
                                    original_table,
                                    translated_table
                                ),
                                start=1
                            ):

                                if len(translated_row) != len(original_row):
                                    raise RuntimeError(
                                        f"Page {page_number}, "
                                        f"table {table_index}, "
                                        f"row {row_index}: "
                                        f"column count changed. "
                                        f"Expected {len(original_row)}, "
                                        f"got {len(translated_row)}."
                                    )

                        # -----------------------------------------
                        # 6. Prepare validated result
                        # -----------------------------------------
                        validated_batch_results[page_number] = {
                            "fingerprint": get_page_fingerprint(item),
                            "translation": translated_text,
                            "translated_tables": translated_tables
                        }

                    # -----------------------------------------
                    # 7. Save ONLY after entire batch passes
                    # -----------------------------------------
                    for page_number, result in validated_batch_results.items():

                        translated_pages[page_number] = result

                        translation_failures.pop(
                            page_number,
                            None
                        )

                    # -----------------------------------------
                    # 8. Save checkpoint after successful batch
                    # -----------------------------------------
                    save_page_translation_checkpoint(
                        selected_pdf,
                        "Hindi",
                        translated_pages,
                        translation_failures
                    )

                except Exception as e:

                    # -----------------------------------------
                    # 9. Do NOT save partial results from
                    #    this failed batch
                    # -----------------------------------------
                    for page_number in batch_page_numbers:

                        translation_failures[page_number] = {
                            "error": str(e)
                        }

                    # -----------------------------------------
                    # 10. Save failure checkpoint
                    # -----------------------------------------
                    save_page_translation_checkpoint(
                        selected_pdf,
                        "Hindi",
                        translated_pages,
                        translation_failures
                    )

                    st.error(
                        f"❌ Batch failed "
                        f"(Pages {', '.join(batch_page_numbers)}): "
                        f"{e}"
                    )

                # -----------------------------------------
                # 11. Progress
                # -----------------------------------------
                batch_progress.progress(
                    (batch_index + 1) / total_batches
                )

                # -----------------------------------------
                # 12. Small delay between batches
                # -----------------------------------------
                if batch_index + 1 < total_batches:
                    time.sleep(2)


            # ------------------------------------------
            # CHECK ALL TRANSLATED PAGES
            # ------------------------------------------

            missing_pages = []
            invalid_pages = []

            for page_data in structured_pages:

                page_number = str(
                    page_data["page"]
                )

                original_text = str(
                    page_data.get("text", "")
                ).strip()

                original_tables = page_data.get(
                    "tables",
                    []
                )

                saved = translated_pages.get(
                    page_number
                )

                # --------------------------------------
                # PAGE EXISTENCE
                # --------------------------------------

                if not isinstance(
                    saved,
                    dict
                ):
                    missing_pages.append(
                        page_number
                    )
                    continue

                translated_text = str(
                    saved.get(
                        "translation",
                        ""
                    )
                ).strip()

                translated_tables = saved.get(
                    "translated_tables",
                    []
                )

                # --------------------------------------
                # FINGERPRINT VALIDATION
                # --------------------------------------

                saved_fingerprint = saved.get(
                    "fingerprint"
                )

                current_fingerprint = (
                    get_page_fingerprint(
                        page_data
                    )
                )

                if saved_fingerprint != current_fingerprint:

                    invalid_pages.append(
                        page_number
                    )

                    continue

                # --------------------------------------
                # TEXT VALIDATION
                # --------------------------------------

                if (
                    not translated_text
                    and not original_tables
                    and original_text
                ):

                    invalid_pages.append(
                        page_number
                    )

                    continue

                # --------------------------------------
                # TABLE LIST VALIDATION
                # --------------------------------------

                if not isinstance(
                    translated_tables,
                    list
                ):

                    invalid_pages.append(
                        page_number
                    )

                    continue

                if len(translated_tables) != len(
                    original_tables
                ):

                    invalid_pages.append(
                        page_number
                    )

                    continue

                # --------------------------------------
                # TABLE STRUCTURE VALIDATION
                # --------------------------------------

                page_table_valid = True

                for original_table, translated_table in zip(
                    original_tables,
                    translated_tables
                ):

                    if not isinstance(
                        translated_table,
                        list
                    ):

                        page_table_valid = False
                        break

                    # ----------------------------------
                    # ROW COUNT
                    # ----------------------------------

                    if len(original_table) != len(
                        translated_table
                    ):

                        page_table_valid = False
                        break

                    # ----------------------------------
                    # COLUMN COUNT
                    # ----------------------------------

                    for original_row, translated_row in zip(
                        original_table,
                        translated_table
                    ):

                        if not isinstance(
                            translated_row,
                            list
                        ):

                            page_table_valid = False
                            break

                        if len(original_row) != len(
                            translated_row
                        ):

                            page_table_valid = False
                            break

                    if not page_table_valid:
                        break

                if not page_table_valid:

                    invalid_pages.append(
                        page_number
                    )


            # ------------------------------------------
            # FINAL PROBLEM PAGE CHECK
            # ------------------------------------------

            if missing_pages or invalid_pages:

                all_problem_pages = sorted(
                    set(
                        missing_pages
                        + invalid_pages
                    ),
                    key=lambda x: int(x)
                )

                st.error(
                    "❌ Hindi PDF translation is incomplete."
                )

                st.warning(
                    "Problem pages: "
                    + ", ".join(
                        all_problem_pages
                    )
                )

                st.info(
                    "No incomplete Hindi PDF will be created. "
                    "Please translate the problem pages again."
                )

                st.stop()


            # ------------------------------------------
            # PREPARE OUTPUT PAGES
            # ------------------------------------------

            output_pages = []

            for page_data in structured_pages:

                page_number = str(
                    page_data["page"]
                )

                saved = translated_pages.get(
                    page_number
                )

                output_pages.append({
                    "page":
                        page_data["page"],

                    "translated_text":
                        str(
                            saved.get(
                                "translation",
                                ""
                            )
                        ).strip(),

                    "translated_tables":
                        saved.get(
                            "translated_tables",
                            []
                        )
                })


            # ------------------------------------------
            # FINAL OUTPUT PAGE COUNT
            # ------------------------------------------

            if len(output_pages) != len(
                structured_pages
            ):

                st.error(
                    "❌ Final Hindi PDF page count mismatch."
                )

                st.stop()


            # ------------------------------------------
            # CREATE HINDI PDF
            # ------------------------------------------

            with st.spinner(
                "Creating Hindi PDF..."
            ):

                try:

                    translated_pdf = (
                        create_hindi_pdf(
                            output_pages
                        )
                    )

                except Exception as e:

                    st.error(
                        f"⚠️ Hindi PDF creation failed: {e}"
                    )

                    st.stop()


            # ------------------------------------------
            # SAVE TEXT IN SESSION
            # ------------------------------------------

            st.session_state.translated_pdf_text = (
                "\n\n".join(
                    page.get(
                        "translated_text",
                        ""
                    )
                    for page in output_pages
                )
            )


            # ------------------------------------------
            # DOWNLOAD
            # ------------------------------------------

            st.download_button(
                "📥 Download Hindi PDF",

                data=translated_pdf,

                file_name=(
                    f"{os.path.splitext(selected_pdf)[0]}"
                    "_Hindi.pdf"
                ),

                mime="application/pdf",

                use_container_width=True
            )

            st.success(
                "✅ Hindi PDF created successfully."
            )

                
    
        
        # ==========================================
        # PDF COMPARISON
        # ==========================================

        selected_pdfs = st.session_state.get(
            "selected_pdfs",
            []
        )

        st.markdown(
            "### 🔍 Compare PDFs"
        )

        if len(selected_pdfs) < 2:

            st.button(
                "🔍 Compare PDFs",
                use_container_width=True,
                disabled=True
            )

            st.caption(
                "Select 2 PDFs to compare them."
            )

        elif len(selected_pdfs) == 2:

            if st.button(
                "🔍 Compare PDFs",
                use_container_width=True
            ):

                st.session_state.compare_pdfs = True

        else:

            st.warning(
                "Please select exactly 2 PDFs "
                "for comparison."
            )


        # ------------------------------------------
        # CURRENT PDF
        # ------------------------------------------

        st.caption(
            f"Current PDF: {selected_pdf}"
        )


        # ------------------------------------------
        # CHAT HISTORY
        # ------------------------------------------

        messages = get_chat_history(
            st.session_state.selected_pdfs
        )


        # ------------------------------------------
        # CHAT CONTROLS
        # ------------------------------------------

        col1,col2 = st.columns(2)


        with col1:

            # ==========================================
            # CLEAR CHAT
            # ==========================================

            if st.button(
                "🧹 Clear Chat",
                use_container_width=True
            ):

                st.session_state.confirm_clear_chat = True


            if st.session_state.get(
                "confirm_clear_chat",
                False
            ):

                st.warning(
                    "⚠️ Are you sure you want to clear "
                    "this chat?"
                )

                col1, col2 = st.columns(2)

                with col1:

                    if st.button(
                        "✅ Yes, Clear",
                        use_container_width=True,
                        key="confirm_clear_chat_btn"
                    ):

                        clear_current_chat()

                        st.session_state.confirm_clear_chat = (
                            False
                        )

                        st.success(
                            "Chat cleared successfully."
                        )

                        st.rerun()

                with col2:

                    if st.button(
                        "❌ Cancel",
                        use_container_width=True,
                        key="cancel_clear_chat_btn"
                    ):

                        st.session_state.confirm_clear_chat = (
                            False
                        )

                        st.rerun()
                
        
        with col2:  
            if st.button(
                "➕ New Chat",
                use_container_width=True
            ):

                st.session_state.chat_histories[
                    selected_pdf
                ] = []

                st.session_state.suggested_questions = []

                st.session_state.pending_question = None

                st.session_state.pop(
                    "pdf_summary",
                    None
                )

                st.session_state.pop(
                    "key_points",
                    None
                )

                st.session_state.pop(
                    "simple_explanation",
                    None
                )

                st.session_state.pop(
                    "quiz",
                    None
                )

                st.session_state.pop(
                    "quiz_result",
                    None
                )

                st.rerun()


        with col1:
            # ==================================================
            # DELETE PDF
            # ==================================================

            if selected_pdf:

                if st.button(
                    "🗑️ Delete PDF",
                    use_container_width=True
                ):

                    st.session_state.confirm_delete = True


            # ==================================================
            # DELETE CONFIRMATION
            # ==================================================

            if st.session_state.get(
                "confirm_delete",
                False
            ):

                st.warning(
                    f"Are you sure you want to delete "
                    f"**{selected_pdf}**?"
                )

                col1, col2 = st.columns(2)

                with col1:

                    if st.button(
                        "✅ Yes, Delete",
                        key="confirm_delete_yes",
                        use_container_width=True
                    ):

                        delete_pdf(
                            selected_pdf
                        )

                        st.session_state.confirm_delete = False

                        st.success(
                            "PDF data deleted successfully."
                        )

                        st.rerun()

                with col2:

                    if st.button(
                        "❌ Cancel",
                        key="confirm_delete_no",
                        use_container_width=True
                    ):

                        st.session_state.confirm_delete = False

                        st.rerun()
            
            
            with st.sidebar:
                with st.expander(
                    "📊 AI Usage"
                ):
                    show_ai_usage_dashboard()
        
        # ==========================================
        # EXPORT CHAT
        # ==========================================

        if messages:

            st.subheader(
                "📥 Export Chat"
            )

            chat_text = ""

            for message in messages:

                role = (
                    "You"
                    if message["role"] == "user"
                    else "AI Assistant"
                )

                chat_text += (
                    f"{role}:\n"
                    f"{message['content']}\n\n"
                )

            col1, col2 = st.columns(2)

            with col1:

                st.download_button(
                    "📄 Download TXT",
                    data=chat_text,
                    file_name=(
                        f"{os.path.splitext(selected_pdf)[0]}"
                        "_Chat.txt"
                    ),
                    mime="text/plain",
                    use_container_width=True
                )

            with col2:

                import json

                chat_json = json.dumps(
                    messages,
                    ensure_ascii=False,
                    indent=2
                )

                st.download_button(
                    "📋 Download JSON",
                    data=chat_json,
                    file_name=(
                        f"{os.path.splitext(selected_pdf)[0]}"
                        "_Chat.json"
                    ),
                    mime="application/json",
                    use_container_width=True
                )
        



    else:

        selected_pdf = None

        messages = []

        st.info(
            "No processed PDFs yet."
        )


    # ----------------------------------------------
    # SIDEBAR FOOTER
    # ----------------------------------------------

    st.divider()

    st.caption(
        "🤖 AI PDF Assistant"
    )

    # st.caption(
    #     "Powered by RAG + ChromaDB + Gemini"
    # )
    
# ==================================================
# MAIN CHAT AREA
# ==================================================

if selected_pdf:

    # Get chat history for selected PDF
    selected_chat_pdfs = (
        st.session_state.get(
            "selected_pdfs",
            []
        )
    )

    if selected_chat_pdfs:

        chat_key = get_multi_pdf_chat_key(
            selected_chat_pdfs
        )

        messages = get_chat_history(
            chat_key
        )

    else:

        messages = []

    st.title(
        "🤖 AI PDF Assistant"
    )

    st.caption(
        "Chatting with: "
        + ", ".join(
            st.session_state.selected_pdfs
        )
    )
    
    
    # ==========================================
    # PDF INFORMATION
    # ==========================================

    pdf_info = get_pdf_info(
        selected_pdf
    )

    if pdf_info:

        with st.expander(
            "📊 PDF Information",
            expanded=False
        ):

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "📑 Pages",
                    pdf_info["pages"]
                )

            with col2:

                st.metric(
                    "🧩 Chunks",
                    pdf_info["chunks"]
                )

            with col3:

                st.metric(
                    "📝 Characters",
                    f"{pdf_info['characters']:,}"
                )

            with col4:

                st.metric(
                    "✅ Status",
                    pdf_info["status"]
                )

            st.caption(
                f"📄 {pdf_info['filename']}"
            )
    
    

    # ==========================================
    # PDF SUMMARY
    # ==========================================

    if "pdf_summary" in st.session_state:

        with st.expander(
            "📝 PDF Summary",
            expanded=True
        ):

            st.markdown(
                st.session_state.pdf_summary
            )

    # ==========================================
    # KEY POINTS
    # ==========================================

    if "key_points" in st.session_state:

        with st.expander(
            "📌 Key Points",
            expanded=True
        ):

            st.markdown(
                st.session_state.key_points
            )
    
    
    # ==========================================
    # PDF NOTES
    # ==========================================
    if st.session_state.get("pdf_notes"):
        with st.expander(
            "📚 AI Study Notes",
            expanded=True
        ):

            st.markdown(
                st.session_state.pdf_notes
            )
    

    
    # ============================================
    # FLASH CARDS   
    # ============================================
    
    if st.session_state.get(
        "flashcards"
    ):

        with st.expander(
            "🃏 AI Flashcards",
            expanded=True
        ):

            for i, card in enumerate(
                st.session_state.flashcards
            ):

                if not isinstance(
                    card,
                    dict
                ):

                    continue

                question = card.get(
                    "question",
                    ""
                )

                answer = card.get(
                    "answer",
                    ""
                )

                st.markdown(
                    f"### 🃏 Card {i + 1}"
                )

                st.markdown(
                    f"**Q:** {question}"
                )

                with st.expander(
                    "👁️ Show Answer",
                    expanded=False
                ):

                    st.markdown(
                        f"**A:** {answer}"
                    )

                st.divider()
    

    # ==========================================
    # VISUAL CONCEPT MAP
    # ==========================================

    if st.session_state.get("concept_map"):

        with st.expander(
            "🗺️ Visual Concept Map",
            expanded=True
        ):

            concepts = (
                st.session_state.concept_map
            )

            # --------------------------------------
            # FIND ROOT
            # --------------------------------------

            root = next(
                (
                    c for c in concepts
                    if c.get("parent") is None
                ),
                None
            )

            if not root:

                st.error(
                    "Could not identify the main concept."
                )

            else:

                nodes = []
                edges = []

                children = {}

                for concept in concepts:

                    name = concept.get(
                        "name",
                        ""
                    ).strip()

                    parent = concept.get(
                        "parent"
                    )

                    if not name:
                        continue

                    if parent not in children:

                        children[parent] = []

                    children[parent].append(
                        name
                    )

                # ----------------------------------
                # CREATE NODES
                # ----------------------------------

                all_names = set()

                for concept in concepts:

                    name = concept.get(
                        "name",
                        ""
                    ).strip()

                    if not name:
                        continue

                    if name in all_names:
                        continue

                    all_names.add(name)

                    is_root = (
                        name == root["name"]
                    )

                    nodes.append(
                        Node(
                            id=name,
                            label=name,
                            size=38 if is_root else 25,
                            shape="dot"
                        )
                    )

                # ----------------------------------
                # CREATE EDGES
                # ----------------------------------

                for concept in concepts:

                    name = concept.get(
                        "name",
                        ""
                    ).strip()

                    parent = concept.get(
                        "parent"
                    )

                    if parent:

                        edges.append(
                            Edge(
                                source=parent,
                                target=name
                            )
                        )

                # ----------------------------------
                # GRAPH CONFIG
                # ----------------------------------

                config = Config(
                    width="100%",
                    height=650,
                    directed=True,

                    physics={
                        "enabled": False
                    },

                    interaction={
                        "dragNodes": True,
                        "dragView": True,
                        "zoomView": True
                    },

                    nodes={
                        "font": {
                            "size": 16
                        }
                    },

                    edges={
                        "arrows": {
                            "to": {
                                "enabled": True
                            }
                        }
                    },

                    layout={
                        "hierarchical": {
                            "enabled": True,
                            "direction": "UD",
                            "sortMethod": "directed",
                            "levelSeparation": 150,
                            "nodeSpacing": 180,
                            "treeSpacing": 220
                        }
                    }
                )

                selected_node = agraph(
                    nodes=nodes,
                    edges=edges,
                    config=config
                )
                

                if selected_node:

                    st.session_state.selected_concept = (
                        selected_node
                    )
                    
                
                if st.session_state.get(
                    "selected_concept"
                ):

                    selected_concept = (
                        st.session_state.selected_concept
                    )

                    st.markdown(
                        f"### 📌 {selected_concept}"
                    )

                    if st.button(
                        "🧠 Explain This Concept",
                        key="explain_selected_concept"
                    ):

                        with st.spinner(
                            "Explaining concept..."
                        ):

                            try:

                                documents = get_document_context(
                                    selected_pdf
                                )

                                if not documents:

                                    st.warning(
                                        "No PDF content found."
                                    )

                                else:

                                    context = "\n\n".join(
                                        documents
                                    )

                                    prompt = f"""
                You are an AI PDF learning assistant.

                Explain ONLY the selected concept using
                the provided PDF context.

                Selected Concept:
                {selected_concept}

                Response Language:
                {st.session_state.get(
                    "response_language",
                    "English"
                )}

                Learning Purpose:
                {st.session_state.get(
                    "concept_learning_purpose",
                    "General Learning"
                )}

                Explain directly.

                Include:

                1. What it is
                2. How it works
                3. Example
                4. Related concepts

                Rules:

                - Use ONLY the PDF context.
                - Do not use outside knowledge.
                - Do not mention these instructions.
                - Do not greet the user.

                PDF CONTEXT:
                -------------------------
                {context}
                -------------------------
                """

                                    explanation = (
                                        safe_generate_content(
                                            prompt
                                        )
                                    )

                                    if explanation:

                                        st.session_state[
                                            "selected_concept_explanation"
                                        ] = explanation.strip()

                                    else:

                                        st.warning(
                                            "No explanation was generated."
                                        )

                            except RuntimeError as e:

                                st.error(
                                    f"⚠️ {str(e)}"
                                )

                            except Exception as e:

                                st.error(
                                    f"⚠️ Could not explain "
                                    f"'{selected_concept}': {str(e)}"
                                )
                                
                if st.session_state.get(
                    "selected_concept_explanation"
                ):

                    st.markdown(
                        "### 🧠 Explanation"
                    )

                    st.markdown(
                        st.session_state[
                            "selected_concept_explanation"
                        ]
                    )
    
    
    # ==========================================
    # PDF COMPARISON
    # ==========================================
 
    if (
        st.session_state.get(
            "compare_pdfs",
            False
        )
        and len(
            st.session_state.get(
                "selected_pdfs",
                []
            )
        ) == 2
    ):

        pdf1 = (
            st.session_state.selected_pdfs[0]
        )

        pdf2 = (
            st.session_state.selected_pdfs[1]
        )
        
        similarity = calculate_pdf_similarity(
            pdf1,
            pdf2
        )

        st.markdown(
            "## 🔍 PDF Comparison"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                f"📄 {pdf1}"
            )

        with col2:

            st.info(
                f"📄 {pdf2}"
            )

        if st.button(
            "🚀 Start Comparison",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing both PDFs..."
            ):

                try:
                    
                    if similarity is not None:
                        st.metric(
                            "📊 Document Similarity",
                            f"{similarity}%"
                        )
                        

                    documents1 = (
                        get_document_context(
                            pdf1
                        )
                    )

                    documents2 = (
                        get_document_context(
                            pdf2
                        )
                    )

                    if not documents1:

                        st.warning(
                            f"No content found in {pdf1}."
                        )

                        st.stop()

                    if not documents2:

                        st.warning(
                            f"No content found in {pdf2}."
                        )

                        st.stop()

                    context1 = "\n\n".join(
                        documents1
                    )

                    context2 = "\n\n".join(
                        documents2
                    )

                    MAX_COMPARE_CHARS = 12000

                    context1 = context1[
                        :MAX_COMPARE_CHARS
                    ]

                    context2 = context2[
                        :MAX_COMPARE_CHARS
                    ]

                    comparison = generate_pdf_comparison(
                        pdf1,
                        context1,
                        pdf2,
                        context2
                    )


                    if comparison:

                        st.session_state.pdf_comparison = (
                            comparison.strip()
                        )

                    else:

                        st.warning(
                            "Could not generate comparison."
                        )

                except RuntimeError as e:

                    st.error(
                        f"⚠️ {str(e)}"
                    )

                except Exception as e:

                    st.error(
                        f"⚠️ Comparison failed: {str(e)}"
                    )
    
    if st.session_state.get(
        "pdf_comparison"
    ):

        with st.expander(
            "📊 Comparison Result",
            expanded=True
        ):

            st.markdown(
                st.session_state.pdf_comparison
            )
        
    # ==========================================
    # IMPORT QUESTIONS
    # ==========================================
    
    if st.session_state.get(
        "important_questions"
    ):
        with st.expander(
            "💡 Important Exam Questions",
            expanded=True
        ):

            st.markdown(
                st.session_state.important_questions
            )
    

    # ==========================================
    # SIMPLE EXPLANATION
    # ==========================================

    if "simple_explanation" in st.session_state:

        with st.expander(
            "🧠 Simple Explanation",
            expanded=True
        ):

            st.markdown(
                st.session_state.simple_explanation
            )

    # ==========================================
    # INTERACTIVE QUIZ
    # ==========================================

    if (
        "quiz" in st.session_state
        and isinstance(
            st.session_state.quiz,
            list
        )
    ):

        with st.expander(
            "❓ AI Quiz",
            expanded=True
        ):

            quiz = st.session_state.quiz

            user_answers = {}

            for i, item in enumerate(quiz):

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                st.markdown(
                    f"### Question {i + 1}"
                )

                st.write(
                    item["question"]
                )

                options = item.get(
                    "options",
                    []
                )

                if not options:
                    continue

                selected = st.radio(
                    "Choose an answer:",
                    options,
                    key=f"quiz_answer_{i}"
                )

                user_answers[i] = selected

            if st.button(
                "✅ Submit Quiz",
                use_container_width=True
            ):

                score = 0
                total = len(quiz)

                st.session_state.quiz_result = {}

                for i, item in enumerate(quiz):

                    if not isinstance(
                        item,
                        dict
                    ):
                        continue

                    options = item.get(
                        "options",
                        []
                    )

                    correct_index = item.get(
                        "answer",
                        0
                    )

                    if (
                        correct_index
                        >= len(options)
                    ):
                        continue

                    correct_answer = (
                        options[correct_index]
                    )

                    selected_answer = (
                        user_answers.get(i)
                    )

                    if (
                        selected_answer
                        == correct_answer
                    ):

                        score += 1

                        st.session_state.quiz_result[
                            i
                        ] = True

                    else:

                        st.session_state.quiz_result[
                            i
                        ] = False

                percentage = int(
                    (score / total) * 100
                ) if total else 0

                st.divider()

                st.subheader(
                    "📊 Quiz Result"
                )

                st.metric(
                    "Score",
                    f"{score}/{total}"
                )

                st.write(
                    f"**Percentage: {percentage}%**"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.success(
                        f"✅ Correct: {score}"
                    )

                with col2:

                    st.error(
                        f"❌ Incorrect: "
                        f"{total - score}"
                    )

                st.divider()

                st.subheader(
                    "📖 Answer Review"
                )

                for i, item in enumerate(quiz):

                    if not isinstance(
                        item,
                        dict
                    ):
                        continue

                    options = item.get(
                        "options",
                        []
                    )

                    correct_index = item.get(
                        "answer",
                        0
                    )

                    if (
                        correct_index
                        >= len(options)
                    ):
                        continue

                    correct_answer = (
                        options[correct_index]
                    )

                    if st.session_state.quiz_result.get(
                        i
                    ):

                        st.success(
                            f"Question {i + 1}: "
                            f"Correct ✅"
                        )

                    else:

                        st.error(
                            f"Question {i + 1}: "
                            f"Incorrect ❌"
                        )

                        st.write(
                            f"Correct answer: "
                            f"**{correct_answer}**"
                        )

                    explanation = item.get(
                        "explanation",
                        ""
                    )

                    if explanation:

                        st.caption(
                            explanation
                        )

                if st.button(
                    "🔄 Try Again",
                    use_container_width=True
                ):

                    st.session_state.pop(
                        "quiz",
                        None
                    )

                    st.session_state.pop(
                        "quiz_result",
                        None
                    )

                    st.rerun()



# else:

#     st.title("🤖 AI PDF Assistant")

#     st.info(
#         "Upload and process a PDF from the sidebar "
#         "to start chatting."
#     )



# ==========================================
# CHAT SEARCH RESULTS
# ==========================================

if chat_search:

    search_term = (
        chat_search
        .strip()
        .lower()
    )

    matched_messages = []

    for i, message in enumerate(messages):

        if search_term in message.get(
            "content",
            ""
        ).lower():

            matched_messages.append(
                (
                    i,
                    message
                )
            )

    if matched_messages:

        with st.expander(
            f"🔎 Search Results "
            f"({len(matched_messages)})",
            expanded=True
        ):

            for i, message in matched_messages:

                role = message.get(
                    "role",
                    "unknown"
                )

                if role == "user":

                    st.markdown(
                        "👤 **You**"
                    )

                else:

                    st.markdown(
                        "🤖 **AI Assistant**"
                    )

                st.write(
                    message.get(
                        "content",
                        ""
                    )
                )

                st.divider()

    else:

        st.info(
            "No matching messages found."
        )
    

    
# ==================================================
# CHAT HISTORY DISPLAY
# ==================================================
for i, message in enumerate(messages):

    with st.chat_message(
        message["role"]
    ):

        # ------------------------------------------
        # MESSAGE CONTENT
        # ------------------------------------------

        st.markdown(
            message["content"]
        )

        if message.get("timestamp"):

            st.caption(
                message["timestamp"]
            )
            
        
        if message["role"] == "assistant":
            # ------------------------------------------
            # COMPACT ASSISTANT ACTIONS
            # ------------------------------------------

            if (
                message["role"] == "assistant"
                and i > 0
                and messages[i - 1]["role"] == "user"
            ):

                previous_question = (
                    messages[i - 1]["content"]
                )

                chat_pdf_key = "|".join(
                    sorted(
                        st.session_state.get(
                            "selected_pdfs",
                            [selected_pdf]
                        )
                    )
                )

                save_id = (
                    f"{chat_pdf_key}|"
                    f"{previous_question}|"
                    f"{message['content']}"
                )

                saved_ids = [
                    item.get("id")
                    for item in st.session_state.saved_answers
                ]

                # --------------------------------------
                # SMALL ACTION BUTTONS
                # --------------------------------------

                col1, col2, col3, col4, col5 = st.columns(
                    [16,0.7, .7, .7, .7,]
                )
            
                
                
                with col1:
                    # st.button(
                    #     "",
                    #     key=f"copy_{i}",
                    #     help="Copy answer",
                    #     icon=":material/content_copy:",
                    #     type="tertiary"
                    # )
                    copy_button(
                        message["content"],
                        # tooltip="Copy response",
                        copied_label="Copied!",
                        key=f"copy_{i}"
                    )
                
                    

                # SAVE / REMOVE
                with col5:
                    if save_id in saved_ids:
                        if st.button(
                            "",
                            key=f"unsave_{i}",
                            help="Remove saved answer",
                            icon=":material/bookmark:",
                            type="tertiary"
                        ):

                            st.session_state.saved_answers = [
                                item
                                for item in st.session_state.saved_answers
                                if item.get("id") != save_id
                            ]

                            st.rerun()

                    else:

                        if st.button(
                            "",
                            key=f"save_{i}",
                            help="Save answer",
                            icon=":material/bookmark_border:",
                            type="tertiary"
                        ):

                            st.session_state.saved_answers.append(
                                {
                                    "id": save_id,
                                    "question": previous_question,
                                    "answer": message["content"],
                                    "pdf": chat_pdf_key
                                }
                            )

                            st.rerun()

                # REGENERATE
                with col4:
                    if st.button(
                        "",
                        key=f"regenerate_{i}",
                        help="Regenerate answer",
                        icon=":material/refresh:",
                        type="tertiary"
                    ):

                        st.session_state.regenerate_question = (
                            previous_question
                        )

                        st.session_state.regenerate_index = i

                        st.rerun()

                # HELPFUL
                with col2:
                    if st.button(
                        "",
                        key=f"helpful_{i}",
                        help="Helpful",
                        icon=":material/thumb_up:",
                        type="tertiary"
                    ):

                        st.session_state.answer_feedback[i] = (
                            "helpful"
                        )

                        st.toast(
                            "Thanks for your feedback!"
                        )
                
                # NOT HELPFUL
                with col3:
                    if st.button(
                        "",
                        key=f"not_helpful_{i}",
                        help="Not helpful",
                        icon=":material/thumb_down:",
                        type="tertiary"
                    ):

                        st.session_state.answer_feedback[i] = (
                            "not_helpful"
                        )

                        st.toast(
                            "Thanks for your feedback."
                        )

# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask a question about your PDF..."
)


# ==================================================
# PENDING SUGGESTED QUESTION
# ==================================================

if st.session_state.get(
    "pending_question"
):

    question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None
    

# ==================================================
# REGENERATE QUESTIONS
# ==================================================
if (
    "regenerate_question"
    in st.session_state
):

    question = (
        st.session_state.regenerate_question
    )

    del st.session_state.regenerate_question


# ==================================================
# QUESTION PROCESSING
# ==================================================

if question:

    # ==========================================
    # SHOW USER MESSAGE
    # ==========================================

    with st.chat_message("user"):

        st.markdown(question)


    # ==========================================
    # COPY OLD CHAT HISTORY
    # ==========================================

    chat_history = messages.copy()


    # ==========================================
    # RETRIEVE RELEVANT CHUNKS
    # ==========================================

    with st.spinner(
        "Searching the PDF..."
    ):

        try:

            if (
                is_follow_up(question)
                and chat_history
            ):

                previous_questions = []

                for message in chat_history[-4:]:

                    if (
                        message["role"]
                        == "user"
                    ):

                        previous_questions.append(
                            message["content"]
                        )

                standalone_question = (
                    "Previous user questions: "
                    + " ".join(
                        previous_questions
                    )
                    + "\nCurrent question: "
                    + question
                )

            else:

                standalone_question = question

            # ==================================
            # SEARCH CHROMADB
            # ==================================

            optimization = (
                get_ai_optimization_settings()
            )

            documents, metadatas, distances = (
                retrieve_from_multiple_pdfs(
                    standalone_question,
                    st.session_state.selected_pdfs,
                    n_results=optimization["max_chunks"]
                )
            )


        except Exception as e:

            st.error(
                "⚠️ I could not search the "
                "selected PDF. Please try again."
            )

            st.stop()


    # ==========================================
    # CHECK RETRIEVAL
    # ==========================================

    if not documents:

        st.warning(
            "I could not find relevant information "
            "in the selected PDF."
        )

        st.stop()


    # ==========================================
    # CREATE CONTEXT
    # ==========================================

    context = "\n\n".join(
        documents
    )

    MAX_CONTEXT_CHARS = (
        optimization["context_limit"]
    )

    if len(context) > MAX_CONTEXT_CHARS:

        context = context[
            :MAX_CONTEXT_CHARS
        ]


    # ==========================================
    # DEBUG: RETRIEVED CONTEXT
    # ==========================================

    with st.expander(
        "🔍 Retrieved PDF Context"
    ):

        for i, (
            document,
            metadata,
            distance
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances
            )
        ):

            st.write(
                f"### Chunk {i + 1}"
            )

            st.write(
                f"**File:** "
                f"{metadata.get('filename', 'Unknown')}"
            )

            st.write(
                f"**Page:** "
                f"{metadata.get('page', 'Unknown')}"
            )

            st.write(
                f"**Section:** "
                f"{metadata.get('section', 'General')}"
            )

            st.write(
                f"**Vector Distance:** "
                f"{distance:.4f}"
            )

            st.write(document)

            st.divider()


    # ==========================================
    # CONTEXT SENT TO GEMINI
    # ==========================================

    with st.expander(
        "🧠 Context Sent to Gemini"
    ):

        st.write(context)


    # ==========================================
    # GENERATE ANSWER
    # ==========================================

    with st.spinner(
        "Generating answer..."
    ):
        try:

            answer = generate_answer(
                question,
                context,
                chat_history,
                st.session_state.response_language
            )

        except RuntimeError as e:

            error_message = str(e)

            st.warning(
                f"⚠️ AI service unavailable.\n\n"
                f"{error_message}"
            )
            
            st.info(
                "📴 Offline AI Mode: "
                "Using your PDF content without Gemini."
            )

            answer = local_pdf_fallback(
                question,
                documents,
                metadatas,
                distances,
                st.session_state.response_language
            )


    # ==========================================
    # SHOW AI RESPONSE
    # ==========================================

    with st.chat_message("assistant"):
        st.markdown(answer)

        save_id = (
            f"{selected_pdf}|"
            f"{question}|"
            f"{answer}"
        )

        saved_ids = [
            item.get("id")
            for item in st.session_state.saved_answers
        ]

        if save_id in saved_ids:

            if st.button(
                "🗑️ Remove Saved Answer",
                key=f"remove_{hash(save_id)}"
            ):

                st.session_state.saved_answers = [
                    item
                    for item in st.session_state.saved_answers
                    if item.get("id") != save_id
                ]

                st.rerun()

        else:
            
            if st.button(
                "",
                key=f"save_{hash(save_id)}",
                help="Save answer",
                icon=":material/bookmark_border:",
                type="tertiary"
            ):

                st.session_state.saved_answers.append(
                    {
                        "id": save_id,
                        "question": question,
                        "answer": answer,
                        "pdf": selected_pdf
                    }
                )

                st.rerun()


    # ==========================================
    # SAVE CONVERSATION
    # ==========================================

    regenerate_index = st.session_state.pop(
        "regenerate_index",
        None
    )

    if regenerate_index is not None:

        # Remove old user + assistant messages
        if (
            regenerate_index > 0
            and regenerate_index < len(messages)
            and messages[regenerate_index]["role"] == "assistant"
        ):

            messages.pop(
                regenerate_index
            )

            messages.pop(
                regenerate_index - 1
            )

    # Save new conversation

    messages.append(
        {
            "role": "user",
            "content": question,
            "timestamp": datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            )
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            )
        }
    )


    # ==========================================
    # SOURCES
    # ==========================================

    if metadatas:

        shown_sources = set()

        for metadata in metadatas:

            filename = metadata.get(
                "filename",
                "Unknown"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            section = metadata.get(
                "section",
                "General"
            )

            source = (
                filename,
                page,
                section
            )

            shown_sources.add(
                source
            )

        st.subheader(
            f"📚 Sources ({len(shown_sources)})"
        )

        # Group sources by PDF
        sources_by_pdf = {}

        for (
            filename,
            page,
            section
        ) in shown_sources:

            if filename not in sources_by_pdf:

                sources_by_pdf[filename] = []

            sources_by_pdf[filename].append(
                (
                    page,
                    section
                )
            )

        # Display PDF-wise
        for filename, sources in (
            sources_by_pdf.items()
        ):

            with st.expander(
                f"📄 {filename}",
                expanded=False
            ):

                for page, section in sources:

                    st.markdown(
                        f"""
                        **Page:** {page}

                        **Section:** {section}
                        """
                    )

                    st.divider()

    else:

        st.info(
            "No source information available."
        ) 