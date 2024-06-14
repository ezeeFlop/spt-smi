import asyncio
import logging
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO
from typing import List, Optional
from pymilvus import Collection, connections, DataType, FieldSchema, CollectionSchema
from pydantic import BaseModel, validator
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)


class VectorDocument(BaseModel):
    text: str
    vector: Optional[List[float]] = None

    @validator('vector', always=True)
    def validate_vector(cls, v, values, **kwargs):
        if v is not None and len(v) != 768:
            raise ValueError('Vector must be of length 768')
        return v


class SearchParams(BaseModel):
    metric_type: str = "IP"
    params: dict = {}


class RAG:
    def __init__(self, model, collection_name: str):
        """Initialize the RAG class with a specific embedding model and Milvus collection name."""
        self.model = model
        self.collection_name = collection_name
        self.collection = None
        asyncio.run(self.connect_to_milvus())

    async def connect_to_milvus(self):
        """Asynchronous connection to Milvus database."""
        try:
            connections.connect("default", host='localhost', port='19530')
            await asyncio.to_thread(self.setup_collection)
        except Exception as e:
            logging.error("Failed to connect to Milvus: %s", e)

    def setup_collection(self):
        """Setup the collection schema for storing document vectors in Milvus."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        schema = CollectionSchema(
            fields, description="Collection for storing document vectors")
        self.collection = Collection(name=self.collection_name, schema=schema)
        self.collection.load()

    async def vectorize_documents(self, documents: List[VectorDocument]):
        """Vectorize a list of documents using the specified embedding model."""
        texts = [doc.text for doc in documents]
        try:
            vectors = await asyncio.to_thread(self.model.encode, texts)
            for doc, vector in zip(documents, vectors):
                doc.vector = vector.tolist()
        except Exception as e:
            logging.error("Failed to vectorize documents: %s", e)

    async def add_documents_to_collection(self, documents: List[VectorDocument]):
        """Add a list of vectorized documents to the Milvus collection."""
        try:
            vectors = [
                doc.vector for doc in documents if doc.vector is not None]
            if vectors:
                await asyncio.to_thread(self.collection.insert, [vectors])
        except Exception as e:
            logging.error("Failed to add documents to Milvus: %s", e)

    async def search_documents(self, query: str, search_params: SearchParams, top_k: int = 5):
        """Search for the most similar documents to a given query."""
        try:
            query_vector = await asyncio.to_thread(self.model.encode, [query])
            results = await asyncio.to_thread(
                self.collection.search,
                [query_vector[0].tolist()],
                "embedding",
                search_params.dict(),
                limit=top_k
            )
            return results
        except Exception as e:
            logging.error("Failed to search documents: %s", e)
            return []

    @staticmethod
    async def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
        """Extract text from a PDF file provided as bytes."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            logging.error("Failed to extract text from PDF bytes: %s", e)
            return ""

    @staticmethod
    async def extract_text_from_image_bytes(image_bytes: bytes) -> str:
        """Extract text from an image file provided as bytes using OCR."""
        try:
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logging.error("Failed to extract text from image bytes: %s", e)
            return ""

    async def vectorize_from_pdf_bytes(self, pdf_bytes: bytes) -> VectorDocument:
        """Extract text from a PDF file in bytes and return a vectorized document."""
        text = await self.extract_text_from_pdf_bytes(pdf_bytes)
        try:
            vector = await asyncio.to_thread(self.model.encode, [text])[0].tolist()
            return VectorDocument(text=text, vector=vector)
        except Exception as e:
            logging.error("Failed to vectorize PDF bytes: %s", e)
            return VectorDocument(text="")

    async def vectorize_from_image_bytes(self, image_bytes: bytes) -> VectorDocument:
        """Extract text from an image file in bytes and return a vectorized document."""
        text = await self.extract_text_from_image_bytes(image_bytes)
        try:
            vector = await asyncio.to_thread(self.model.encode, [text])[0].tolist()
            return VectorDocument(text=text, vector=vector)
        except Exception as e:
            logging.error("Failed to vectorize image bytes: %s", e)
            return VectorDocument(text="")


async def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    rag = RAG(model, 'milvus_rag')

    # Example PDF and image data as bytes
    pdf_bytes = open('/path/to/your/file.pdf', 'rb').read()
    image_bytes = open('/path/to/your/image.png', 'rb').read()

    # Vectorize PDF and image bytes
    pdf_doc = await rag.vectorize_from_pdf_bytes(pdf_bytes)
    image_doc = await rag.vectorize_from_image_bytes(image_bytes)

    # Add to Milvus collection
    await rag.add_documents_to_collection([pdf_doc, image_doc])

    # Search with custom parameters
    search_params = SearchParams(metric_type="IP", params={"nprobe": 10})
    results = await rag.search_documents("What is Milvus?", search_params)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
