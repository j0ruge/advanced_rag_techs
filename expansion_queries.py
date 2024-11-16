from helper_utils import project_embeddings, word_wrap;
from pypdf import PdfReader;
import os;
from openai import OpenAI;
from dotenv import load_dotenv;
import umap;

# Load the environment variables from the .env file
load_dotenv();

# Get the API key from the environment variables
openai_key = os.getenv("OPENAI_API_KEY");
client = OpenAI(api_key=openai_key);

# Extract the text from the PDF
reader = PdfReader("data/microsoft-annual-report.pdf");
pdf_texts = [page.extract_text().strip() for page in reader.pages];

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text];

# Split the text into smaller chunks
# to avoid the 4096 token limit

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
);

# Split the text into smaller chunks

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
);

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
);

character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts));

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# now we import chromadb and the SentenceTransformerEmbeddingFunction
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction();

# we then instantiate the Chroma client and create a collection called "microsoft-collection"
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))];

chroma_collection.add(ids=ids, documents=token_split_texts);
count = chroma_collection.count();

print(f"Added {count} documents to the collection");


query = "What was the total revenue for the year?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]
