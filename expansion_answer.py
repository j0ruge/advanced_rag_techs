from helper_utils import word_wrap;
from pypdf import PdfReader;
import os;
from openai import OpenAI;
from dotenv import load_dotenv;

# Load the environment variables from the .env file
load_dotenv();

# Get the API key from the environment variables
openai_key = os.getenv("OPENAI_API_KEY");
client = OpenAI(api_key=openai_key);

# print(openai_key);

reader = PdfReader("data/microsoft-annual-report.pdf");
pdf_texts = [page.extract_text().strip() for page in reader.pages];

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text];

#print(
#     word_wrap(
#         pdf_texts[0],
#         width=100,
#     )
# );

# split the text into smaller chunks
# to avoid the 4096 token limit

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
);

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
);

character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts));

# print(word_wrap(character_split_texts[10]));
# print(f"\nTotal chunks: {len(character_split_texts)}");

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
);

token_split_texts = [];

for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text);

# print(word_wrap(character_split_texts[10]));
# print(f"\nTotal chunks: {len(character_split_texts)}");


# Embedding the text chunks
import chromadb;
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction;

embedding_function = SentenceTransformerEmbeddingFunction();
# print(embedding_function([token_split_texts[10]]));


# Chrmomadb client

chroma_client = chromadb.Client();

chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
);

# extract the embeddings of the token_split_texts

ids = [str(index) for index in range(len(token_split_texts))];

chroma_collection.add(ids=ids, documents=token_split_texts);
count = chroma_collection.count();

query = "What was the total revenue for the year?"

results = chroma_collection.query(query_texts=[query], n_results=5);
retrieved_documents = results["documents"][0];

# for document in retrieved_documents:
#     print(word_wrap(document));
#     print("\n");


# Models
# gpt-4o-mini
# gpt-3.5-turbo


def augment_query_generated(query, model="gpt-4o-mini"):
    prompt = """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {   
            "role": "user",
            "content": query
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    );
    content = response.choices[0].message.content;
    return content;

original_query = "What was the total profit for the year, and how does it compare to the previous year?";
hypothetical_answer = augment_query_generated(original_query);

joint_query = f"{original_query} {hypothetical_answer}";
# print(word_wrap(joint_query));

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0];

for document in retrieved_documents:
    print(word_wrap(document));
    print("\n");