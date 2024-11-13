from helper_utils import word_wrap;
from pypdf import PdfReader;
import os;
from openai import OpenAI;
from dotenv import load_dotenv;

# Load the environment variables from the .env file
load_dotenv();

# Get the API key from the environment variables
openai_key = os.getenv("OPENAI_API_KEY");

print(openai_key);

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

print(word_wrap(character_split_texts[10]));

print(f"Number of chunks: {len(character_split_texts)}");