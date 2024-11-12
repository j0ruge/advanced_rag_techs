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