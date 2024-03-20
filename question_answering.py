# Import or define RecursiveCharacterTextSplitter and OpenAIEmbeddings here
# ...

# Rest of your imports
from fastapi import FastAPI, File, UploadFile, Form,HTTPException
from PyPDF2 import PdfReader
from fastapi.responses import JSONResponse
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import aioredis
import faiss
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"]=""
app = FastAPI()

origins = [
    "http://localhost:3000",  # Assuming your React app runs on localhost:3000
    "https://your-react-app-domain.com"  # Add your production domain if applicable
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

redis_pool: aioredis.Redis = None

@app.on_event("startup")
async def startup_event():
    global redis_pool
    # Initialize the Redis connection pool
    redis_pool = await aioredis.from_url("redis://localhost", encoding="utf-8")

@app.on_event("shutdown")
async def shutdown_event():
    global redis_pool
    await redis_pool.close()

async def get_cached_data(key: str):
    data = await redis_pool.get(key)
    print(data)
    print(type(data))
    return data.decode('utf-8') if data else None


async def set_cache_data(key: str, value):
    await redis_pool.set(key, json.dumps(value))


def read_pdf(pdf_file):
    pdfreader = PdfReader(pdf_file)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

class Question(BaseModel):
    message: str
# @app.post("/ask-question/")
@app.post("/ask-question/")
async def ask_question(question: Question):
    pdf_files="real_estate_doc.pdf"
    if not pdf_files:
        return JSONResponse(content={"error": "Please upload at least one PDF file"}, status_code=400)
    else:
        cache_key = f"{question.message}"
        print(cache_key)
        cached_ans = await get_cached_data(cache_key)
        if cached_ans:
            print("Present in cache")
            return {'ans':cached_ans}
        else:
            print("Generating answer from doc")
            all_text = read_pdf(pdf_files)
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
            texts = text_splitter.split_text(all_text)
            embeddings = OpenAIEmbeddings()
            document_search = FAISS.from_texts(texts, embeddings)        
            chain = load_qa_chain(OpenAI(), chain_type="stuff")  # Replace 'appropriate_type' with actual chain type
            docs = document_search.similarity_search(question.message)
            final_prompt = (
        "1. You are a realtor with decades of experience\n"
        "2. The text enclosed in curly brackets is a question asked by a user in a chatbot\n"
        "3. Response must contain only the reply to the text enclosed in curly brackets\n"
        "4. The replies must be in a friendly and conversational tone\n"
        "5. Do not respond with a description of yourself\n"
        "6. The replies must not be too long\n"
        f"{question.message}"
    )
            answer=chain.run(input_documents=docs, question=final_prompt)
            await set_cache_data(question.message,answer)

        return {'ans':answer}
  

# redis_pool: aioredis.Redis = None

# class Question(BaseModel):
#     message: str



# @app.post("/ask-question/")
# async def ask_question(question: Question):
#     pdf_files = "real_estate_doc.pdf"
#     if not pdf_files:
#         return JSONResponse(content={"error": "Please upload at least one PDF file"}, status_code=400)
    
#     # Attempt to fetch cached document search results

    
#     if not docs:
#         all_text = read_pdf(pdf_files)
#         text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
#         texts = text_splitter.split_text(all_text)
#         embeddings = OpenAIEmbeddings()
#         document_search = FAISS.from_texts(texts, embeddings)
#         docs = document_search.similarity_search(question.message)
#         # Cache the document search results
#         await cache_data(cache_key, docs)

#     chain = load_qa_chain(OpenAI(), chain_type="stuff")  # Placeholder for actual chain type
#     final_prompt = (
#         "1. You are a realtor with decades of experience\n"
#         "2. The text enclosed in curly brackets is a question asked by a user in a chatbot\n"
#         "3. Response must contain only the reply to the text enclosed in curly brackets\n"
#         "4. The replies must be in a friendly and conversational tone\n"
#         "5. Do not respond with a description of yourself\n"
#         "6. The replies must not be too long\n"
#         f"{question.message}"
#     )
#     answer = chain.run(input_documents=docs, question=final_prompt)

#     return {'ans': answer}