from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import pipeline
import streamlit as st
import base64

# use cpu

# Load the summarization chain
checkpoint = 'model'
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, \
                                                       device_map = 'cpu', \
                                                       torch_dtype = torch.float32)


def pdf_preprocessing(file):
    pdf_loader = PyPDFLoader(file)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final = ""
    for text in texts:
        print(text)
        final += text.page_content
    return final

def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=300,
        min_length=30,
    )

    input_text = pdf_preprocessing(filepath)
    summary = pipe_sum(input_text)
    result = summary[0]['summary_text']
    return result

print(llm_pipeline('python_tutorial.pdf'))