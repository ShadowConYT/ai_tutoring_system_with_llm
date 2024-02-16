from langchain.text_splitter import TextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import pipeline
import streamlit as st
import base64

# Load the summarization chain
checkpoint = 'model'
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, \
                                                       device_map = 'auto', \
                                                       torch_dtype = torch.float32)


test_model = pipeline('text2text-generation', model=base_model, tokenizer=tokenizer)
input_prompt ="what is love?"
result = test_model(input_prompt, max_length=50, num_return_sequences=1)
print(result)