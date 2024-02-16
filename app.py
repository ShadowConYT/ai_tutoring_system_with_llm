from langchain.textsplitter import TextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import pipeline
import streamlit as st
import base64

# Load the summarization chain