import gradio as gr
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
load_dotenv()
import torch
from ctransformers import AutoModelForCausalLM
import pyttsx3 as tts

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def load_doc(pdf_doc):

    loader = PyMuPDFLoader(pdf_doc.name)
    documents = loader.load()
    embedding = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents(documents)
    db = Chroma.from_documents(text, embedding)
    #llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b", model_kwargs={"temperature": 1.0, "max_length": 1250})
    # load my local model from directory
    #tokenizer = AutoTokenizer.from_pretrained("bloom")
    config = {
    'max_new_tokens': 256,
    'repetition_penalty': 1.1,
    'temperature': 0.7,
    'stream': True
    }
    model = AutoModelForCausalLM.from_pretrained("bloke/",
                                                model_file='llama-2-7b-chat.ggmlv3.q4_K_M.bin', 
                                                model_type="llama",
                                                **config)


    global chain
    #chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    chain = RetrievalQA.from_chain_type(llm=model,chain_type="stuff",retriever=db.as_retriever())
    return 'Document has successfully been loaded'

def text_to_speech(text):
    engine = tts.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

def answer_query(query):
    question = query
    res = chain.run(question)
    res = res.split("Question:")[1].split("Helpful Answer:")[1].strip()
    text_to_speech(res)
    return res
html = """
<div style="text-align:center; max width: 700px;">
    <h1>Chat With Your PDF</h1>
    <p> Upload a PDF File, then click on Load PDF File <br>
    Once the document has been loaded you can begin chatting with the PDF =)
</div>"""
css = """container{max-width:700px; margin-left:auto; margin-right:auto,padding:20px}"""
with gr.Blocks(css=css,theme=gr.themes.Monochrome()) as demo:
    gr.HTML(html)
    with gr.Column():
        gr.Markdown('ChatPDF')
        pdf_doc = gr.File(label="Load a pdf",file_types=['.pdf','.docx'],type='filepath')
        with gr.Row():
            load_pdf = gr.Button('Load pdf file')
            status = gr.Textbox(label="Status",placeholder='',interactive=False)


        with gr.Row():
            input = gr.Textbox(label="type in your question")
            output = gr.Textbox(label="output")

        submit_query = gr.Button("submit")

        load_pdf.click(load_doc,inputs=pdf_doc,outputs=status)

        submit_query.click(answer_query,input,output)

demo.launch()