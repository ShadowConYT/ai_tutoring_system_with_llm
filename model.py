from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
print("VectorStoreIndex,SimpleDirectoryReader,ServiceContext imported")
from llama_index.llms.huggingface import HuggingFaceLLM
print("HuggingFaceLLM imported")
from llama_index.core.prompts.prompts import SimpleInputPrompt
print("SimpleInputPrompt imported")
from ctransformers import  AutoModelForCausalLM
print("AutoModelForCausalLM imported")
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
print("HuggingFaceEmbeddings imported")
from llama_index.core import ServiceContext
print("ServiceContext imported")
from llama_index.embeddings.langchain import LangchainEmbedding
print("LangchainEmbedding imported")
from langchain_community.document_loaders import PyPDFLoader
print("PyPDFLoader imported")
import json
import torch
import os
from dotenv import load_dotenv
load_dotenv()

HuggingFace_Api = os.environ.get('HF_TOKEN')

documents = SimpleDirectoryReader('./testing/docs').load_data()
print("SimpleDirectoryReader imported")

def get_system_prompt():
    '''This function is used to load the system prompt from the prompts.json file'''

    with open('prompts.json') as f:
        data = json.load(f)
    return data['Default']

query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

def load_model(context_window: int, max_new_tokens: int):
    '''This function is used to load the model from the HuggingFaceLLM'''

    print(f"""Available Cuda: {torch.cuda.get_device_name()} \n
            Trying to load the model model""")
    try:
        llm = HuggingFaceLLM(context_window=context_window,
                            max_new_tokens=max_new_tokens,
                            generate_kwargs={"temperature": 0.0, "do_sample": False},
                            system_prompt=get_system_prompt(),
                            query_wrapper_prompt=query_wrapper_prompt,
                            tokenizer_name="./meta",
                            model_name="./meta",
                            device_map="cuda",
                            # uncomment this if using CUDA to reduce memory usage
                            model_kwargs={"torch_dtype": torch.float16,"load_in_8bit":True }
                        )
        print("Model Loaded")
        return llm
    except Exception as e:
        print(f"Error: {e}")
        return None

def embed_model():
    '''This function is used to load the model from the LangchainEmbedding'''

    embed = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

    service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=load_model(context_window=4096, max_new_tokens=256),
    embed_model=embed
    )
    return service_context

def get_index():
    '''This function is used to load the index from the VectorStoreIndex'''

    index=VectorStoreIndex.from_documents(documents,service_context=embed_model())
    return index

def main():
    query_engine=get_index().as_query_engine()
    response=query_engine.query("what is this PDF tells about?")
    out = response
    print(response)

if __name__ == "__main__":
    main()