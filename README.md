## AI Tutoring System using Language Models

This repository contains the code for the AI Tutoring System using Language Models. The system is designed to help students with their homework and assignments by providing them with explanations and solutions to their problems. The system uses state-of-the-art language models to understand and generate natural language text, and it can be used to answer questions, solve problems, and provide explanations for a wide range of subjects and topics.

Important Note : if You are a Windows user after installing Langchain you have manually do the following steps to make it work.

1. Go to ``` Your_Env\lanchain_community\document_loaders\pebblo.py```
2. Replace the following code
```python
    "From This"
        try:
            file_owner_uid = os.stat(file_path).st_uid
            file_owner_name = pwd.getpwuid(file_owner_uid).pw_name
        except Exception:
            file_owner_name = "unknown"
        return file_owner_name

    "To This"
        try:
            import pwd
            file_owner_uid = os.stat(file_path).st_uid
            file_owner_name = pwd.getpwuid(file_owner_uid).pw_name
        except Exception:
            file_owner_name = "unknown"
        return file_owner_name
    
    "Also remove the import pwd from the top of the file"
```
    

### Download and Store the Model in the `models` folder
<b><span>Model Used: </span><a href="https://huggingface.co/MBZUAI/LaMini-Flan-T5-783M">LaMini-T5-Flan</a></b>

<b>Authors: [Ajay](https://github.com/ShadowConYT)</b>