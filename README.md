## AI Tutoring System Backend for the Unity 3d Model with Audio

### Systems Implemented
- PDF Parser
- Large Language Model
- Audio Processing
***
### Base Idea
The Base Idea of this implementation is using RAG to process the PDF and then using the output to generate the audio and the 3d model. The 3d model will be generated using Unity and the audio will be generated using the Large Language Model.
***
### Installation
- Install the requirements using the following command
```bash
pip install -r requirements.txt
```
- Download and save the Large Language Model from the following link
- Model Used : [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
***
### Prerequisites
- Anaconda or MiniConda
- Python 3.8 or above
- HuggingFace API Key
- Access to Meta llama 2
- Elevenlabs API Key

**Note** : The Elevenlabs API Key is required for the audio processing part of the system.

**Save the API Keys in the .env file**

#### Author : [ShadowCon](https://github.com/ShadowConYT)

***

**Note** : The system is still under development and the documentation will be updated as the system is developed further. If You want to practice with the model you can access it using the jupyter file in the ` testing folder `.