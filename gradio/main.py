import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_chat import ChatModel
from embedding_pipeline import setup_rag_pipeline
import torch

if not torch.cuda.is_available():
    print("Warning: CUDA GPU is not available. The model will run on CPU, which may be slow.")
else:
    print(f"CUDA GPU is available: {torch.cuda.get_device_name(0)}")

model_name = "XiaomiMiMo/MiMo-7B-RL-0530"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

EMBEDDING_MODEL, FAISS_INDEX, CHUNK_STORE = setup_rag_pipeline()

chat_model = ChatModel(model=model, tokenizer=tokenizer, max_length=2048, embedding_model=EMBEDDING_MODEL, faiss_index=FAISS_INDEX, chunk_store=CHUNK_STORE, faiss_k=5)

iface = gr.ChatInterface(
    fn=chat_model.chat,
    title="Film chatbot",
    chatbot=gr.Chatbot(),
    type="messages",
    description="An open-source chatbot powered for film recomendations.",
    examples=[["Can you solve this math problem for me? Find the value of x in the equation 2x + 5 = 15."], ["Write a python function to find the nth Fibonacci number."]]
)

iface.launch(share=True)