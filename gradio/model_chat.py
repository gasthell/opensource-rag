import numpy as np

class ChatModel:
    def __init__(self, model, tokenizer, max_length=2048, embedding_model=None, faiss_index=None, chunk_store=None, faiss_k=5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.chunk_store = chunk_store
        self.faiss_k = faiss_k

    def chat(self, message, history):
        conversation = []
        for user, assistant in history:
            conversation.append({"role": "user", "content": user})
            conversation.append({"role": "assistant", "content": assistant})
        conversation.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return response
    
    def embedding_chat(self, message, history):
        query_embedding = self.embedding_model.encode([message])
        k = self.faiss_k
        distances, indices = self.faiss_index.search(np.array(query_embedding), k)
        
        retrieved_chunks = [self.chunk_store[i] for i in indices[0]]
        
        context = ""
        for chunk in retrieved_chunks:
            context += f"Source: {chunk['source']}\nContent: {chunk['content']}\n\n"

        conversation = []
        for user_turn, bot_turn in history:
            conversation.append({"role": "user", "content": user_turn})
            conversation.append({"role": "assistant", "content": bot_turn})

        final_user_prompt_with_context = f"""Please answer the following question based on the provided context. If the answer is not in the context, say you cannot find the information in the documents and try to create answer by your own words.

    [CONTEXT FROM DOCUMENTS]
    {context}
    [QUESTION]
    {message}
    """
        
        conversation.append({"role": "user", "content": final_user_prompt_with_context})

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answer_start = full_response.rfind('[/INST]') + len('[/INST]')
        final_answer = full_response[answer_start:].strip()

        return final_answer