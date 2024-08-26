import gradio as gr
from utils import load_embeddings, load_pdf, spilt_text, load_vector_db, create_prompt, load_model, load_pipe, generate_response, generate_rag_response

embeddings = load_embeddings("sentence-transformers/all-MiniLM-L6-v2")

def reset_params():
    temp = 1.0
    max_tokens = 500
    top_k = 50
    top_p = 1.0
    return temp, max_tokens, top_k, top_p

def user(user_message, chat_history):
    chat_history.append((user_message["text"], None))
    return gr.MultimodalTextbox(value=user_message), chat_history

def bot(user_message, chat_history, temp, max_tokens, top_k, top_p):
    model, tokenizer = load_model("microsoft/Phi-3-mini-4k-instruct")
    if user_message["files"]:
        docs = load_pdf(user_message["files"])
        splits, ids = spilt_text(docs, 300, 50)
        vector_store = load_vector_db("collection", embeddings)
        vector_store.add_documents(documents=splits, ids=ids)
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}"""
        prompt = create_prompt(template)
        llm = load_pipe(model, tokenizer, temp, max_tokens, top_k, top_p)
        bot_message = generate_rag_response(retriever, prompt, llm, user_message["text"])
    else:
        bot_message = generate_response(user_message["text"], model, tokenizer, temp, max_tokens, top_k, top_p)
    chat_history.append((None, bot_message))
    return gr.MultimodalTextbox(value=None), chat_history

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Parameters", open=True):
                temp = gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=1.0, step=0.1, interactive=True, info="Controls the randomness in the response, use higher to be more creative")
                max_tokens = gr.Slider(label="Token limit", value=500, minimum=0, maximum=1024, step=1, interactive=True, info="Limits the maximum output of tokens for response")
                top_k = gr.Slider(label="Top-k", value=50, minimum=0, maximum=50, step=1, interactive=True, info="Limits the model to choosing from the top 'k' most likely next tokens")
                top_p = gr.Slider(label="Top-p", value=1.0, minimum=0.0, maximum=1.0, step=0.1, interactive=True, info="Limits the model's choices to the smallest set of tokens whose cumulative probability is at least 'p'")
                reset_params_button = gr.Button("RESET PARAMETERS")
                reset_params_button.click(reset_params, None, outputs=[temp, max_tokens, top_k, top_p])
            
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(placeholder="<strong>Welcome to Chatbot</strong>", height=500, avatar_images=(None, "images/assistant.png"))
            msg = gr.MultimodalTextbox(label="Context", placeholder="Enter message or upload file", file_count="multiple", interactive=True, show_label=False)
            clear_button = gr.Button("Clear")  
            
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [msg, chatbot, temp, max_tokens, top_k, top_p], [msg, chatbot]
            )
            clear_button.click(lambda: None, None, chatbot, queue=False)
  
if __name__ == "__main__":
    demo.launch()
