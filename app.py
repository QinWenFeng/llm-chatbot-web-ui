import gradio as gr
from utils import (
    list_model, load_embeddings, load_pdf, split_text, load_chroma_db,
    results_to_json,  create_prompt, load_llm, generate_response, generate_rag_response
)

def user(user_message, chat_history):
    chat_history.append((user_message, None))
    return gr.Textbox(value=user_message), chat_history

def bot(user_message, uploaded_file, chat_history, model, endpoint, temp, max_tokens, top_p, k):
    llm = load_llm(model, endpoint, temp, max_tokens, top_p)
    if uploaded_file:
        docs = load_pdf(uploaded_file)
        chunks, ids = split_text(docs, 300, 50)
        embeddings_function = load_embeddings("sentence-transformers/all-MiniLM-L6-v2")
        vector_store = load_chroma_db("collection", embeddings_function, "pdf")
        vector_store.add_documents(documents=chunks, ids=ids)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k},
        )
        search_result = retriever.invoke(user_message)
        search_result_json = results_to_json(search_result)
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}"""
        prompt = create_prompt(template)
        bot_message = generate_rag_response(retriever, prompt, llm, user_message)
    else:
        search_result_json = None
        bot_message = generate_response(user_message, llm)
    chat_history.append((None, bot_message))
    return gr.Textbox(value=None), chat_history, search_result_json

with gr.Blocks() as demo:
    with gr.Tab("Chat"):
        with gr.Row():
            # Left panel
            with gr.Column(scale=1):
                models_list = list_model()
                model = gr.Dropdown(models_list, label="OLLAMA_MODEL") 
                endpoint = gr.Textbox(label="LOCAL_OLLAMA_ENDPOINT", value="http://localhost:11434/v1", interactive=True)

                with gr.Accordion("Upload File"):  
                    uploaded_file = gr.File(file_count="multiple")
                
            # Middle panel
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(placeholder="<strong>Welcome to Chatbot</strong>", height=500, avatar_images=(None, "images/assistant.png"))
                input_msg = gr.Textbox(label="Input Message", show_label=False)
                clear_button = gr.Button("Clear")

                clear_button.click(lambda: None, None, chatbot, queue=False)

            # Right panel
            with gr.Column(scale=1):
                search_result_json = gr.JSON(label="Search Result", min_width=50, height=580)
    
    with gr.Tab("Settings"):
        with gr.Accordion("Generation Parameters"):
            temp = gr.Slider(label="Temperature", value=0.7, minimum=0.0, maximum=1.0, step=0.1, interactive=True, info="Controls the randomness in the response, use higher to be more creative")
            max_tokens = gr.Slider(label="Token limit", value=5000, minimum=0, maximum=10000, step=1, interactive=True, info="Limits the maximum output of tokens for response")
            top_p = gr.Slider(label="Top-p", value=1.0, minimum=0.0, maximum=1.0, step=0.1, interactive=True, info="Limits the model's choices to the smallest set of tokens whose cumulative probability is at least 'p'")

        with gr.Accordion("Search Parameters"):
            k = gr.Number(label="k", value=3, minimum=1, interactive=True)

    input_msg.submit(user, [input_msg, chatbot], [input_msg, chatbot], queue=False).then(
        bot, [input_msg, uploaded_file, chatbot, model, endpoint, temp, max_tokens, top_p, k], [input_msg, chatbot, search_result_json]
    )        



if __name__ == "__main__":
    demo.launch()