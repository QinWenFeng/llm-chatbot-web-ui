import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

checkpoint = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained( 
    checkpoint,  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def reset_params():
    temp = 1.0
    max_tokens = 500
    top_k = 50
    top_p = 1.0
    return temp, max_tokens, top_k, top_p
    
def generate(user_message, temp, max_tokens, top_k, top_p):
    pipe = pipeline( 
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
    ) 

    generation_args = { 
        "max_new_tokens": max_tokens, 
        "return_full_text": False, 
        "do_sample": True, 
        "temperature": temp, 
        "top_k": top_k,
        "top_p": top_p,
    } 
    
    output = pipe(user_message, **generation_args) 
    return output[0]['generated_text']

def respond(user_message, chat_history, temp, max_tokens, top_k, top_p):
    bot_message = generate(user_message, temp, max_tokens, top_k, top_p)
    chat_history.append((user_message, bot_message))
    return "", chat_history

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            temp = gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=1.0, step=0.1, interactive=True, info="Controls the randomness in the response, use higher to be more creative")
            max_tokens = gr.Slider(label="Token limit", value=500, minimum=0, maximum=1024, step=1, interactive=True, info="Limits the maximum output of tokens for response")
            top_k = gr.Slider(label="Top-k", value=50, minimum=0, maximum=50, step=1, interactive=True, info="Limits the model to choosing from the top 'k' most likely next tokens")
            top_p = gr.Slider(label="Top-p", value=1.0, minimum=0.0, maximum=1.0, step=0.1, interactive=True, info="Limits the model's choices to the smallest set of tokens whose cumulative probability is at least 'p'")
            reset_params_button = gr.Button("RESET PARAMETERS")
            reset_params_button.click(reset_params, None, outputs=[temp, max_tokens, top_k, top_p])
            
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(placeholder="<strong>Welcome to Chatbot</strong>", height=500, avatar_images=(None, "images/assistant.png"))
            msg = gr.Textbox(placeholder="Type a message")
            clear_button = gr.Button("Clear")  
            
            msg.submit(respond, [msg, chatbot, temp, max_tokens, top_k, top_p], [msg, chatbot])
            clear_button.click(lambda: None, None, chatbot, queue=False)
  
if __name__ == "__main__":
    demo.launch()