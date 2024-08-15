from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

def load_embeddings(embeddings_model):
    return HuggingFaceEmbeddings(model_name = embeddings_model)

def load_pdf(file):
    loader = PyPDFLoader(file)
    docs = loader.load()
    return docs

def spilt_text(docs, chunksize, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunksize, 
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)
    ids = [f"{i}" for i in range(len(splits))]
    return splits, ids
    
def load_vector_db(collection_name, embedding_function):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    
def create_prompt(template):
    return ChatPromptTemplate.from_template(template)
    
def load_model(llm_model):
    model = AutoModelForCausalLM.from_pretrained( 
        llm_model,  
        device_map="cuda",  
        torch_dtype="auto",  
        trust_remote_code=True, 
    ) 
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    return model, tokenizer

def load_pipe(model, tokenizer, temp, max_tokens, top_k, top_p):
    pipe = pipeline( 
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temp, 
        top_k=top_k,
        top_p=top_p,
    ) 
    return HuggingFacePipeline(pipeline=pipe)

def generate_response(user_message, model, tokenizer, temp, max_tokens, top_k, top_p):
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


def generate_rag_response(retriever, prompt, llm, question):
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)
