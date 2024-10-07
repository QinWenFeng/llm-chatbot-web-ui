import ollama

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# List all available LLM models
def list_model():
    models_info = ollama.list()
    models_list = [model['model'] for model in models_info['models']]
    return models_list

# Loads embedding model from HuggingFace
def load_embeddings(embedding_model):
    return HuggingFaceEmbeddings(model_name=embedding_model)

# Loads the content from a list of PDF files
def load_pdf(files):
    docs = []
    for file in files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())
    return docs

# Splits the documents into smaller chunks for processing
def split_text(docs, chunksize, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunksize, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    ids = [f"{i}" for i in range(len(chunks))]
    return chunks, ids
    
# Loads a Chroma vector database
def load_chroma_db(collection_name, embedding_function, persistent_directory):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=f"./db/{persistent_directory}",
    )

# Converts retrived result to json format
def results_to_json(results):
    results_json = []
    for res in results:
        results_json.append(
            {
                "page_content": res.page_content,
                "metadata": res.metadata,
            }
        )
    return results_json

# Creates prompt template
def create_prompt(template):
    return ChatPromptTemplate.from_template(template)
    
# Loads LLM models from the environment variables
def load_llm(model, endpoint, temp, max_tokens, top_p):
    llm = ChatOpenAI(
        model=model,
        base_url=endpoint,
        api_key="ollama", 
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return llm

# Generates response without RAG
def generate_response(user_message, llm):   
    response = llm.invoke(user_message)
    return response.content

# Generates response with RAG
def generate_rag_response(retriever, prompt, llm, question):
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(question)
    return response