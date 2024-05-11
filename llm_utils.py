import sagemaker, boto3, json
from sagemaker.session import Session
from sagemaker.model import Model
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.predictor import Predictor
from sagemaker.utils import name_from_base
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
import os
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from llama_util import chat
from langchain.vectorstores import FAISS
from content import get_docs_from_folder
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import (
    LocalFileStore
)
import calendar;
import time;

class LlamaContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs" : [[{"role" : "system",
        "content" : "You are a kind robot."},
        {"role" : "user", "content" : prompt}]],
        "parameters" : {**model_kwargs}})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

def get_sagemaker_llm(endpoint_name):    
    parameters = {
        "max_length": 1000,
        "temperature": 0.1,
        }
    aws_region = boto3.Session().region_name
    sess = sagemaker.Session()
    content_handler = LlamaContentHandler()
    
    sm_llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=aws_region,
        model_kwargs=parameters,
        endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
        content_handler=content_handler,
    )    
    print(f' Sagemaker llm endpoint {endpoint_name} is configured to use.')
    return sm_llm;

def select_llm(llmname:str):
    supported_llms = ["OpenAI","Sagemaker-Lama2-7B"]
    global selected_llm
    global llm
    global embedding
    if(llmname not in supported_llms):
        raise Exception("Not supported "+llmname)
    selected_llm = llmname;
    load_dotenv()
  
    if(llmname == "OpenAI"):       
        openai.api_key = os.getenv("OPENAI_API_KEY")
        model_name="gpt-4"
        if os.getenv("OPEN_AI_MODEL") is None:
            model_name="gpt-4"
        else:
            model_name=os.getenv("OPEN_AI_MODEL")
        print("using model "+model_name)
        llm = ChatOpenAI(model_name=model_name,request_timeout=60)
        embedding = OpenAIEmbeddings()
    elif(llmname == "Sagemaker-Lama2-7B"):
        llm = get_sagemaker_llm(os.getenv("SAGEMAKER_ENDPOINT"))
        embedding = GPT4AllEmbeddings()

def chunk_embed_store(docs,chunk_size=1000, chunk_overlap=50):
    load_dotenv()
    #vec_store_loc = os.getenv("VECTOR_STORE_LOCATION")
    #print("vec_store_loc is "+vec_store_loc)
    #if not os.path.exists(vec_store_loc):
    #    os.makedirs(vec_store_loc)
    #fs = LocalFileStore(vec_store_loc)
    #cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    #    embedding, fs, namespace=embedding.model
    #)    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(docs)
    return FAISS.from_documents(all_splits, embedding)

def get_response(vecstore:VectorStore,question:str):
    docs = vecstore.similarity_search(question)
    source_content_list = []
    for doc in docs:
        source_content_list.append(doc.page_content)
    source_content = "\n".join(source_content_list)
    return chat(source_content,question)

def get_kb_vector_db():
    load_dotenv()
    print("kb_docs_loc -> "+os.getenv("KB_DOCS_LOCATION"))
    kb_docs_loc = os.getenv("KB_DOCS_LOCATION")    
    return chunk_embed_store(get_docs_from_folder(kb_docs_loc),chunk_size=2500,chunk_overlap=100)

def get_response_openai(vecstore:VectorStore,question:str):
    ## Use a shorter template to reduce the number of tokens in the prompt
    template = """Create a final answer to the given questions using the provided document excerpts (given in no particular order) as sources. ALWAYS include a "SOURCES" section in your answer citing only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not have enough information to answer the question and leave the SOURCES section empty. Use only the provided documents and do not attempt to fabricate an answer.

    ---------

    QUESTION: What  is the purpose of ARPA-H?
    =========
    Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
    SOURCES: 1-32
    Content: While we're at it, let's make sure every American can get the health care they need. \n\nWe've already made historic investments in health care. \n\nWe've made it easier for Americans to get the care they need, when they need it. \n\nWe've made it easier for Americans to get the treatments they need, when they need them. \n\nWe've made it easier for Americans to get the medications they need, when they need them.
    SOURCES: 1-33
    Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat's why I'm calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
    SOURCES: 1-30
    =========
    FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
    SOURCES: 1-32

    ---------

    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""

    STUFF_PROMPT = PromptTemplate(
        template=template, input_variables=["summaries", "question"]
    )
    model_name="gpt-4"
    if os.getenv("OPEN_AI_MODEL") is None:
        model_name="gpt-4"
    else:
        model_name=os.getenv("OPEN_AI_MODEL")
    print("using model "+model_name)    
    llm = ChatOpenAI(model_name=model_name)
    t1 = calendar.timegm(time.gmtime())
    docs = vecstore.similarity_search(question)
    t2 = calendar.timegm(time.gmtime())    
    print("Time taken to query vec store "+str(t2-t1))
    chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type="stuff",
            prompt=STUFF_PROMPT,
        )
    result = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    t3= calendar.timegm(time.gmtime())
    print("Time taken to get open ai response "+str(t3-t2))
    return result
