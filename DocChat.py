from langchain.embeddings import SentenceTransformerEmbeddings #HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import os 
import copy
import pprint
#import google.generativeai as palm
from langchain.llms import GooglePalm
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


@st.cache_resource
def getapi():
    return str(open("API.txt","r",encoding='utf-8').read())


PALM_API=getapi()
#palm.configure(api_key=PALM_API)


@st.cache_resource
def getmodel():
    "test"
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss", embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 10})
    #prompt=getprompt()
    llm=GooglePalm(google_api_key=PALM_API,temperature=0,max_output_tokens=512)
    qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                          chain_type='refine',
                                          retriever=retriever,
                                          return_source_documents=True,
                                          #chain_type_kwargs={'prompt': prompt},
                                        verbose=True)
    return qa_llm

@st.cache_resource
def getprompt():
    template = """Use the information to elaborate in points about the user's query.
    If user mentions something not in the 'Context', just answer that you don't know.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Query: {question}
    
    Only return the helpful answer below and nothing else.
    
    Helpful answer:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'])
    return prompt

def parseresult(result):
    
    PARSED=copy.deepcopy(result)
    docs=PARSED['source_documents']
    sourcepage=[]
    for d in docs:
        sourcepage.append(d.metadata['page'])
    PARSED['source_pages']=copy.deepcopy(sourcepage)
    del sourcepage,result
    return PARSED

def getsources(result):
    sources=[]
    for s in result['source_documents']:
        sources.append(f"{s.metadata}")
    return sources
    
st.title('Query Docs')

prompt=st.sidebar.text_input("Enter query")
try:
    llm=getmodel()
except:
    st.write("CANNOT LOAD MODEL OR DATABASE")
    #print("ERROR LOADING MODEL OR DATABASE")

if prompt:
    if prompt.find("exit")==0:
        import sys
        sys.exit()
    try:
        result=parseresult(llm(prompt))
        sources=getsources(result)
        result=result["result"]
    except:
        result="Error in retrieving! \n You can try reframing your query, if it doesnt work there may be something broken. \n :/ "
        sources=[]
        
    print(">>>>>>>>>>>>><<<<<<<<<<<<<<<<<")
    st.header("Result")
    st.write(result)
    st.header("Sources")
    st.write(sources)