from langchain.embeddings import SentenceTransformerEmbeddings #HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import os 
import copy
import pprint
#import google.generativeai as palm
from langchain.llms import GooglePalm
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from tempfile import NamedTemporaryFile
import streamlit as st
from ingest import pageextract
import warnings
warnings.filterwarnings("ignore")



MODES=["Page-By-Page","Complete"]

FILES=os.listdir("data")


if 'count' not in st.session_state:
    st.session_state.count=0

if 'mode' not in st.session_state:
    st.session_state.view=False
    st.session_state.mode=MODES[0]
    st.session_state.page=0


@st.cache_resource
def getapi():
    return str(open("API.txt","r",encoding='utf-8').read())


PALM_API=getapi()
#palm.configure(api_key=PALM_API)


@st.cache_resource
def getmodel():
    llm=GooglePalm(google_api_key=PALM_API,temperature=0,max_output_tokens=4000)
    return llm

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

def startview():
    #st.runtime.legacy_caching.clear_cache()
    st.session_state.page=0
    st.session_state.mode=mode
    try:
        st.session_state.data=getData()
    except:
        st.write("ERROR IN LOADING DATA.")
    

def resetview():
    st.session_state.view=False
    st.session_state.mode=mode
    st.session_state.page=0
    



def getdata(fi):
    #print(fi.path)
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n","."," "],chunk_size=2000, chunk_overlap=100)
    texts=[]
    with NamedTemporaryFile(dir='.', suffix='.pdf',delete=False) as f:
        f.write(fi.getbuffer())
        #print("DAADD>>>",f.name)
        texts=texts+copy.deepcopy(text_splitter.split_documents(PyPDFLoader(f.name).load()))
        #your_function_which_takes_a_path()
    #print(f"{fi}>>>>><<<<<{texts[0:20]}")
    os.remove(f.name)
    return fi.read()
mode=st.sidebar.radio("Pick one", MODES,on_change=resetview)
file=st.sidebar.selectbox("Pick one", FILES)
def getData():
    texts=[]
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n","."," "],chunk_size=1000, chunk_overlap=100)
    texts=texts+copy.deepcopy(text_splitter.split_documents(PyPDFLoader("data/"+file).load()))
    print(">>Data recieved.")
    #print(texts[0])
    st.session_state.dummy=copy.deepcopy(texts[0])
    return pageextract(texts)#print(f"*******{texts[0]}")

def sliderch():
    st.session_state.page=BP

st.session_state.file=file
st.title(f'{mode} Summary')

#file=st.sidebar.file_uploader("Upload a CSV")#,on_change=getdata)
prompt=False#st.sidebar.text_input("Enter query")
but=st.sidebar.button("Click me",on_click=startview)
#st.write(but)
if (but or st.session_state.view) and (st.session_state.mode==MODES[0]) and file:
    col1, col2, col3 = st.columns([1, 3, 3])
    BB=col1.button("Next page")
    PP=col1.button("Prev page")
    BP=col1.slider("Pick a page", 0, len(st.session_state.data),on_change=sliderch)
    #print(BP)
    try:
        chain = load_summarize_chain(getmodel(), chain_type="stuff")
    except:
        st.write("ERROR IN LOADING MODEL.")
    col3.header("Summary : ")
    #tt=getData(file)
    #no=st.number_input("Pick a page", 0, tt[-1].metadata['page'])
    #print(f"%%%%%%%{file}")
    st.session_state.view=True
    try:
        if BB:
            st.session_state.page+=1
            if st.session_state.page >=len(st.session_state.data):
                st.session_state.page=len(st.session_state.data)-1
        if PP:
            st.session_state.page-=1
            if st.session_state.page <0:
                st.session_state.page=0
        col2.header(f"Page {st.session_state.page} Preview: ")    
    
        col2.write(st.session_state.data[st.session_state.page].page_content[0:600]+" .......")
        st.session_state.dummy.page_content=st.session_state.data[st.session_state.page].page_content
        col3.write(chain.run([st.session_state.dummy]))
    except:
        col3.write("ISSUES IN GENERATING SUMMARY")
    #st.write(file.read())
elif (but or st.session_state.view) and (st.session_state.mode==MODES[1]) and file:
    col1,col2=st.columns([2,3])
    col1.write("WORK IN")
    col2.write("PROGRESS ")
    st.session_state.view=True
    st.session_state.file=file