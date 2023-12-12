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


if 'work' not in st.session_state:
    st.session_state.work=False


@st.cache_resource
def getapi():
    return str(open("API.txt","r",encoding='utf-8').read())


PALM_API=getapi()
#palm.configure(api_key=PALM_API)


@st.cache_resource
def getmodel():
    llm=GooglePalm(google_api_key=PALM_API,temperature=0,max_output_tokens=512)
    return llm

@st.cache_resource
def getprompts():
    reviewerprompt=""" You are an expert at reviewing and creating questions. Infer the domain of relevance from the below text,
    and assume a persona of an expert-reviewer in that domain to generate question-answer pairs from the given text.
    Your goal is to prepare user for their exam and quiz.
        You do this by asking questions about the text below:
    
        ------------
        {text}
        ------------
    
        Create questions that will prepare the user for their tests.
        Make sure not to lose any important information.

    """
    refineprompt = ("""
    You are an expert at creating practice questions based on material provided.
    Your goal is to help user prepare for a test.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------
    
    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    Return a parsed JSON object.
    """
    )
    checkerprompt="""
    You are an expert at reviewing asnwers to questions based on material provided.
    Your goal is to help user prepare for a test.
    We have received some answer given by the user to questions mentioned.
    The actual answers are mentioned under RESULT.
    Compare how well the user answers compare with actual RESULT and generate feedback.
    ------------
    {question}
    {RESULT}
    {USER}
    ------------
    Feedback:
    """

    promptques = PromptTemplate(template=reviewerprompt, input_variables=["text"])
    refineques= PromptTemplate(template=refineprompt,input_variables=["existing_answer","text"])
    checker=PromptTemplate(template=checkerprompt,input_variables=["question","RESULT","USER"])
    
    return (promptques,refineques,checker)

def startview():
    #st.legacy_caching.caching.clear_cache()
    #st.runtime.legacy_caching.clear_cache()
    st.session_state.page=0
    st.session_state.work=True
    st.session_state["Answer:"] = ""
    #st.session_state.mode=mode
    getQues.clear()
    getData.clear()
    st.session_state.QA=None
    try:
        st.session_state.data=getData()
        if len(st.session_state.data)>10:
            st.sidebar.write("Loading initial 10 pages.")
            st.session_state.data=st.session_state.data[0:10]
        
    except:
        st.write("ERROR IN LOADING DATA.")

def nextques():
    st.session_state["Answer:"] = ""
    st.session_state.page +=1
    if st.session_state.page >= len(st.session_state.QA[0]):
         st.session_state.page = len(st.session_state.QA[0])-1

def prevques():
    st.session_state["Answer:"] = ""
    st.session_state.page -=1
    if st.session_state.page <0:
         st.session_state.page = 0  

#mode=st.sidebar.radio("Pick one", MODES,on_change=resetview)
file=st.sidebar.selectbox("Pick one", FILES)
@st.cache_resource
def getData():
    texts=[]
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n","."," "],chunk_size=1000, chunk_overlap=100)
    texts=texts+copy.deepcopy(text_splitter.split_documents(PyPDFLoader("data/"+file).load()))
    print(">>Data recieved.")
    st.session_state.dummy=copy.deepcopy(texts[0])
    return pageextract(texts)

def sliderch():
    st.session_state.page=BP




st.session_state.file=file
st.title(f'Quiz:')
prompt=False
but=st.sidebar.button("Quiz me!",on_click=startview)

llm=getmodel()
pques,prefine,pcheck=getprompts()

@st.cache_resource
def getQues():
    print("GETTING>>>>>")
    #print(prefine)
    quesgen=load_summarize_chain(llm=llm,chain_type='refine',verbose=False,question_prompt=pques,refine_prompt=prefine)
    ques = quesgen.run([st.session_state.data[st.session_state.page]])
    print(ques)
    if ques.find("Answer")==-1:
        lbl="A:"
    else:
        lbl="Answer:"
    QA=ques.split("\n\n")
    qs=[]
    ans=[]
    for qa in QA:
        qs.append(qa.split(lbl)[0].split(":")[1])
        ans.append(qa.split(lbl)[1])
        
    #qs=ques.split("ANSWERS")[0]
    #ans=ques.split("ANSWERS")[1]
    #print("################",ques.split("\n\n"))
    #return (ques)
    return (qs,ans)

if but or st.session_state.work:
    col2, col3 = st.columns([ 7,1])
    try:
        qs,ans=getQues()
        st.session_state.QA=copy.deepcopy((qs,ans))
        col2.header(f" {st.session_state.QA[0][st.session_state.page]}")
        answer=col2.text_input("Answer:",key="Answer:")
        c21,_,c22=col2.columns([2.5,2.5,2.5])
        BB=c22.button("NEXT QUES",on_click=nextques)
        PP=c21.button("PREV QUES",on_click=prevques)
        col2.header("Feedback:")

        if answer:
            check={"question":st.session_state.QA[0][st.session_state.page],"RESULT":st.session_state.QA[1][st.session_state.page],"USER":answer}
            chain = pcheck | llm
            feed=chain.invoke(check)
            col2.write(f"Expected Answer: {st.session_state.QA[1][st.session_state.page]}")
            col2.write(f"Feedback : {feed}")
    
    except:
        col2.write("ERROR LOADING QUESTIONS.")
    