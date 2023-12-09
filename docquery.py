from langchain.embeddings import SentenceTransformerEmbeddings #HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import os 
import copy
import pprint
#import google.generativeai as palm
from langchain.llms import GooglePalm
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

PALM_API="AIzaSyAIzDH7NVopxUvOL8PAqBnKZqdmAoXeS28"
#palm.configure(api_key=PALM_API)


def getmodel():
    "test"
    PALM_API="AIzaSyAIzDH7NVopxUvOL8PAqBnKZqdmAoXeS28"
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss", embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 10})
    #prompt=getprompt()
    llm=GooglePalm(google_api_key=PALM_API,temperature=0.00003,max_output_tokens=512)
    qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                          chain_type='refine',
                                          retriever=retriever,
                                          return_source_documents=True,
                                          #chain_type_kwargs={'prompt': prompt},
                                        verbose=True)
    return qa_llm

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

def EXTRACT():
    print(f"{'>>>'*17} QUERY DOCS{'<<<'*17}")
    try:
        llm=getmodel()
    except:
        print("CANNOT LOAD MODEL OR DATABASE")
        print(f"{'###'*40}")
        return
    while True:
        print(f"{'###'*40}")
        prompt=input("(To stop querying enter exit) \n Query  : ")

        if prompt:
            if prompt.find('exit')==0:
                return
            else:
                pass

            try:
                result=parseresult(llm(prompt))
                sources=getsources(result)
                result=result["result"]
            except:
                result='Error ocurred!'
                sources=[]
            print(f"{'!!!'*40}")
            print(f"QUERY: {prompt}")
            print(f"{'###'*40}")
            print("RESULT:")
            #print(f"{'###'*40}")
            print(result)
            print(f"{'$$$'*40}")
            print("SOURCES:")
            #print(f"{'$$$'*40}")
            print(sources)
            print(f"{'>>>'*40}")