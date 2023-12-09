from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings #HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

import os
def getdoctext(dir):
    os.chdir(dir)
    files=os.listdir()
    texts=[]
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n","."," "],chunk_size=2000, chunk_overlap=100)
    for file in files:
        if ".pdf" in file:
            print(file)
            texts=texts + text_splitter.split_documents(PyPDFLoader(file).load())
    os.chdir("..")
    return texts

def pageextract(texts):
    PAGES=[]
    id=0
    while id<len(texts):
        if (texts[id].metadata['page']-7)==len(PAGES):
            temp=''
            while id<len(texts) and (texts[id].metadata['page']-7)==len(PAGES) :
                temp=temp+texts[id].page_content
                id=id+1
            PAGES.append(copy.deepcopy(texts[0]))
            PAGES[-1].page_content=temp
            PAGES[-1].metadata['page']=1*len(PAGES)
    
    del PAGES

def makedb(chunks,embeddings):
    
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss")
    del db

    
def INGESTER():
    chunks=getdoctext("data")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    makedb(chunks,embeddings)