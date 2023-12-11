from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings #HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import os 
import copy
import time
import pprint
#import google.generativeai as palm
from langchain.llms import GooglePalm
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from tempfile import NamedTemporaryFile

import streamlit
import streamlit.web.cli as stcli

import os,sys
import subprocess
import sentence_transformers
import nltk
import warnings
warnings.filterwarnings("ignore")

def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path



while True:
    choice=input(f"{'>>'*10} \n Choose : 0-> Ingest ; 1->Query ; 2->Summary ; 3->Exit \n : ")
    if choice=='0':
        from ingest import INGESTER
        print(f"\n\n{'>>'*10}INGESTING!{'<<'*10}\n")
        print("(NOTE: It may take sometime depending on the documents in the data folder. Wait for it to end, it'll inform you of the status.)\n")
        print(f"{'###'*20}")
        try:
            print(f"\n\n{'>>>'*7} Ingesting Files: \n\n")
            INGESTER()
            print(f"\n\n{'>>'*7}INGESTED! Database ready for query!{'<<'*7}\n\n")
        except:
            print(f"\n\n{'>>'*7}ERROR OCCURED! Database not created!{'<<'*7}\n\n")
    elif choice=='1':
        #from docquery import EXTRACT
        print("\nSTARTING QUERYING!\n")
        try:
            sys.argv = [
            "streamlit",
            "run",
            resolve_path("DocChat.py"),
            "--global.developmentMode=false",
            ]
            sys.exit(stcli.main())
        except:
            print("\n\nERROR WHILE INITIATING QUERYING!\n\n")
        #EXTRACT()

    elif choice=='2':
        print("\nSTARTING Summary-ing!\n")
        try:
            sys.argv = [
            "streamlit",
            "run",
            resolve_path("DocSumm.py"),
            "--global.developmentMode=false",
            ]
            sys.exit(stcli.main())
        except:
            print("\n\nERROR WHILE INITIATING Summary-ing!\n\n")
    elif choice=='3':
        print("Exiting!")
        break
    else:
        print("Invalid choice!")
