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
#import streamlit as st
import os
import subprocess
import sentence_transformers
import nltk

while True:
    choice=input(f"{'>>'*10} \n Choose : 0-> Ingest ; 1->Query ; 2->Exit \n : ")

    if choice=='0':
        from ingest import INGESTER
        print("INGESTING!")
        INGESTER()
        print("Ingested")
    elif choice=='1':
        from docquery import EXTRACT
        print("QUERYING!")
        EXTRACT()
    elif choice=='2':
        print("Exiting!")
        time.sleep(2)
        break
    else:
        print("Invalid choice!")
