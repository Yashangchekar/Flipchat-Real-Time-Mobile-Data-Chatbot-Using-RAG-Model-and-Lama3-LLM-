# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:56:08 2024

@author: yash
"""

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192",streaming=True)
#llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192")

# 1. Load, chunk and index the contents of the blog to create a retriever.
import bs4
loader = WebBaseLoader(
    web_paths=("https://www.flipkart.com/search?q=mobiles&sid=tyy%2C4io&as=on&as-show=on&otracker=AS_QueryStore_OrganicAutoSuggest_2_7_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_2_7_na_na_na&as-pos=2&as-type=RECENT&suggestionId=mobiles%7CMobiles&requestId=6b019d01-4d80-43a5-ba42-9f39c182b117&as-searchtext=mobiles",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("DOjaWF gdgoEp","DOjaWF gdgoEp","DOjaWF gdgoEp")
        )
    ),
)

docs=loader.load()



text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splits=text_splitter.split_documents(docs)
vectorstore=FAISS.from_documents(documents=splits,embedding=embeddings)
retriever=vectorstore.as_retriever() #Interface for vectordb




retriever_tool=create_retriever_tool(retriever,"flipchat","flipkart mobile search")


tools=[retriever_tool]
# Create an agent executor
## Prompt Template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)






# Streamlit UI setup
st.title("FLipChat 🛒 📱 ⓕ 🤖")

# User input for the chatbot
user_input = st.text_input("You: ", placeholder="Ask me anything about mobile phones...")

# Callback handler for Streamlit
callback_handler = StreamlitCallbackHandler(st.container())

if user_input:
    # Run the retrieval-augmented generation chain
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": user_input})
        st.write(response['answer'])
#        response_text = response['answer'] if response['answer'] else "I'm not sure about that."

    # Display the output
#    st.text_area("Chatbot:", value=response_text['answer'], height=200)