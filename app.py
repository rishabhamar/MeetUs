import streamlit as st
from fpdf import FPDF
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from datetime import datetime, time
from email.mime.text import MIMEText
import base64
import json
import os
import requests

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Setup Streamlit
st.title("MeetUs")
st.write("Upload Documents Related to the meeting")


# Check if api key is provided
llm=ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

##chat interface 
session_id=st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files = st.file_uploader("Upload your Documents", type=["pdf"], accept_multiple_files=True)

## Process Uploaded files

if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name

        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)

#Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits,embedding = embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt=(
        "What is the purpose of the meeting?" 
        "What are you hoping to achieve?"
        "Who are the attendees? This will help me determine the level of detail and formality needed."
        "What are the main topics to be discussed? List the key issues or subjects that need to be addressed."
        "Are there any pre-reading materials or background information? Attendees should be prepared."
        "Is there a desired outcome or decision to be made?"
    )

    contextualize_q_prompt= ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    #Answer Question

    system_prompt = (
        "Your are a meeting manager"
        "Your aim is to create agendas for current meeting from the uploaded file"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system" , system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = "Create a point wise structured agenda for the meeting"
    if user_input:
        if 'response' not in st.session_state:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },
            )
            st.session_state.response = response  # Store the response in session state
            st.session_state.history = session_history.messages
            
        st.title("Meeting Agenda")
        st.write("Assistant:", st.session_state.response['answer'])

        if st.button("Download AI Response as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Add content
            pdf.cell(200, 10, txt="AI Meeting Manager Response", ln=True, align="C")
            pdf.ln(10)  # New line

            # AI response
            pdf.multi_cell(0, 10, txt="Assistant's Response:\n" + st.session_state.response['answer'])

            # Save the PDF
            pdf_output = f"./ai_response_{session_id}.pdf"
            pdf.output(pdf_output)

            with open(pdf_output, "rb") as file:
                st.download_button(
                    label="Download PDF",
                    data=file,
                    file_name=f"ai_response_{session_id}.pdf",
                    mime="application/pdf"
                )
