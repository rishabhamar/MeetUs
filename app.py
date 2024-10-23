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
import speech_recognition as sr
import moviepy.editor as mp
import tempfile
import os
from pydub import AudioSegment
import io
import whisper
import speech_recognition as sr

from dotenv import load_dotenv
load_dotenv()

# Function to handle meeting agenda creation
def page_agenda():
    st.title("Meeting Agenda")

    # Set up environment variables and API keys
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    # Initialize embeddings and language model
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Setup Streamlit
    st.write("Upload Documents Related to the meeting")


    # Check if api key is provided
    llm=ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

    ##chat interface 
    session_id=st.sidebar.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files = st.file_uploader("Upload your Documents", type=["pdf"], accept_multiple_files=True)

    ## Process Uploaded files

    if uploaded_files:
        # Load and process documents
        documents=[]
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            temppdf=f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            # Save uploaded file temporarily
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    #Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits,embedding = embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt=(
"You are an expert meeting facilitator. Your task is to create a comprehensive and effective meeting agenda based on the provided documents and any additional input from the user. Consider the following:"
    "1. The overall purpose and desired outcomes of the meeting"
    "2. The key topics that need to be addressed"
    "3. The appropriate time allocation for each agenda item"
    "4. Any pre-meeting preparation required for attendees"
    "5. Opportunities for participant engagement and discussion"
    "6. A clear structure that flows logically from one topic to the next"
    "Create a well-structured, time-bound agenda that will ensure a productive and focused meeting."
        )

         # Set up prompts and chains for agenda creation
        contextualize_q_prompt= ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        # Set up system prompt and QA chain
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

        # Function to get or create session history
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        # Set up conversational RAG chain
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Generate meeting agenda
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
            
            # Display generated agenda
            st.title("Meeting Agenda")
            st.write("Assistant:", st.session_state.response['answer'])

            # Provide option to download agenda as PDF
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
                pdf_path = os.path.join(os.getcwd(),'data' ,'Agenda.pdf')
                pdf.output(pdf_path)

                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name=f"Agenda.pdf",
                        mime="application/pdf"
                    )

# Function to handle meeting video upload and transcription
def page_upload():
    st.title("Upload Meeting")

    # Function to convert video to audio
    def video_to_audio(video_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(video_file.read())
            video_path = tmpfile.name

        current_dir=os.getcwd()
        audio_path=os.path.join(current_dir,'temp','audio.wav')
        
        try:
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, codec='pcm_s16le')  # Specify codec
            video.close()
        except Exception as e:
            st.error(f"Error converting video to audio: {str(e)}")
            return None
        finally:
            if 'video' in locals():
                video.close()
            
            for attempt in range(5):
                try:
                    os.unlink(video_path)
                    break
                except PermissionError:
                    time.sleep(1)

        return audio_path

    # Function to transcribe audio
    def transcribe_audio(audio_file_path):
        try:
            model=sr.Recognizer()
            with sr.AudioFile(audio_file_path) as source:
                audio = model.listen(source)
                transcript = model.recognize_whisper(audio)
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
        return transcript

    # Function to save transcript as PDF
    def save_transcript_as_pdf(transcript,save_path):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size = 12)
        pdf.multi_cell(0, 10, transcript)
        pdf.output(save_path)

     # Set up file uploader for video
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)
        
        # Transcribe video when button is clicked
        if st.button("Transcribe"):
            with st.spinner("Processing..."):
                try:
                    audio_file_path = video_to_audio(uploaded_file)
                    transcript = transcribe_audio(audio_file_path)
                    st.success("Transcription Complete!")
                    pdf_path = os.path.join(os.getcwd(),'data' ,'Transcript.pdf')
                    pdf_output = save_transcript_as_pdf(transcript,pdf_path)

                    # Provide option to download transcript as PDF
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                    
                    st.download_button(
                        label="Download Transcript as PDF",
                        data=pdf_bytes,
                        file_name="transcript.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

def page_track():
    st.title("Meeting Tracking")  # Set the page title

    # Set API keys from environment variables
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')  
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    # Load HuggingFace embeddings model for text embedding
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize the LLM with API key for the model "Gemma2-9b-It"
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

    # User input for session ID (default value is "default_session")
    session_id = st.sidebar.text_input("Session ID", value="default_session")

    # Initialize session state to store information if it doesn't already exist
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Upload file handler, accepting PDF files
    uploaded_files = st.file_uploader("Upload your Documents", type=["pdf"], accept_multiple_files=True)

    # If files are uploaded, process them
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            # Save uploaded PDF file locally
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

        documents = []
        # Load the content of the uploaded PDF file
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

        # Split the documents into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Store document embeddings in Chroma vectorstore
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Define a system prompt to contextualize the question
        contextualize_q_system_prompt = (
            "What is the tracking of meeting"
        )

        # Create prompt template for conversation history
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # Create a history-aware retriever
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Define the system prompt for meeting analysis
        system_prompt = (
            "You are an AI meeting analyst. Your task is to compare the original meeting agenda with the actual meeting transcript and provide a detailed analysis"
            "\n\n"
            "{context}"
        )

        # Create prompt template for answering questions
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # Chain LLM and document retriever for question-answering
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Function to retrieve session-specific chat history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Define a chain to handle conversation with history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Example user input to analyze meeting flow
        user_input = "Analyze the meeting flow and effectiveness based on the agenda and transcript."
        
        if user_input:
            # If no previous response, run RAG model to generate response
            if 'response' not in st.session_state:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    },
                )
                st.session_state.response = response  # Store the response in session state
                st.session_state.history = session_history.messages
            
            # Display AI response for meeting flow
            st.title("Meeting Flow")
            st.write("Assistant:", st.session_state.response['answer'])

            # Option to download the AI-generated response as a PDF
            if st.button("Download AI Response as PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                # Add content to the PDF
                pdf.cell(200, 10, txt="AI Meeting Manager Response", ln=True, align="C")
                pdf.ln(10)  # New line

                # Write the AI response in the PDF
                pdf.multi_cell(0, 10, txt="Assistant's Response:\n" + st.session_state.response['answer'])

                # Save the PDF
                pdf_path = os.path.join(os.getcwd(), 'data', 'Track.pdf')
                pdf.output(pdf_path)

                # Provide download link for the PDF
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name=f"Track.pdf",
                        mime="application/pdf"
                    )

def page_summary():
    st.title("Meeting Summary")  # Set the page title

    # Set API keys from environment variables
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    # Load HuggingFace embeddings model for text embedding
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Write introductory text for file upload
    st.write("Upload Documents Related to the meeting")

    # Initialize the LLM with API key for the model "Gemma2-9b-It"
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

    # User input for session ID (default value is "default_session")
    session_id = st.sidebar.text_input("Session ID", value="default_session")

    # Initialize session state to store information if it doesn't already exist
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Upload file handler, accepting PDF files
    uploaded_files = st.file_uploader("Upload your Documents", type=["pdf"], accept_multiple_files=True)

    # If files are uploaded, process them
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            # Save uploaded PDF file locally
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

        documents = []
        # Load the content of the uploaded PDF file
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

        # Split the documents into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Store document embeddings in Chroma vectorstore
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Define a system prompt to contextualize the question
        contextualize_q_system_prompt = (
            "What is the summary of meeting"
        )

        # Create prompt template for conversation history
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # Create a history-aware retriever
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Define the system prompt for meeting summary
        system_prompt = (
            "As an AI meeting summarizer, your task is to create a concise yet comprehensive summary of the entire meeting"
            "\n\n"
            "{context}"
        )

        # Create prompt template for answering questions
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # Chain LLM and document retriever for question-answering
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Function to retrieve session-specific chat history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Define a chain to handle conversation with history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Example user input to generate meeting summary
        user_input = "Provide a comprehensive summary of the meeting based on the analysis."
        
        if user_input:
            # If no previous response, run RAG model to generate response
            if 'response' not in st.session_state:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    },
                )
                st.session_state.response = response  # Store the response in session state
                st.session_state.history = session_history.messages
            
            # Display AI response for meeting summary
            st.title("Meeting Summary")
            st.write("Assistant:", st.session_state.response['answer'])

            # Option to download the AI-generated summary as a PDF
            if st.button("Download AI Response as PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                # Add content to the PDF
                pdf.cell(200, 10, txt="AI Meeting Manager Response", ln=True, align="C")
                pdf.ln(10)  # New line

                # Write the AI response in the PDF
                pdf.multi_cell(0, 10, txt="Assistant's Response:\n" + st.session_state.response['answer'])

                # Save the PDF
                pdf_path = os.path.join(os.getcwd(), 'data', 'Summary.pdf')
                pdf.output(pdf_path)

                # Provide download link for the PDF
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name=f"Summary.pdf",
                        mime="application/pdf"
                    )

# Sidebar navigation to switch between pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Meeting Agenda", "Upload Meeting", "Meeting Tracking", "Meeting Summary"])

# Display the selected page
if page == "Meeting Agenda":
    page_agenda()
elif page == "Upload Meeting":
    page_upload()
elif page == "Meeting Tracking":
    page_track()
elif page == "Meeting Summary":
    page_summary()
