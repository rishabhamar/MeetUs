# MeetUs

## Setup

* Import the file into your pc

* Then go the file location

### Create Virtual Environment

**python -m venv myenv**

### Enabling virtual Environment

**myenv\Scripts\activate**

### Starting app

**streamlit run app.py**

## Usage Guidelines

### User Interface

The application interface consists of a sidebar for navigation and a main content area for specific functionalities.

#### Meeting Agenda:

Upload relevant documents (e.g., meeting objectives, project proposals).
Interact with the chat interface to provide details about the meeting.
Download the generated agenda as a PDF.

#### Upload Meeting:

Upload a video recording of the meeting.
Transcribe the video and download the transcript as a PDF.

#### Meeting Tracking:

Upload meeting agenda and transcript documents (PDF format).
Analyze meeting flow and effectiveness through a chat interface.
Download the analysis report as a PDF.

#### Meeting Summary:

Upload meeting agenda and transcript documents (PDF format).
Generate a concise summary based on the uploaded documents.
Download the meeting summary as a PDF.

## Dependencies:

The code relies on the following Python libraries:

streamlit: For creating interactive web applications.
fpdf: For generating PDF documents.
langchain & related sub-packages: For natural language processing tasks like text processing, embedding generation, and retrieval.
pydotenv: For managing environment variables securely.
google-auth & related libraries: For authentication with Google APIs (if using Groq LLM).

# Architecture Breakdown:

## Streamlit App:

The main entry point of the application, providing a user interface with interactive elements like file uploaders, buttons, and text boxes.
Handles user interactions and navigates between different pages based on user choices.

## Language Models:

Hugging Face Embeddings: Used for generating embeddings of text documents, enabling semantic understanding and comparison.
ChatGroq: A large language model from Groq, employed for generating text responses, such as meeting agendas, summaries, and analysis.
Document Processing:

PyPDFLoader: Loads PDF documents into a structured format.
RecursiveCharacterTextSplitter: Splits text into smaller chunks for efficient processing.
Chroma Vectorstore: Stores embeddings of document chunks for fast retrieval.

## Prompt Engineering:

ChatPromptTemplate: Defines prompts for interacting with the language models, providing context and instructions.
MessagesPlaceholder: Used to insert conversation history into prompts.
Retrieval Chains:

create_history_aware_retriever: Combines the language model and retriever to maintain conversation context.
create_retrieval_chain: Creates a chain for retrieving relevant information from the vectorstore based on user queries.

## QA Chains:

create_stuff_documents_chain: Chains the language model and document retriever for question-answering tasks.
Conversation History:

ChatMessageHistory: Stores conversation history to maintain context and enable more coherent responses.

## Additional Modules:

FPDF: For generating PDF documents from generated text.
google_auth & related libraries: For authentication with Google APIs (if using Groq LLM).
whisper: For speech recognition in video transcription.
moviepy: For video editing and audio extraction.
pydub: For audio manipulation.
speech_recognition: For audio transcription.

# Workflow:

User Interaction: The user interacts with the Streamlit interface, uploading documents or videos as needed.
Document Processing: Uploaded documents are processed, split into chunks, and embedded.
Prompt Engineering: Prompts are constructed based on user input and conversation history.
Retrieval: Relevant information is retrieved from the vectorstore based on the prompts.
Language Model Processing: The language model generates text responses based on the retrieved information and prompts.
Output: The generated text is displayed to the user or saved as a PDF.

# Key Components:

Embedding Generation: Creates semantic representations of text for efficient retrieval and comparison.
Prompt Engineering: Carefully crafted prompts guide the language model to generate relevant and informative responses.
Retrieval: Efficiently retrieves relevant information from the vectorstore based on user queries.
Language Model: Generates human-like text responses based on the provided prompts and information.
User Interface: Provides a user-friendly way to interact with the application and view results.

# Explaination Video

https://drive.google.com/file/d/1OyGjb2l2yzZ_DgSQe36a8kC68r0KkJim/view?usp=sharing

(Sorry for the bad quality I had to compress the video under 10mb to upload it on github)

