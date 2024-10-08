
import streamlit as st # library for creating web apps in Python.
from PyPDF2 import PdfReader # library to read and extract text from PDF files.
from langchain.text_splitter import RecursiveCharacterTextSplitter #framework for building applications with language models (used here for text processing and embedding).
import os      #module that provides functions to interact with the operating system (e.g., to load environment variables).
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai # Contains functions and classes to interact with Google’s generative AI APIs.
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv # library to read environment variables from a .env file.

# Load environment variables
load_dotenv() #Loads environment variables from a .env file.
api_key = os.getenv("GOOGLE_API_KEY") #retrieves the API key for Google’s generative AI from environment variables.
genai.configure(api_key=api_key) #This configures the Google generative AI API with the previously retrieved API key, allowing you to make calls to the API.


def get_pdf_text(pdf_docs): # Starts the definition of a function named get_pdf_text that takes a list of PDF documents as input.
    text = "" #nitializes an empty string variable to accumulate the extracted text.
    for pdf in pdf_docs: # Begins a loop to process each PDF document in the list.
        pdf_reader = PdfReader(pdf) #Creates a PdfReader object for the current PDF document, enabling text extraction.
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None return from extract_text
    return text #Returns the complete text extracted from all the provided PDF documents.


def get_text_chunks(text): #Starts the definition of a function named get_text_chunks that takes a string of text as input.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000) #Creates an instance of RecursiveCharacterTextSplitter, specifying a maximum chunk size and overlap between chunks
    chunks = text_splitter.split_text(text)#Splits the input text into manageable chunks.
    return chunks #Returns the list of text chunks created from the input text.


def get_vector_store(text_chunks): #Starts the definition of a function named get_vector_store that takes a list of text chunks as input.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") #Creates an instance of the GoogleGenerativeAIEmbeddings class to generate embeddings from the provided text chunks.
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) #Initializes a FAISS vector store using the text chunks and their corresponding embeddings for efficient similarity searching.
    vector_store.save_local("faiss_index") #Saves the created vector store locally with the name "faiss_index".


def get_conversational_chain(): #def get_conversational_chain():: Begins the definition of a function named get_conversational_chain that creates a question-answering chain.
#prompt_template = """: Starts a multi-line string that defines how to format the question and context for generating responses.
    prompt_template = """
    Answer the question as detailed as possible from the provided context, 
    make sure to provide all the details. If the answer is not in 
    the provided context just say, "answer is not available in the context", 
    don't provide the wrong answer.

    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """
#Creates an instance of the ChatGoogleGenerativeAI model with specified parameters (model name and temperature, which controls randomness).
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#nitializes a prompt template object using the defined prompt_template, specifying the variables it will use.
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#    Loads a question-answering chain using the model and prompt template, allowing it to generate answers based on the context provided.
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question): #Starts the definition of a function named user_input that takes the user’s question as input.
# Creates an instance of the embeddings class for processing the user's question.

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        # Load the vector store with dangerous deserialization enabled
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Perform similarity search
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()

        # Get the response from the chain
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply: ", response["output_text"])
        
    except Exception as e:
        st.error(f"Error loading vector store or generating response: {e}")


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
