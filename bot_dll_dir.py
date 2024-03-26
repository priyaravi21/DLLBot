import streamlit as st
import os
import pandas as pd
import chromadb
import io
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azureml.core import Workspace, Dataset
from dotenv import load_dotenv
load_dotenv()



# TO DO: set as env variables to hide values

deployment_name = '3-5-turbo'
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
connect_str = os.getenv('connect_str')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
# os.environ["OPENAI_PROXY"] = "http://your-corporate-proxy:8080"


enable_memory = True
enable_prompt = True
enable_source = True
enable_web_RE = False


container_name="bot-dll-pdf"
if connect_str is not None:
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)

else:
    st.warning("Connect str not found")



folder_name = "PA_DLL.xlsx"


def load_excel_from_blob():
    try:
        # Iterate through the blobs in the specified folder
        for blob in container_client.walk_blobs(name_starts_with=folder_name):
            # Download the blob content
            blob_client = BlobClient.from_connection_string(connect_str, container_name, blob.name)
            blob_data = blob_client.download_blob().readall()
            df = pd.read_excel(io.BytesIO(blob_data))

            return df

    except Exception as ex:
        st.warning(f"An error occurred: {ex}")

@st.cache_data 
def search_and_display_results(query_text):
    df = load_excel_from_blob()
    client = chromadb.Client()
    collection = client.create_collection("collect-my-documentss")
    
    collection.add(documents=df["Content"].tolist(), ids=df["Link"].tolist())
    relevant_files = []
    results = collection.query(query_texts=[query_text], n_results=1, include=['distances', 'metadatas'])
    relevant_filenames = results["ids"]
    for sublist in relevant_filenames:
        for filename in sublist:
            relevant_files.append(filename)  # Collect relevant files
            st.write(filename)
    filter = df[df['Link'].str.contains(filename)]
    
    st.dataframe(filter)
    text_data = filter['Content']
    client.delete_collection('collect-my-documentss')
    return text_data, relevant_files  # Return text data and relevant files

def search_from_db(text_data):
    #st.write(text_data)
    print("Inside search_from_db function")  # Debugging statement
    
    # Split the plain text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
    docs = text_splitter.create_documents(text_data)
    embeddings = OpenAIEmbeddings(deployment="embedding")
    #st.write(embeddings)
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Define the prompt template
    prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. Answer only using the data fed. Answer very precisely and keep the answers as few words as possible. if your answer is too long, reply in bullet in points 

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(prompt_template)

    # Initialize memory and model objects
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    model = AzureChatOpenAI(deployment_name=deployment_name, temperature=0)
    
    # Create ConversationalRetrievalChain with prompt template
    chat_history = [] # [("role:system","content:You are an AI assistant that helps people find information.")]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    model = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=0
    )

    if enable_memory & enable_prompt:
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=enable_source,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            condense_question_llm=model
        )
    elif enable_memory:
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=enable_source,
        )
    elif enable_prompt:
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=vectorstore.as_retriever(),
            return_source_documents=enable_source,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            condense_question_llm=model
        )
    else:
        qa = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=vectorstore.as_retriever(),
            return_source_documents=enable_source,
        )
    return qa

def main():
    st.set_page_config(page_title="One stop solution for learning", page_icon="💬", layout="centered", initial_sidebar_state="auto", menu_items=None)

    st.title("Lessons Navigator")
    session_state = st.session_state

    if 'text_data' not in session_state:
        session_state.text_data = None
    if 'qa' not in session_state:
        session_state.qa = None
    if 'relevant_files' not in session_state:
        session_state.relevant_files = None
    if 'chat_history' not in session_state:
        session_state.chat_history = []
    if 'last_answered_query_num' not in session_state:
        session_state.last_answered_query_num = 0
    if 'question_answers' not in session_state:
        session_state.question_answers = []  # List to store question-answer pairs

    query_text = st.text_input("Search by keyword:")
    
    if st.button('search'):
        session_state.text_data, session_state.relevant_files = search_and_display_results(query_text)
        session_state.qa = search_from_db(session_state.text_data)
        
    if session_state.text_data is not None and session_state.qa is not None:
        st.sidebar.write("Relevant files:")
        if session_state.relevant_files:
            for file in session_state.relevant_files:
                st.sidebar.write("File Link:", file)  # Display file links in sidebar

        enable_memory = True
        # Set enable_memory to True to enable memory feature
        
        # Infinite loop for handling questions
        query_num = session_state.last_answered_query_num + 1  # Increment query number
        query_variable_name = f"query{query_num}"
        query_key = f"question_input_{query_num}_{query_variable_name}"  # Generate unique key
        
        # Display previous questions and answers
        for qa_pair in session_state.question_answers:
            st.markdown(f"**{qa_pair[0]}**")
            st.markdown(f"**Answer:** {qa_pair[1]}")
        
        exec(f"{query_variable_name} = st.text_input(f'Ask your question :', key='{query_key}')")
            
        button_key = f"button_{query_num}"  # Generate unique key for button
        if st.button("Ask", key=button_key):
            query = eval(query_variable_name)  # Get the value of the current query variable
            if enable_memory:
                result = session_state.qa({"question": query})
            else:
                result = session_state.qa({"question": query, "chat_history": session_state.chat_history})
                session_state.chat_history.append((query, result["answer"]))  # Append the query and answer to chat_history

            st.markdown(f"**Your question:** **{query}**")
            st.markdown(f"**Answer:** {result['answer']}")
                
            # Store the question and answer in session state
            session_state.question_answers.append((query, result["answer"]))
                
            # Update the last answered query number
            session_state.last_answered_query_num = query_num

if __name__ == "__main__":
    main()




