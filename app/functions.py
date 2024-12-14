# Backend llm calculations (all frontend code is in streamlit_app.py)
""" 
This python file is equivalent to the notebook data_extraction.ipynb except that it is imported into streamlit_app.py
also i added a rag chain equivalent to the one in the notebook, i also used some os and tempfile modules to work with the files 
also i added uuid to unique ids for the vector store text embeddings to avoid duplicate embeddings from the overlapping chunks
"""

# all imports (see notebook for more info)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import os # to delete the temporary file for temporary file
import tempfile # to create a temporary file for PyPDFLoader
import uuid # to create a unique id for the temporary file for PyPDFLoader
import pandas as pd # to convert the pdf to a dataframe and then to a table
import re # to clean the text  

# used to clean up the file name  (as we need it for the vector store)
def clean_filename(filename):
    # NOTE:
    # the following is done because the filename is used as the collection name in the vector store
    # and it must follow the rules of a string in a vector store
    
    # Remove spaces and replace with underscores
    filename = filename.replace(" ", "_")
    # Remove any characters that are not alphanumeric, underscore, or hyphen
    filename = re.sub(r'[^a-zA-Z0-9_-]', '', filename)
    # Remove consecutive periods (if any)
    filename = re.sub(r'\.{2,}', '.', filename)
    # Ensure the filename does not start or end with a period
    filename = filename.strip('.')
    # Ensure filename length is between 3 and 63 characters 
    if len(filename) < 3:
        filename = 'default_name'
    elif len(filename) > 63:
        filename = filename[:63]
    # Return the cleaned filename
    return filename

# loading the text from the pdf file
def get_pdf_text(uploaded_file): 
    try: # to handle errors 
        # Read file content
        input_file = uploaded_file.read()

        # Create a temporary file (PyPDFLoader requires a file path to read the PDF hence we need to create a temporary file)
        # it can't work directly with file-like objects or byte streams so we must pass in the actual file
        # in the Streamlit app the uploaded_file is converted to a byte stream to be displayed not here
        temp_file = tempfile.NamedTemporaryFile(delete=False) # delete=False to keep the file after it is closed 
        temp_file.write(input_file) # write the content of the uploaded file to the temporary file
        temp_file.close() # close the temporary file

        # load PDF document using PyPDFLoader
        loader = PyPDFLoader(temp_file.name) # pass the path to the temporary file
        documents = loader.load() # load the pdf document

        return documents # return the loaded pdf document
    
    finally: # to delete the temporary file once we are done
        os.unlink(temp_file.name) # delete the temporary file

# splitting the text into chunks
def split_document(documents, chunk_size, chunk_overlap): 
    # split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap, # overlap between chunks
                                          length_function=len, # function to determine the length of a chunk
                                          separators=["\n\n", "\n", " "]) # separators to split on here we split on new linee pages and spaces
    
    # return the chunks
    return text_splitter.split_documents(documents)

# get the embedding function from the OpenAI API
def get_embedding_function(api_key):
    # create embedding function
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=api_key # using the text-embedding-ada-002 model 
    )
    # return the embedding function
    return embeddings

# creating the vector store database for the chunks to be stored and compared
def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    # this avoids duplicate documents in the vector store for overlapping chunks
    unique_ids = set() # create a set to store unique ids a set can't have duplicates
    unique_chunks = [] # create a list to store unique chunks, we check the id before we add it so we dont need a set here
    
    for chunk, id in zip(chunks, ids): # iterate over the chunks and their ids, we use zip to map each chunk to its id
        if id not in unique_ids: # if the id is not in the set
            unique_ids.add(id) # add the id to the set
            unique_chunks.append(chunk) # add the chunk to the list

    # Create a new Chroma database from the documents chunks and use the embedding function
    vectorstore = Chroma.from_documents(documents=unique_chunks, # pass in the unique chunks
                                        collection_name=clean_filename(file_name), # clean the file name to use it as the collection name
                                        embedding=embedding_function, # pass in the embedding function
                                        ids=list(unique_ids), # pass in the unique ids
                                        persist_directory = vector_store_path) # pass in the vector store path

    # The database should save automatically after we create it
    # but we can also force it to save using the persist() method
    vectorstore.persist()
    
    # return the vector store database
    return vectorstore

# creating the vector store from our pdf text 
# this is where we call the split_document to split the pdf into chunks and the pass that into the vector store along with the embedding function
# all our previous helper functions are called here, but since we need a new db for each pdf we call this function in streamlit app file
def create_vectorstore_from_texts(documents, api_key, file_name):
    # split the text into chunks
    docs = split_document(documents, chunk_size=1000, chunk_overlap=200)
    
    # define embedding function
    embedding_function = get_embedding_function(api_key)

    # create a vector store now that we have the text and embedding function to pass into the vector store
    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    
    # return the vector store now filled with the pdf text
    return vectorstore


# Prompt template (little different from the original in notebook)
PROMPT_TEMPLATE = """
You are an assistant for answering questions about a research paper.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING!

{context}

---

Answer the question based on the above context: {question}
"""

# class to get sources and resoning for our answer
class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    
# class to pass in the sources and resoning for our answer to the question about title, summary, publication date and author
class ExtractedInfoWithSources(BaseModel):
    """Extracted information about the research article"""
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources

# function to format the doc passed in into a single string joined by two newlines
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# in this function we create the prompt template and the rag chain and pass in out question to the llm
# retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings;
# RunnablePassthrough() passes through the input question unchanged.
def query_document(vectorstore, query, api_key): # pass in the vector store and the question to be asked along with the api key
    # define llm
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    # define retriever (gets text chunks from vector store based on similarity)
    retriever=vectorstore.as_retriever(search_type="similarity")
    # define prompt template with context and question
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # define rag chain, this calls the retriever to get context and then passes the context and question into the prompt template
    # then finally passes the prompt template into the llm with structured output, struct defined as ExtractedInfoWithSources this is strict
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
        )
    # call the rag chain to get the structured response, this is where we call the llm
    structured_response = rag_chain.invoke(query)
    
    # with this new structured response we can create a table to organize our data
    # we use pandas to create a dataframe and pass in out the structured response as a dictionary
    df = pd.DataFrame([structured_response.dict()])

    # Transforming into a table with 3 rows onw for each field in the answer
    answer_row = []
    source_row = []
    reasoning_row = []

    # for each column in the dataframe i.e for each question we ask we have 3 rows for the answer, sources and reasoning for that question
    for col in df.columns:
        answer_row.append(df[col][0]['answer']) # we extract the answer from the df and add it to the answer row
        source_row.append(df[col][0]['sources']) # do the same for the sources
        reasoning_row.append(df[col][0]['reasoning']) # do the same for the reasoning

    # Create new dataframe with 3 rows we will add the created rows to this dataframe table
    # we pass in our rows as we have created them above, pass in our columns as they are in the dataframe and a list of indexes to act as headers i.e column names
    structured_response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning'])
  
    # return the structured response dataframe trasposed i.e rows become columns, for readability
    return structured_response_df.T
