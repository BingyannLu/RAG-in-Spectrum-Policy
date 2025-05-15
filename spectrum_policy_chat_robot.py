# -*- coding: utf-8 -*-
"""Spectrum_Policy_Chat-robot.ipynb

## Step A. Load Drive
"""

from google.colab import drive
drive.mount('/content/drive')

"""## Step B. Install All Library"""

!pip install pypdf
!pip install langchain_openai
!pip install --upgrade --quiet  langchain langchain-openai
!pip install typing-extensions==4.8.0
!pip install kaleido
!pip install fastapi
!pip install uvicorn
!pip install python-multipart
!pip install chromadb

import numpy as np
from langchain.document_loaders import PyPDFLoader
import os
import chromadb
import re
import csv
from io import TextIOWrapper
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

"""## Step C. Get API Key"""

# Enter the OpenAI API key

import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# import dotenv

# dotenv.load_dotenv()

"""## Step D. Read Data to Vector Database

### D.1 Pdf Loader - Load data to local variable documents
"""

def pdf_loader_from_folder(pdf_folder_path):
  documents = []
  for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        # print(pdf_path)
        document = loader.load_and_split()
        rmv_zero_page_doc = document
        # if len(document) > 1 and document[1].page_content.startswith("TABLE OF CONTENTS"):
        #   rmv_zero_page_doc = document[1:]
        for page in rmv_zero_page_doc:
          page.metadata['source'] = file
        documents = documents + rmv_zero_page_doc
  return documents

pdf_folder_path = "/content/drive/MyDrive/Spectrum-Policy-Dataset/CSV-files/NTIA_National_Spectrum_Strategy.csv"
# Load PDF files
# After processing, documents list should contain all document chunks
# For every document chunk, overload 'source' as file name and keep page number
documents = pdf_loader_from_folder(pdf_folder_path)

"""### D.2 Add Metadata"""

documents[0].page_content

def find_author_improved(text):
    # Define patterns to search for authors, improved to handle various endings
    patterns = [
        r"submitted\s+BY\s+([A-Za-z \t&–\-]+)",  # Improved pattern for "submitted by"
        r"COMMENTS\s+OF\s+([A-Za-z \t&–\-]+)"    # Improved pattern for "comments of"
    ]

    # Iterate through patterns to find matches
    for pattern in patterns:
        # Ignore case sensitivity
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Return the first captured group, aiming for a cleaner author name
            return match.group(1).strip()  # Strip to remove any leading/trailing whitespace

    return "Author not found"

ans = ""
for doc in documents:
  if doc.metadata['page'] == 0:
    ans = find_author_improved(doc.page_content)
    if ans == "Author not found":
      ans = "Author not found"
  doc.metadata['author'] = ans

def add_metadata(documents, metadata, value):
  for doc in documents:
    doc.metadata[metadata] = value
  return(documents)

documents = add_metadata(documents, "theme", "Partitioning, Disaggregation, and Leasing of Spectrum")

documents[30].metadata

"""### D.3 Output local variable to dataset"""

class PolicyChunkLoader(BaseLoader):
    """Load a `CSV` file into a list of Documents or dump var to a `csv` file

    Store every document chunk into the file row by row.
    The CSV Format looks like below:
    page__content  |source  |page|topic
    Page Content  |file__name|page|policy
    """

    def __init__(
        self,
        file_path: str,
        source_column: str = "page_content",
        metadata_columns: Sequence[str] = ["page", "source"],
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ):
        """

        Args:
            file_path: The path to the CSV file.
            source_column: The PageContent column name.
            metadata_columns: A sequence of column names to use as metadata.
            csv_args: A dictionary of arguments to pass to the csv.DictReader.
              Optional. Defaults to None.
            encoding: The encoding of the CSV file. Optional. Defaults to None.
            autodetect_encoding: Whether to try to autodetect the file encoding.
        """
        self.file_path = file_path
        self.source_column = source_column
        self.metadata_columns = metadata_columns
        self.encoding = encoding
        self.csv_args = csv_args or {}
        self.autodetect_encoding = autodetect_encoding

    def _generate_page_description(self, author):
      return "{} is writing: ".format(author)

    def dump(
        self,
        documents : List[Document]
    ) -> bool:
        with open(self.file_path, 'w', newline='') as csvfile:
          fieldnames = [self.source_column, *self.metadata_columns]
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writeheader()
          for raw in documents:
            raw_dict = {self.source_column: raw.page_content}
            raw_dict.update({col : raw.metadata[col] for col in self.metadata_columns})
            try:
              writer.writerow(raw_dict)
            except:
              print(raw_dict['page'], raw_dict['source'])


    def load(self, apply_metadata_to_content = False) -> List[Document]:
        """Load data into document objects."""

        docs = []
        try:
            with open(self.file_path, newline="", encoding=self.encoding) as csvfile:
                docs = self.__read_file(csvfile, apply_metadata_to_content)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    try:
                        with open(
                            self.file_path, newline="", encoding=encoding.encoding
                        ) as csvfile:
                            docs = self.__read_file(csvfile, apply_metadata_to_content)
                            break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        return docs

    def __read_file(self, csvfile: TextIOWrapper, apply_metadata_to_content = False) -> List[Document]:
        docs = []

        csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
        for i, row in enumerate(csv_reader):
            metadata = {}
            for meta_col in self.metadata_columns:
                try:
                    metadata[meta_col] = row[meta_col]
                except KeyError:
                    raise ValueError(f"Metadata column '{meta_col}' not found in CSV file.")
            try:
                page_content = row[self.source_column]
                if apply_metadata_to_content:
                  page_content = self._generate_page_description(metadata['author'])
            except KeyError:
                raise ValueError(
                    f"Source column '{self.source_column}' not found in CSV file."
                )
            doc = Document(page_content=page_content, metadata=metadata)
            for col in self.metadata_columns:
              doc.metadata[col] = metadata[col]
            docs.append(doc)

        return docs

DATASET_PATH = "/content/drive/MyDrive/Spectrum-Policy-Dataset/CSV-files/fcc-23-232.csv"

documents[-1]

# Download
chunk_loader = PolicyChunkLoader(DATASET_PATH, metadata_columns=["page", "source", "author", "theme"])
chunk_loader.dump(documents)

# Upload
loader = PolicyChunkLoader(file_path=DATASET_PATH, metadata_columns=["page", "source", "author", "theme"])

documents = loader.load()
documents[0]

"""### D.4 Output dataset to DB Vector"""

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
# raw_documents = TextLoader('../../../state_of_the_union.txt').load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# db.delete_collection()
db = Chroma.from_documents(documents, OpenAIEmbeddings())

query = "What is ATT's opinion?"
docs = db.similarity_search(query, 20)
for doc in docs:
  print(doc.metadata)

documents[0]

"""## Step E. Algorithm Exploration"""

embeddings = OpenAIEmbeddings()

# Embedding Content
texts = [doc.page_content for doc in documents]
content_vector_matrix = np.array(embeddings.embed_documents(texts))

# Embedding Metadata
metadata_author = [doc.metadata['author'] for doc in documents]
author_vector_matrix = np.array(embeddings.embed_documents(metadata_author))

def sum_of_squares(arr):
  return np.square(arr).sum(axis=1)
def distance_from_metadata_and_content(query, author_vector_matrix, content_vector_matrix,w_content,w_author):
  query_vector = np.array(embeddings.embed_query(query))
  distance_author = author_vector_matrix - query_vector
  distance_content = content_vector_matrix - query_vector
  ret = w_author*sum_of_squares(distance_author) + w_content*sum_of_squares(distance_content)
  return ret

def nearest_docs_idx(query, distance_function, author_vector_matrix, content_vector_matrix,w_content,w_author,  n) -> List[int]:
  """
  Given query, find nearest documents indexs based an distance_function
  Args:
    query: query sentence
    distance_function: A function takes query as input and returns an distance array for each document.
    n: number of documents to return
  Returns:
    The nearest documents indexes
  """
  distance_array = distance_function(query, author_vector_matrix, content_vector_matrix,w_content,w_author)
  sorted_index_array = np.argsort(distance_array)[0:n]
  return sorted_index_array

"""### E.1 Metadata and Content Weighted Average Distance"""

query = "What is ATT's opinion?"
related_docs_idx = nearest_docs_idx(query, distance_from_metadata_and_content, author_vector_matrix, content_vector_matrix, 0.7,0.3, 20)

related_docs = [documents[idx] for idx in related_docs_idx]

for a in related_docs:
  print(a.metadata)

"""## Step F. Conversation chains

The first time iteration
"""

# pip install -U langchain langchain-community

from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms.fake import FakeStreamingListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from operator import itemgetter

system_msg_1 = "Try your best to answer the problem.:\n\n{context}"
system_msg_2 = "Assume you are another filer. Write a counter argument for the passage. Don't mention any filer's name. Just generate the opposite opinion.:\n\n{context}"
system_msg_3 = """
If the information in the document contains the answer, you will give an accurate answer. If you can not get answer from the document, you will generate
'I can not answer the question because of the insufficient information in documents.':\n\n{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [("system", system_msg_1), ("human", query)]
)

# prompt = ChatPromptTemplate.from_messages(
#     [("system", system_prompt)]
# )

llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09")
chain = create_stuff_documents_chain(llm, prompt)

docs = related_docs

ans = chain.invoke({"context": docs})
ans

"""Second Iteration"""

related_docs_idx_2 = nearest_docs_idx(ans, distance_from_metadata_and_content, author_vector_matrix, content_vector_matrix,0.8,0.2,20)
related_docs_2 = [documents[idx] for idx in related_docs_idx_2]

for a in related_docs_2:
  print(a.metadata)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_msg_1), ("human", query)]
)

llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09")
chain = create_stuff_documents_chain(llm, prompt)

docs = related_docs_2

chain.invoke({"context": docs})

"""##  Expired code (abandoned)

### Build retriever for Langchain
"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
from typing import Callable
class CustomizedRetriever(VectorStoreRetriever):
    """Customized Retriever class for VectorStore."""
    number_to_retrieve :int
    algorithm : Callable
    author_vector_matrix :np.ndarray
    content_vector_matrix :np.ndarray

    def __init__(self, vectorstore, algorithm, author_vector_matrix, content_vector_matrix, k, **kwargs):
      super().__init__(vectorstore = vectorstore, **kwargs)
      self.number_to_retrieve = k
      self.algorithm = algorithm
      self.author_vector_matrix = author_vector_matrix
      self.content_vector_matrix = content_vector_matrix

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        related_docs = nearest_docs_idx(query, self.algorithm, self.author_vector_matrix, self.content_vector_matrix,  self.number_to_retrieve)
        return related_docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:

        return self._get_relevant_documents(docs)

"""### Query from LLM"""

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
NUMBER_CHUNKS_TO_RETRIEVE = 10

llm = ChatOpenAI(model_name='gpt-4-1106-preview', temperature=0.0)
memory = ConversationSummaryBufferMemory(
    llm=llm, memory_key="chat_history", max_token_limit=10, return_messages=True
)

retriever = CustomizedRetriever(db, distance_from_metadata_and_content, author_vector_matrix, content_vector_matrix, NUMBER_CHUNKS_TO_RETRIEVE)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

"""### Combine datasource and LLM answer"""

import pprint
pp = pprint.PrettyPrinter(indent=4)
class PolicyRobot:
  _conversation_chain = None
  _db = None
  def __init__(self, conversation_chain, db):
    self._conversation_chain = conversation_chain
    self._db = db

  def ask(self, query):
    llm_ret = self._conversation_chain(query)
    db_ret = self._db.similarity_search(query, NUMBER_CHUNKS_TO_RETRIEVE)
    pp.pprint(llm_ret)
    pp.pprint("===============================================")
    pp.pprint("Datasource:")
    for doc in db_ret:
      pp.pprint(doc.metadata)

my_robot = PolicyRobot(qa, db)

"""### Conversation Test"""

my_robot.ask("Could you summarize AT&T's comment")

db.get(where={"source": {"$in" : ["NCTA NSF NOI - Reply Comments.pdf"]}})

"""### Set up UI"""

!git clone https://github.com/homanp/langchain-ui.git

!cd langchain-ui

!npm install

"""1.  

Q. Which organization focus on the transaction cost of the regarding partitioning, disaggregation, and leasing rules proposed?
  
A. R street cares and Google about the transaction cost.

2.

Q. Who's against R street optinion on performance requirements.

A. CTIA. CTIA supports the performance requirements shoulnb't be lowered while R street thinks it's necessary to remove the requirements to reduce transaction costs.

3.

Q. Who wants to clarify the definition of carrier and benefit from leasing proposal.

A. MIDCO wants a broader carrier definition to also benifit from the proposal
"""
