from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter
import streamlit as st
import tempfile
import os
import pandas as pd

api_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = api_creds['gemini_api']

st.set_page_config(page_title="File QA Assistant")
st.title("Welcome to PDF QA Chatbot")

@st.cache_resource(ttl=3600)
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(docs)

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print(embeddings_model.embed_query("hello"))
    vectordb = Chroma.from_documents(doc_chunks, embeddings_model)

    retriever = vectordb.as_retriever()
    return retriever

class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)

uploaded_files = st.sidebar.file_uploader(
   label = "Upload PDF files", type=["pdf"],
   accept_multiple_files = True   
)
if not uploaded_files:
   st.info("Please upload PDF documents to continue")
   st.stop()

retriever = configure_retriever(uploaded_files)

gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)

qa_template = """
                Use only the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know,
                don't try to make your an answer. Keep the answer as concise as possible.
                
                {context}
                
                Question: {question}
                """

qa_prompt = ChatPromptTemplate.from_template(qa_template)

def format_docs(docs):
   return "\n\n".join([d.page_content for d in docs])

qa_rag_chain = (
   {
      "context":itemgetter("question")
      |
      retriever
      |
      format_docs,
      "question":itemgetter("question")
   }
   |
   qa_prompt
   |
   gemini_model
)

streamlit_msg_history = StreamlitChatMessageHistory()

for msg in streamlit_msg_history.messages:
   st.chat_message(msg.type).write(msg.content)

class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg: st.write): # type: ignore
      BaseCallbackHandler.__init__(self)
      self.msg = msg
      self.sources = []
    
    def on_retriever_end(self, documents, *, run_id, parent_run_id = None, **kwargs):
       source_ids = []
       for d in documents:
          metadata = {
             "source": d.metadata["source"],
             "page": d.metadata["page"],
             "content":d.page_content[:200]
          }
          idx = (metadata["source"], metadata["page"])
          if idx not in source_ids:
             source_ids.append(idx)
             self.sources.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id = None, **kwargs):
       if len(self.sources):
          st.markdown("__Sources:__"+"\n")
          st.dataframe(data=pd.DataFrame(self.sources[:3]),width=1000)

if user_prompt := st.chat_input():
  st.chat_message("human").write(user_prompt)
  with st.chat_message("ai"):
      stream_handler = StreamHandler(st.empty())
      sources_container = st.write("")
      pm_handler = PostMessageHandler(sources_container)
      config = {"configurable":{"session_id":"any"},
              "callbacks":[stream_handler]}
      response = qa_rag_chain.invoke({"question": user_prompt}, config)  
      st.write(str(response.content))
      streamlit_msg_history.add_user_message(user_prompt)
      streamlit_msg_history.add_ai_message(response.content)