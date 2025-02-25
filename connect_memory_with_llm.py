import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# setup llm (mistral with huggingface)

HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
      repo_id=huggingface_repo_id,
      temperature=0.5,
      model_kwargs={"token":HF_TOKEN,
                    "max_length":"512"}
    )
    return llm
# connect llm with faiss and create chain

custom_prompt_template = """
Use the piece of information provided in the context to answer the user's question. If you dont know the answer, just say you dont know, dont try to make up an answer.add()Dont provide anything out of the given context

Context : {context}
Question : {question}

Start the answer directly. No small talk please

"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

#load  Database

DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)


# create QA chain

qa_chain= RetrievalQA.from_chain_type(
  llm = load_llm(huggingface_repo_id),
  chain_type="stuff",
  retriever = db.as_retriever(search_kwargs={'k':3}),
  chain_type_kwargs={'prompt':set_custom_prompt(custom_prompt_template)}
)


# Now invoke with the simple query

user_query = input("write Query here")
response = qa_chain.invoke({'query':user_query})
print("RESULT:",response["result"])
