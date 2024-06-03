model_path = "THUDM/chatglm-6b"
endpoint_url = "http://0.0.0.0:8080"

txt_dir = "txts_for_langchain"

from langchain_community.document_loaders import directory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import chatglm
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
import os


def load_docs(dir = txt_dir):
    # load txt
    loader = directory.DirectoryLoader(dir)
    docs = loader.load()

    # split txt
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(docs)
    return split_docs


embedding_model_dict = {
    "ernie-tiny" : "nghuvong/ernie-3.0-nano-zh",
    "ernie-base" : "nghuvong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2" : "uer/sbent-base-chinese-nli",
    "text2vec3" : "shibing624/text2vec-base-chinese",
}

def load_embedding_model(model_name):
    encode_kwargs={"normalize_embeddings":False}
    model_kwargs={"device":"cuda:0"}
    return HuggingFaceEmbeddings(
        model_name = embedding_model_dict[model_name],
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

embeddings = load_embedding_model("text2vec3")

# vector store
def store_chroma(docs, embeddings, persist_directory):
    db = chroma.Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db

if not os.path.exists('chroma_data'):
    docs = load_docs()
    db = store_chroma(docs,embeddings,"chroma_data")
    print("record docs in vector store")
else:
    db = chroma.Chroma(persist_directory="chroma_data", embedding_function=embeddings)
    print(db.__sizeof__)

# connect to model api
llm = chatglm.ChatGLM(
    endpoint_url = endpoint_url,
    max_token = 80000,
    top_p = 0.9
)

#db test
# query = "芙莉莲的魔法"
# docs = db.similarity_search(query, k=3) # default k is 4
# print(len(docs))
# for doc in docs:
#     print("="*100)
#     print(doc.page_content)


retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

response = qa.run('芙莉莲喜欢吃什么')
print(response)
