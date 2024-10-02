from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/
# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/
# https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/ vector store. pip install langchain-chroma

loader = PyPDFLoader("INSERT A PATH TO PDF")

if __name__ == '__main__':
    docs = loader.load_and_split()

    hf_embedding = HuggingFaceEmbeddings()

    db = Chroma.from_documents(docs, hf_embedding)

    # run ollama list
    model = Ollama(model="llama3.1")

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=db.as_retriever(),
    )

    question = "<ASK YOUR QUESTION>?"

    result = qa_chain.invoke({"query": question})

    print(result['result'])
