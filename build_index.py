from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext
)


if __name__ == "__main__":

    documents_dir = "./documents"
    index_dir = "./storage"

    storage_context = StorageContext.from_defaults()
    documents = SimpleDirectoryReader(documents_dir).load_data()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    storage_context.persist(index_dir)