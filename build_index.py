import pickle
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode


if __name__ == "__main__":

    documents_dir = "./documents"
    index_dir = "./data/index"
    dict_file_path = "./data/all_nodes_dict.pkl"

    storage_context = StorageContext.from_defaults()
    documents = SimpleDirectoryReader(documents_dir).load_data()

    node_parser = SentenceSplitter(chunk_size=1024)
    base_nodes = node_parser.get_nodes_from_documents(documents)

    sub_chunk_sizes = [128]
    sub_node_splitters = [
        SentenceSplitter(chunk_size=c, chunk_overlap=10) for c in sub_chunk_sizes
    ]

    all_nodes = []
    for base_node in base_nodes:
        for splitter in sub_node_splitters:
            sub_nodes = splitter.get_nodes_from_documents([base_node])
            sub_index_nodes = [
                IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_index_nodes)

        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)

    all_nodes_dict = {n.node_id: n for n in all_nodes}

    with open(dict_file_path, 'wb') as f:
        pickle.dump(all_nodes_dict, f)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    storage_context.persist(index_dir)