
from llama_index.node_parser import SimpleNodeParser

from llama_index import VectorStoreIndex
from llama_index import SimpleDirectoryReader
from llama_index.schema import NodeWithScore

parser = SimpleNodeParser.from_defaults()

documents = SimpleDirectoryReader('./data').load_data()

index = VectorStoreIndex([])
for doc in documents:
    index.insert(doc)

nodes = parser.get_nodes_from_documents(documents)

def insert_document(doc):
    index.insert(doc)

def get_doc_from_messages(messages):
    retriever = index.as_retriever()
    retrieved_nodes: list[NodeWithScore] = retriever.retrieve(messages[-1]['message'])

    return '\n'.join([node.text for node in retrieved_nodes])