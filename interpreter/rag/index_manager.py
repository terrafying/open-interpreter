import sys

from llama_index.node_parser import SimpleNodeParser, HierarchicalNodeParser

from llama_index import VectorStoreIndex
from llama_index import SimpleDirectoryReader
from llama_index.schema import NodeWithScore


class IndexManager(object):

    def __new__(cls):
        """ Make it a Singleton"""
        if not hasattr(cls, 'instance'):
            cls.instance = super(IndexManager, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.stored_message = ""

    def example(self):
        # Populate data with some example stuff
        self.node_parser = SimpleNodeParser.from_defaults()

        self.documents = SimpleDirectoryReader('./docs', recursive=True).load_data()

        self.index = VectorStoreIndex([])
        for doc in self.documents:
            self.index.insert(doc)

        self.nodes = self.node_parser.get_nodes_from_documents(self.documents)

    def insert_document(self, doc):
        self.index.insert(doc)

    def get_stored_message(self):
        return self.stored_message

    def pop_stored_message(self):
        s = self.stored_message
        self.stored_message = ""
        return s

    def set_stored_message(self, message):
        self.stored_message = message

    def get_doc_from_messages(self, messages):
        retriever = self.index.as_retriever()
        retrieved_nodes: list[NodeWithScore] = retriever.retrieve(messages[-1]['message'])

        return '\n'.join([node.text for node in retrieved_nodes])
