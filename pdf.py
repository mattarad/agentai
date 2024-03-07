from dotenv import load_dotenv

import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader

load_dotenv()


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name))

    return index

canada_pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = PDFReader().load_data(file=canada_pdf_path)
canada_index = get_index(canada_pdf, "canada")
canada_engine = canada_index.as_query_engine()

united_states_pdf_path = os.path.join("data", "United_States.pdf")
united_states_pdf = PDFReader().load_data(file=united_states_pdf_path)
united_states_index = get_index(united_states_pdf, "united_states")
united_states_engine = united_states_index.as_query_engine()