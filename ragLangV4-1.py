from dotenv import load_dotenv
import os
import json
import time
import sys
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import StorageContext
from langchain_pinecone import PineconeVectorStore 
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain_openai import OpenAIEmbeddings
from llama_index.readers.json import JSONReader
from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA

#added import
from langchain_community.chat_message_histories import ChatMessageHistory
from llama_index.core.schema import Document
from llama_index.llms.predibase import PredibaseLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Predibase
from llama_index.core import VectorStoreIndex
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from typing import List

#from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from datetime import datetime

from prompts import DOCUMENT_PROMPT, LLM_CONTEXT_PROMPT
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Set environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ["PREDIBASE_API_TOKEN"] = PREDIBASE_API_TOKEN

# Check if the correct number of arguments are provided
if len(sys.argv) != 3:
    print("Usage: python myScript.py <pinecone_index_name> <folder_name>")
    sys.exit(1)
# Access the arguments
#script_name = sys.argv[0]  # The name of the script
pinecone_index_name = sys.argv[1]         # The first argument
folder_name = sys.argv[2]         # The second argument


def format_datetime_as_number(datetime_str: str) -> int:
    date_obj = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%SZ')
    formatted_datetime = date_obj.strftime('%Y%m%d%H%M%S')
    return int(formatted_datetime)

def get_metadata(document_segment, JSONobject, document_id, json_key):
    obj = JSONobject

    if "_MAIN_DOC" not in document_id:
        date_number = format_datetime_as_number(obj[json_key]['publishedAt'])
        captions = document_segment.split()
        metadata = {
            "document_id": document_id,
            "video_id": obj[json_key]['videoId'],
            "title": obj[json_key]['title'],
            "words_in_title": obj[json_key]['title'].split(),
            "date": date_number,
            "view_count": obj[json_key]['stats']['viewCount'],
            "comment_count": obj[json_key]['stats']['commentCount'],
            "captions": captions, #array of words in the document_segment for query purposes. Match up to time stamps later
            "text": document_segment,
            "MAIN_DOC": False,
        }
        if 'likeCount' in obj[json_key]['stats']:
            metadata["like_count"] = obj[json_key]['stats']['likeCount']
    else:
        metadata = {
            "document_id": document_id,
            "channel_information": json.dumps(obj[json_key]['channelInformation']),
            "text": document_segment,
            "MAIN_DOC": True,
        }
    
    return metadata

def split_document(document, JSONobject, max_size):
    chunks = []
    current_chunk = {}
    current_size = 0
    count = 1

    words = document.split()
    json_key = words[0]

    for word in words:
        word_size = len(word.encode('utf-8'))
        
        if current_size + word_size > max_size:  ### +1 No longer needed; no leading space; +1 for the leading space stripped later
            current_chunk["metadata"] = get_metadata(current_chunk[key], JSONobject, key, json_key)
            chunks.append(current_chunk)
            count += 1
            current_chunk = {}
            current_size = 0

        key = f"{json_key}#chunk{count}"
        
        if key in current_chunk:
            current_chunk[key] += ' ' + word
            current_size += word_size + 1
        else:
            current_chunk[key] = word
            current_size += word_size
    
    if current_chunk:
        current_chunk["metadata"] = get_metadata(current_chunk[key], JSONobject, key, json_key)
        chunks.append(current_chunk)
    
    return chunks

def split_documents(documents, JSONobjects, max_size):
    all_chunks = []
    for i, document in enumerate(documents):
        all_chunks.extend(split_document(document, JSONobjects[i], max_size))
    return all_chunks


def main():
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

    index_name = pinecone_index_name #code for manual input:input('Enter index name: ')
    model_name = 'text-embedding-ada-002'  
    embeddings = OpenAIEmbeddings(  
        model=model_name,  
        openai_api_key=os.environ['OPENAI_API_KEY']  
    )

    # Check if the index exists and create if it doesn't
    if index_name not in [index['name'] for index in pc.list_indexes()]:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name, 
            dimension=1536,
            metric='dotproduct',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # Wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(1)
        print(f"Index '{index_name}' created and ready.")

    # Construct vector store and custom storage context
    #pincone_vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, text_key="title")
    #pinecone_storage_context = StorageContext.from_defaults(vector_store=pincone_vector_store)

    # Load documents
    input_dir = folder_name #code for manual input:input('Enter directory name: ')
    documents = []
    JSONobjects = []
    reader = JSONReader(levels_back=0)

    # Loop through files in directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            input_file = os.path.join(input_dir, file_name)
            # Open and read the JSON file
            with open(input_file, 'r') as file:
                obj = json.load(file)
                JSONobjects.append(obj)
            documents.extend(reader.load_data(input_file=input_file, extra_info={}))

    print(f"Loaded {len(documents)} documents from '{input_dir}'.")
    print(f"Loaded {len(JSONobjects)} JSON objects from '{input_dir}'.")

    # Split documents into chunks to be within maximum file size
    max_chunk_size = 10240  # Maximum size in bytes
    chunked_documents = split_documents([doc.text for doc in documents], JSONobjects, max_chunk_size)
    print(f"Seperated {len(documents)} documents into {len(chunked_documents)} chunks.")

    index = pc.Index(index_name)
    index_stats = index.describe_index_stats()
    vectorsNeedUpsertion = True if index_stats.total_vector_count != len(chunked_documents) else False
    print(f"chunks({len(chunked_documents)}) = total_vector_count({index_stats.total_vector_count})? --> vectorsNeedUpsertion: {vectorsNeedUpsertion}")

    # vectorsNeedUpsertion = False ## A few missing vectors not loading
    # print(f"Missing vectors not loading --> Override: vectorsNeedUpsertion: {vectorsNeedUpsertion}")

    if vectorsNeedUpsertion:
        # Convert chunked documents to Document class instances; from  Module - llama_index.core.schema, Class - Document
        llama_docs = [Document(text=list(chunk.values())[0]) for chunk in chunked_documents]
        # Convert documents to LangChain format
        docs = [doc.to_langchain_format() for doc in llama_docs]
        print(f"Formatted {len(docs)} chunks.")

        # Extract texts from documents
        texts = [doc.page_content for doc in docs]

        # Generate embeddings for the texts
        embeddings_list = embeddings.embed_documents(texts) ####Change to docs

        # Assign IDs and prepare vectors for upsert
        vectors = []
        for i, embedding in enumerate(embeddings_list):
            doc_id = f"{list(chunked_documents[i].keys())[0]}"#Uses unique chunk keys as ID
            # Prepare metadata manually
            metadata = chunked_documents[i]['metadata']
            # Prepare vector in the required format
            vector = {"id": doc_id, "values": embedding, "metadata": metadata}
            vectors.append(vector)
        print(f"Prepared {len(vectors)} vectors for upsert.")

        #function to split up upsert http message size
        def vector_list(data, num_of_vectors):
            for i in range(0, len(data), num_of_vectors):
                yield data[i:i + num_of_vectors]
        # Upsert vectors to Pinecone index
        for vects in vector_list(vectors, 100):
            upsert_response = index.upsert(vects)
            print("Upsert response:", upsert_response)

        # Verify the index's new stats
        print("Index stats:", index_stats)



    # metadata_field_info = [
    #     {
    #         "name": "date",
    #         "description": "The date the youtube video was published in YYYYMMDDHHMMSS format",
    #         "type": "integer",
    #     },
    #     {
    #         "name": "document_id",
    #         "description": "The document ID inside the vector store. The ID for a Video document is formatted like this: video_id + #chunk number. The ID for a MAIN_DOC document is formatted like this: channelTitile + _MAIN_DOC + #chunk number.",
    #         "type": "string",
    #     },
    #     {
    #         "name": "view_count",
    #         "description": "View count from Youtube video's stats.",
    #         "type": "integer",
    #     },
    #     {
    #         "name": "like_count",
    #         "description": "Like count from Youtube video's stats.",
    #         "type": "integer",
    #     },
    #     {
    #         "name": "comment_count",
    #         "description": "Comment count from Youtube video's stats.",
    #         "type": "integer",
    #     },
    #     {
    #         "name": "title",
    #         "description": "The title of the youtube video",
    #         "type": "string",
    #     },
    #     {
    #         "name": "words_in_title",
    #         "description": "An array created from splitting the title at the whitespaces. Contains each word in the title.",
    #         "type": "array",
    #     },
    #     {
    #         "name": "video_id",
    #         "description": "The video ID from youtube",
    #         "type": "string",
    #     },
    #     {
    #         "name": "channel_information",
    #         "description": "Information about the channel including channelId, handle, channelTitle, description, thumbnail, subscriberCount, and videoCount. In JSON dump string format with escape slashes(\\) before double quotation marks.",
    #         "type": "string",
    #     },
    #     {
    #         "name": "captions",
    #         "description": "Array of all the words in the document. Information about videos and all words said in video.",
    #         "type": "array",
    #     },
    #     {
    #         "name": "text",
    #         "description": "All the text in the document.",
    #         "type": "string",
    #     },
    #     {
    #         "name": "MAIN_DOC",
    #         "description": "If MAIN_DOC = True that means this document is a MAIN_DOC. MAIN_DOC's contain basic channel information including: channelId, handle, channelTitle, description, thumbnail, subscriberCount, and videoCount. MAIN_DOC's also contain a list of all videos(videoIds, titles and publishedAt dates) sorted from newest to oldest. MAIN_DOC's DO NOT include video captions.",
    #         "type": "boolean",
    #     },
    # ]
    metadata_field_info = [
        AttributeInfo(
            name="date",
            description="The date the youtube video was published in YYYYMMDDHHMMSS format",
            type="INTEGER. DO NOT PUT QUOTATIONS AROUND VALUE",
        ),
        # AttributeInfo(
        #     name="document_id",
        #     description="The document ID inside the vector store. The ID for a Video document is formatted like this: video_id + #chunk number. The ID for a MAIN_DOC document is formatted like this: channelTitile + '_MAIN_DOC' + #chunk number.",
        #     type="string",
        # ),
        # AttributeInfo(
        #     name="view_count",
        #     description="View count from Youtube video's stats.",
        #     type="integer",
        # ),
        # AttributeInfo(
        #     name="like_count",
        #     description="Like count from Youtube video's stats.",
        #     type="integer",
        # ),
        # AttributeInfo(
        #     name="comment_count",
        #     description="Comment count from Youtube video's stats.",
        #     type="integer",
        # ),
        AttributeInfo(
            name="title",
            description="The title of the youtube video",
            type="string",
        )#,
        # AttributeInfo(
        #     name="words_in_title",
        #     description="An array created from splitting the title at the whitespaces. Contains each word in the title.",
        #     type="array",
        # ),
        # AttributeInfo(
        #     name="video_id",
        #     description="The video ID from youtube",
        #     type="string",
        # ),
        # AttributeInfo(
        #     name="channel_information",
        #     description="Information about the channel including channelId, handle, channelTitle, description, thumbnail, subscriberCount, and videoCount. In JSON dump string format with escape slashes(\\) before double quotation marks.",
        #     type="string",
        # ),
        # AttributeInfo(
        #     name="captions",
        #     description="Array of all the words in the document. Information about videos and all words said in video.",
        #     type="array",
        # ),
        # AttributeInfo(
        #     name="text",
        #     description="All the text in the document.",
        #     type="string",
        # ),
        # AttributeInfo(
        #     name="MAIN_DOC",
        #     description="If MAIN_DOC = True that means this document is a MAIN_DOC. MAIN_DOC's contain basic channel information including: channelId, handle, channelTitle, description, thumbnail, subscriberCount, and videoCount. MAIN_DOC's also contain a list of all videos(videoIds, titles and publishedAt dates) sorted from newest to oldest. MAIN_DOC's DO NOT include video captions.",
        #     type="boolean",
        # ),
    ]
    document_content_description = f"Information from a Youtube channel titled {input_dir}. Documents are either MAIN_DOC documents or Video documents. Video documents contain captions text of what was said during the video."
    llm = Predibase(
        model="llama-3-1-8b-instruct",
        predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
        temperature=0,
        max_new_tokens=10000,
        # predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
        # adapter_id="yt_lore",
        # adapter_version=1,
    )
    # llm2 = Predibase(
    #     model="llama-3-1-8b",
    #     predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    #     temperature=0,
    #     max_new_tokens=8000,
    #     # predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    #     # adapter_id="yt_lore",
    #     # adapter_version=1,
    # )
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


    # StructuredQuery captures the filters specified by the user
    from langchain.chains.query_constructor.base import (
        StructuredQueryOutputParser,
        get_query_constructor_prompt,
    )
    from langchain_core.runnables import RunnableLambda, RunnableParallel
    from langchain.output_parsers import RetryOutputParser
    from langchain_core.output_parsers import JsonOutputParser

       # Define allowed comparators list
    allowed_comparators = [
        "$eq",  # Equal to (number, string, boolean)
        "$ne",  # Not equal to (number, string, boolean)
        "$gt",  # Greater than (number)
        "$gte",  # Greater than or equal to (number)
        "$lt",  # Less than (number)
        "$lte",  # Less than or equal to (number)
        "$in",  # in a specified array (array)
        "$nin",  # not in a specified array (array)
    ]

    prompt = get_query_constructor_prompt(
        document_content_description,
        metadata_field_info,
        allowed_comparators=allowed_comparators,
        #schema_prompt=""
    )
    output_parser = StructuredQueryOutputParser.from_components()
    output_parser2 = JsonOutputParser()
    def dumpjs(output):
        print("\nDUMP:", json.dumps(output))
        return output
    def reg_str(output):
        return output.replace('```json', '').replace('```', '')
    query_constructor = prompt | llm | dumpjs | output_parser
    
    # retry_parser1 = RetryOutputParser.from_llm(parser=StructuredQueryOutputParser.from_components(), llm=llm)
    # main_query_constructor = RunnableParallel(
    #     completion=query_constructor, prompt_value=prompt
    # ) | RunnableLambda(lambda x: retry_parser1.parse_with_prompt(**x))


    # retriever1 = SelfQueryRetriever.from_llm(
    #     llm,
    #     metadata_field_info,
    #     vectorstore,
    #     document_content_description,
    # )
    retriever2 = SelfQueryRetriever(
        query_constructor=query_constructor,#or main_query_constructor
        vectorstore=vectorstore,
        structured_query_translator=PineconeTranslator(),
        verbose=True,
        enable_limit=True
    )

    def _combine_documents(docs: List) -> str:
        return "\n\n".join(format_document(doc, prompt=DOCUMENT_PROMPT) for doc in docs)
    _context = RunnableParallel(
        context=retriever2 | _combine_documents,
        question=RunnablePassthrough(),
    )
    chain = (
        _context | LLM_CONTEXT_PROMPT | llm | StrOutputParser()
    )

    # retry_parser2 = RetryOutputParser.from_llm(parser=StrOutputParser(), llm=llm)
    # main_chain = RunnableParallel(
    #     completion=chain, prompt_value=_context | LLM_CONTEXT_PROMPT
    # ) | RunnableLambda(lambda x: retry_parser2.parse_with_prompt(**x))

    #main_chain.invoke({"query": "who is leo di caprios gf?"})

    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever1)

    # Example usage
    q1 = f"Who is {input_dir}?"
    q2 = f"Give the 3 original video title ideas in {input_dir}'s style."
    q3 = f"Give me some lore about {input_dir}. What are their most common words?"
    q4 = f"What topics does {input_dir} likes the most? Can you recommend a video in their style?"
    q5 = f"How many videos does {input_dir} have posted on Youtube?"
    q6 = f"What is the most common word said in all the videos {input_dir} has posted on Youtube?"
    q7 = f"What are the titles of the three most recent videos {input_dir} posted on Youtube?"
    q8 = f"How many videos did {input_dir} post in the month of June 2024?"
    q9 = "Give me a title of a video posted after the date 20240625000000"
    queries = [q5,q6,q7,q8]

    query = q9


    # response = qa.invoke(query)
    # print("Response:", response)

    construct = query_constructor.invoke(
        {
            "query": query
        }
    )
    print("Construct:", construct)
    response = chain.invoke({"query": query})
    print("Response:", response)

    # for i, q in enumerate(queries):
    #     construct = query_constructor.invoke(
    #         {
    #             "query": q
    #         }
    #     )
    #     print(f"Construct {i+1}:", construct)
    #     #response = qa.invoke(q)
    #     response = chain.invoke(q)
    #     print(f"Response {i+1}:",response)


if __name__ == '__main__':
    main()

"""
TO DO LIST:
-fix parsing
.replace('```json', '').replace('```', '')
LangChain Output Parser
-finetune
"""