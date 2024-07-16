import argparse
from os.path import join
import json
import os
from transformers import LlamaTokenizer, LlamaForCausalLM

from llama_index.core import StorageContext, Settings, load_index_from_storage, VectorStoreIndex, get_response_synthesizer
from llama_index.readers.json import JSONReader

from llama_index.llms.huggingface import HuggingFaceLLM
 
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from pathlib import Path

import torch
from langdetect import detect

# Set the environment variable for PyTorch memory management
import os

def initModel(query):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Load the finetuned model and tokenizer
    model_path = join(os.getcwd(), 'final_model')
    tokenizer_path = join(os.getcwd(), 'final_tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    system_prompt = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as 
    helpfully as possible, while being safe. Your answers should not include
    any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
    Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain 
    why instead of answering something not correct. If you don't know the answer 
    to a question, please don't share false information.

    Your goal is to provide the most relevant articles to the user's query.<</SYS>>
    """

    llm = HuggingFaceLLM(context_window=4096,
                        max_new_tokens=256,
                        system_prompt=system_prompt,
                        query_wrapper_prompt=query,
                        model=model,
                        tokenizer=tokenizer,
                        device_map="auto",
                        tokenizer_kwargs={"max_length": 4096})

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    return llm

# Function to clear CUDA cache
def clear_cuda_cache():
    torch.cuda.empty_cache()

# Detect language
def detect_language(text):
    return detect(text)

# Generate text based on user query
'''def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Clear CUDA cache before generation to free up memory
    clear_cuda_cache()

    # Generate text with half-precision (float16)
    with torch.cuda.amp.autocast():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=3,  # Reduce the number of beams to reduce memory usage
            early_stopping=True,
            no_repeat_ngram_size=2  # Prevents repetition
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)'''

reader = JSONReader(ensure_ascii=True)
documents = reader.load_data(
                    input_file=r"C:\Users\selma\Desktop\seminar\genai-and-democracy-2024\datastructure\preprocessed-file.json", 
                    extra_info={})

#documents = SimpleDirectoryReader(file_path=Path("\datastructure\preprocessed-file.json")).load_data()
#pipeline = IngestionPipeline(transformations=[TokenTextSplitter(), ...])
#nodes = pipeline.run(documents=documents)
index = VectorStoreIndex.from_documents(documents)





# Handle user query
def handle_user_query(query, query_id, output_path):
    # Detect the language of the query
    detected_language = detect_language(query)
    llm = initModel(query)

    Settings.llm = llm
    # And set the service context

    storage_context = StorageContext.from_defaults(persist_dir=join(os.getcwd(),"dataset"))
    index = load_index_from_storage(storage_context)
    index = VectorStoreIndex.from_documents(documents)
    # Setup index query engine using LLM 
    query_engine = index.as_query_engine()

    retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,)

    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],)

    # query
    response = query_engine.query(query)
    
    # Dummy example of generating queries (to be replaced with your actual logic)
    result = {
        "generated_query": [query, response],  # Example generated queries
        "detected_language": detected_language,
    }
    #output_path = "~\datastructure\user-query-response"
    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)

# Optional function to rank articles (left unchanged)
#def rank_articles(generated_queries, article_representations):
#    result = []
#    print(json.dumps(generated_queries))

if True:
    # handle_user_query("What are the benefits of LLMs in programming?", "1", "output")
    exit(0)

# This is a sample argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Run the inference.')
parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output
    
    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."
    
    for query, query_id in zip(queries, query_ids):
        handle_user_query(query, query_id, output)