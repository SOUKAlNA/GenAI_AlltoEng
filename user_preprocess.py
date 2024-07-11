# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json

import json
from transformers import MarianMTModel, MarianTokenizer
from os.path import join, split as split_path
import os
import requests as r


# TODO Implement the preprocessing steps here
def handle_input_file(file_location, output_path):
    german_articles = join(os.getcwd(), "\dataset\german")
    french_articles = join(os.getcwd(), "\dataset\french")

    
    transformed_data = []

    paths = [french_articles, german_articles]
    for path in paths:
        #Choose correct model:
        if path == french_articles:
            model_name = 'Helsinki-NLP/opus-mt-fr-en'
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
        if path == german_articles:
            model_name = 'Helsinki-NLP/opus-mt-de-en'
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

        #parse and load json objects
        for file_name in [file for file in os.listdir(path) if file.endswith('.json')]:
            with open(join(path, file_name), encoding='utf-8') as f:
                #pass
                data = json.load(f)
                content = data["content"]
                # Call to Helsinki translator
                for i in content:
                    #texts = ' '.join(map(str, i))
                    translated = model.generate(**tokenizer(i, return_tensors="pt", padding=True))
                    translated_content = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

                    '''translated = model.generate(**tokenizer.prepare_seq2seq_batch(texts))
                    translated_content = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]'''

                    transformed_data.append(translated_content)


                    file_name = split_path(file_location)[-1]
                    with open(join(output_path), "w") as f:
                        json.dump({
                            "transformed_representation": transformed_data  
                        }, f)
    

if False:
    handle_input_file("datastructure/input-file.json", "output")
    exit(0)

# This is a useful argparse-setup, you probably want to use in your project:
import argparse
parser = argparse.ArgumentParser(description='Preprocess the data.')
parser.add_argument('--input', type=str, help='Path to the input data.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    files_inp = args.input
    files_out = args.output
    
    for file_location in files_inp:
        handle_input_file(file_location, files_out)

 