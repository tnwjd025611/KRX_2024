import os
import utils
import pandas as pd

from openai import OpenAI
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

from prompt import *

@dataclass
class PoolArguments:
    model_name: Optional[str] = field(default=None, metadata={'help': 'Please write model name'})
    data_path: Optional[str] = field(default=None, metadata={'help': 'Infer data path'})
    output_path: Optional[str] = field(default='./', metadata={'help': 'output data path'})
    keyword: Optional[str] = field(default='', metadata={'help': 'Keyword for prompt'})
    api_key: Optional[str] = field(default=None, metadata={'help': 'OpenAi API Key'})

def general_generate(data, prompt, output_path, file_name, domain, word):
    result = []
    for i in tqdm(range(len(data))):
        message_form = prompt.format(document = data[i])
        response = client.chat.completions.create(
                model=args.model_name,
                response_format={"type":"text"},
                messages=[
                    {"role": "system", "content": f"Hello, you are a senior researcher at an economic research institute with exceptional expertise in analyzing domestic and international financial markets and trends. From now on, you will read {word} and create questions and answers in the {domain} domain."},
                    {"role": "user", "content": message_form}
                ]
            )
        output = response.choices[0].message.content

        result.append({'output': output})
        utils.jdump(result, f'{output_path}/{file_name}.json')

parser = HfArgumentParser(PoolArguments)
args = parser.parse_args_into_dataclasses()[0]

client = OpenAI(api_key=args.api_key)

if args.keyword == 'accounting':
    file_list = utils.list_all_files(args.data_path)
    for file in file_list:
        print(f'{file} Start')
        file_name = os.path.splitext(os.path.basename(file))[0]
        domain = 'financial accounting'
        word = 'lecture notes'
        prompt = prompting()['accounting']
        
        data = utils.jload(file)
        key = list(data.keys())[0]
        data = data[key]
        general_generate(data, prompt, args.output_path, file_name, domain, word)

elif args.keyword == 'book':
    file_list = utils.list_all_files(args.data_path)
    for file in file_list:
        print(f'{file} Start')
        file_name = os.path.splitext(os.path.basename(file))[0]
        domain = 'financial accounting'
        word = 'documents'
        prompt = prompting()['book']

        with open(file, 'r') as file:
            content = file.read()
        chunk_size = 512
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        data = chunks
        general_generate(data, prompt, args.output_path, file_name, domain, word)