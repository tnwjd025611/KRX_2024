import os
import utils
import random
import pandas as pd

from openai import OpenAI
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from datasets import load_dataset

from prompt import *

@dataclass
class PoolArguments:
    model_name: Optional[str] = field(default=None, metadata={'help': 'Please write model name'})
    data_name: Optional[str] = field(default=None, metadata={'help': 'Infer data path'})
    output_path: Optional[str] = field(default='./', metadata={'help': 'output data path'})
    keyword: Optional[str] = field(default='', metadata={'help': 'Keyword for prompt'})
    api_key: Optional[str] = field(default=None, metadata={'help': 'OpenAi API Key'})

def mmlu_generate(data, prompt, output_path, file_name, word):
    result = []
    for i in tqdm(range(len(data)**2)):
        num = random.sample(range(0, len(data)),3)
        message_form = prompt.format(question1=data[num[0]]['question'], question2=data[num[1]]['question'], question3=data[num[2]])
        response = client.chat.completions.create(
                model=args.model_name,
                response_format={"type":"text"},
                messages=[
                    {"role": "system", "content": f"Hello, you are a senior researcher at an economic research institute with exceptional expertise in analyzing domestic and international financial markets and trends. From now on, you will review various {word} and generate questions and answers that are similar but not entirely identical to them."},
                    {"role": "user", "content": message_form}
                ]
            )
        output = response.choices[0].message.content

        result.append({'data': data, 'output': output})
        utils.jdump(result, f'{output_path}/{file_name}.json')



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
    data = load_dataset(args.data_name, "Accounting", split='test')
    print(f'{args.data_name} Start')
    file_name = args.data_name
    word = 'accounting question'
    prompt = prompting()['account_qa']

    mmlu_generate(data, prompt, args.output_path, file_name, word)

elif args.keyword == 'book':
    data = load_dataset(args.data_name, split='train')
    print(f'{args.data_name} Start')
    file_name = args.data_name
    domain = 'book'
    word = 'textbook'
    prompt = prompting()['book']

    chunk_size = 1024
    chunks = []
    for i in range(len(data)):
        content = data[i]['text']
        chunks += [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    general_generate(chunks, prompt, args.output_path, file_name+'_1', domain, word)