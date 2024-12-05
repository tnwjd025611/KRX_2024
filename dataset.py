import os
import copy
import logging

from typing import Dict, Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm

# import pickle
import utils

import transformers
import torch

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {'GENERAL': """As a financial expert, you possess the ability to analyze various financial data and market trends. 

Instructions:
Explore the background, and if there is information relevant to the question, collect it and use it as the basis for your answer. If there is no relevant information in the background, use your inherent knowledge.
1. Analyze the Background: Review the information provided in the 'Background' section.
2. Utilize Relevant Information: If the Background contains data related to the question, use it as the foundation for your response.
3. Leverage Financial Expertise: If no relevant information is present in the Background, rely on your expert knowledge.
4. Provide a Reliable Answer: Ensure your response is clear, accurate, and actionable.

Background: {context}
Question: {question}

Please provide a clear and reliable answer to the following question. 
###정답:""",
              
              'MCQA': """As a financial expert, you possess the ability to analyze various financial data and market trends. 

Instructions:
Choose the answer to the given question from the answer options. When selecting the answer, make sure to use your inherent knowledge and follow a logical reasoning process.
1. Understand the Question: Carefully read and comprehend the provided 'question'.
2. Analyze the Answer Options: Review the given 'answer_options' and select the most appropriate answer based on your financial expertise.
3. Provide the Answer: Respond with only the selected answer, without any additional explanation.

Question: {question}
Answer options: {answer_options}

Please provide a clear and reliable answer to the following question. 
###정답:""",
              
              'MCQA_REASON': """As a financial expert, you possess the ability to analyze various financial data and market trends. 

Instructions:
For the given question, when the answer from the "Answer options" is marked as the correct answer, explain the reason or solution method for why it is the correct answer.
1. Understand the Question: Carefully read and comprehend the provided 'question'.
2. Analyze the Given Answer: Review the 'correct_answer' and determine why it is the most appropriate solution to the 'question'.
3. Provide a Justification: Clearly and concisely explain the reason or solution method that makes the 'correct_answer' valid and appropriate.

Question: {question}
Answer options: {answer_options}
Answer: {golden}

Provide a clear and concise explanation of the reason for the correct answer based on the question and the answer.
###정답:""",
              'SENT': """As a financial expert, you possess the ability to analyze various financial data and market trends. 

Instructions:
Analyze the given sentence from the financial domain. Based on its tone and context, classify it as one of the following: Negative, Positive, or Neutral. Ensure your classification is logical and supported by the content of the sentence.
1. Understand the Sentence: Carefully read and comprehend the given sentence.
2. Identify the Context: Analyze the sentence to understand the financial domain context it refers to (e.g., market performance, financial outlook, risk evaluation).
3. Evaluate the Tone: Assess the overall sentiment of the sentence—whether it conveys a Negative, Positive, or Neutral tone—based on its language, implications, and context.
4. Classify the Sentence: Based on your analysis, classify the sentence as one of the following: Negative, Positive, or Neutral.

Sentence: {sentence}

Please classify the tone of the sentence as one of the following categories:
negative
positive
neutral
###정답:""",}


# def open_file(path):
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#     return data

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizerFast,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(
        strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizerFast
) -> Dict:
    """Tokenize a list of strings"""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors='pt',
            padding='longest',
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids = input_ids,
        labels = labels,
        input_ids_lens = input_ids_lens,
        labels_lens = labels_lens
    )

def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizerFast,
) -> Dict:
    """Preprocess the data by tokenizing"""
    examples = [s+t for s,t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]

    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized['input_ids_lens']):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def get_prompt_format(example): #재작성 예장
    general_prompt = PROMPT_DICT['GENERAL']
    mcqa_prompt = PROMPT_DICT['MCQA']
    reason_prompt = PROMPT_DICT['MCQA_REASON']
    sent_prompt = PROMPT_DICT['SENT']
    
    is_general = example.get('is_general', False)
    is_mcqa = example.get('is_mcqa', False)
    is_reason = example.get('is_reason',False)
    is_sent = example.get('is_sent',False)

    if is_general:
        return general_prompt.format_map(example)
    elif is_mcqa:
        return mcqa_prompt.format_map(example)
    elif is_reason:
        return reason_prompt.format_map(example)
    elif is_sent:
        return sent_prompt.format_map(example)
    else:
        raise ValueError('Check Your Config or Dataset')


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning"""

    def __init__(self, data_path:str, tokenizer: transformers.PreTrainedTokenizerFast):
        super(SupervisedDataset, self).__init__()
        logging.warning('Loading data ...')
        sources = []
        targets = []

        for data in data_path:
            list_data_dict = utils.jload(data)
    
            for example in list_data_dict:
                p = get_prompt_format(example)
                messages = [
                    {"role": "system", "content": "You are a skilled financial expert capable of analyzing various financial data and market trends with precision and clarity."},
                    {"role": "user", "content": p}
                ]
                messages = tokenizer.apply_chat_template(messages, tokenize=False)
                sources.append(messages)
            targets += [
                f'{list_data_dict[num]["answer"]}{tokenizer.eos_token}' for num in range(len(list_data_dict))
            ]

        logging.warning('Tokenizing inputs ... This may take some time ...')
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids = self.input_ids[i], labels=self.labels[i])
        
@dataclass
class DataCollatorForSupervisedDataset(object):
    """collate exmaples for supervised fine-tuning"""
    tokenizer : transformers.PreTrainedTokenizerFast

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ['input_ids', 'labels']
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first = True, padding_value = IGNORE_INDEX
        )
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
def make_supervised_data_module(
        tokenizer : transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )

    # eval_dataset = SupervisedDataset(
    #     tokenizer=tokenizer, data_path=data_args.eval_path
    # )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )