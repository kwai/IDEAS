import copy
import os
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import pickle
from tqdm import tqdm
import torch
import random
import datasets
from datasets import Value, load_dataset, Dataset
import transformers
from transformers import (
    AutoConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import Levenshtein

# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
import jsonlines

import time
import re
import tiktoken
import argparse
from multiprocessing import cpu_count
import multiprocessing
from joblib import Parallel, delayed
import openai
import numpy as np

openai.api_key = ""
enc = tiktoken.encoding_for_model("")
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

from torchmetrics.functional.text.rouge import rouge_score

import sys,os
from infer.infer import ModelInfer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import logging
from google.protobuf import text_format
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

random.seed(42)


def message_to_prompt(message):
    for x in message:
        if type(x['content']) == list:
            x['content'] = ''.join(x['content'])
    message = [x["role"].upper() + ': ' + x['content'] for x in message]
    prompt = ''
    for m in message:
        prompt = prompt + m
        if 'ASSISTANT' in m:
            prompt += "</s>"
        else:
            prompt += " "
    return prompt

def get_result(i, json_content):
    for _ in range(API_MAX_RETRY):
        try:
            message=[{"role": "user", "content": json_content}]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=0,
                frequency_penalty=0.6,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output

def remove_context(message, target_len = 3000, mode = 'chatgpt', tokenizer=None):
    for _ in range(30):
        prompt = message_to_prompt(message)
        if mode == 'chatgpt':
            len_context = len(enc.encode(prompt))
        else:
            len_context = len(tokenizer([prompt]).input_ids[0])
        if len_context > target_len:
            message = message[2:]
        else:
            break
    return message


def select_user_query(message, querys):
    his_query = [x['content'] for x in message if x['role'] == 'user']
    query_dis = []
    for q in querys:
        query_dis.append(sum([rouge_score(q, h, rouge_keys=('rougeL'))['rougeL_fmeasure'] for h in his_query]))
    index = query_dis.index(min(query_dis))
    return querys[index]

def check_different(message, querys):
    scores = []
    for query in querys:
        scores.append(Levenshtein.ratio(message, query))
        
    if scores and max(scores) > 0.8:
        return False
    return True
   
def get_reward_datasets(reward_tokenizer, context, skills_list, skills_after_tokenizer):
    max_length= 512

    context_tokenizer = reward_tokenizer([context], padding=False, max_length=512, truncation=True)
    data_dict = {"skill": skills_list, "input_ids": [], "token_type_ids": [], "attention_mask": []}

    for i in range(len(skills_after_tokenizer['input_ids'])):
        length = len(context_tokenizer['input_ids'][0][:-1]) + len(skills_after_tokenizer["input_ids"][i][1:])
        pad = [0] * (max_length - length)
        input_ids = context_tokenizer['input_ids'][0][:-1] + skills_after_tokenizer["input_ids"][i][1:] + pad
        token_type_ids = context_tokenizer['token_type_ids'][0][:-1] + skills_after_tokenizer["token_type_ids"][i][1:] + pad
        attention_mask = context_tokenizer['attention_mask'][0][:-1] + skills_after_tokenizer["attention_mask"][i][1:] + pad
        
        input_ids = input_ids[-max_length:]
        token_type_ids = token_type_ids[-max_length:]
        attention_mask = attention_mask[-max_length:]
        input_ids[-max_length] = 101

        data_dict["input_ids"].append(input_ids)
        data_dict["token_type_ids"].append(token_type_ids)
        data_dict["attention_mask"].append(attention_mask)

    raw_datasets = Dataset.from_dict(data_dict)
    return raw_datasets


def do_predict(trainer, reward_tokenizer, predict_dataset):
    candidate_list = []
    label_list = ['0', '1']

    try:
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions_idx = np.argmax(predictions, axis=1)

        for index, item in enumerate(predictions_idx):
            item = label_list[item]
            skill = predict_dataset['skill'][index]
            if item == '1':
                candidate_list.append(skill)
    except:
        print('predict error')

    random.shuffle(candidate_list)
    candidate_list = candidate_list[:50]
    
    return candidate_list

def get_user_query(trainer, tokenizer, reward_tokenizer, selected_skill_list, skills_list, skills_after_tokenizer, index, topic, message, model, device, model_max_length):
    start_time = time.time()
    reward_model_data = get_reward_datasets(reward_tokenizer, message[-1]['content'], skills_list, skills_after_tokenizer)
    start_time = time.time()
    skill_list = do_predict(trainer, reward_tokenizer, reward_model_data)

    system_instruct = '''
        You are an expert at generating instructions in human-AI conversations. The following is a dialogue between a user and an AI assistant. User statements start with [USER] and AI assistant statements start with [ASSISTANT]. \
        To make it easier for you to ask for an insightful instruction, I will provide you a candidate of instructional strategies. Instructional strategies are general principles for thoughtfully, purposefully, and strategically generating instructions to obtain desired information, create conversation, stimulate thinking, or encourage others to share. \
        You first need to choose an instructional strategy that fits the current flow of the dialogue from the candidate, and then generate a specific instruction based on the high-level instructional strategy and the dialogue history. \
        The candidate of instructional strategies is: {}. \
        The dialogue history is: 
    '''.format(skill_list)
    len_system = len(tokenizer(system_instruct).input_ids)
    new_message = remove_context(message, model_max_length-len_system, mode = 'ask', tokenizer=tokenizer)
    instruction = [m['content'] for m in new_message if m['role'] == 'user']
    prompt = message_to_prompt(new_message)
    prompt = system_instruct + " " + prompt

    all_output_text = []
    
    start = time.time()
    try:
        output_text = model.forward(prompt)
    except:
        output_text = ""
    output = output_text
    if type(output_text) == list:
        output_text = ''.join(output_text)
    output_text = output_text.split('ASSISTANT')[0]
    output_text = output_text.split('ASS')[0]
    
    if '【question】' in output_text:
        try:
            selected_skill, output_text = output_text.split('【instructional strategy】')[1].split('【instruction】')
        except:
            selected_skill = ""
            output_text = ""

        selected_skill_list.append(selected_skill)

        is_effective = check_different(output_text, instruction)
        if is_effective:
            verify_prompt = '''
                You are an expert in data evaluation of multi-turn instruction-following dialogues. The multi-turn instruction-following dialogue is a multi-turn dialogue process in which USER poses a instruction and ASSISTANT gives the corresponding answer. Currently, the USER raises a new instruction based on the instruction-following dialogue history. Your task is to check whether the proposed instruction is reasonable.
                A reasonable instruction needs to be correct and logically coherent with the dialogue history.
                A correct instruction needs to meet the following two conditions: 
                    (1) The raised instruction must be no contradiction with the answers from ASSISTANT, or information obtained by further reasoning based on the answers from ASSISTANT, or information obtained from a comprehensive understanding of the answers from ASSISTANT, or the existing instructions;
                    (2) The raised instruction cannot be answered by any existing answers from ASSISTANT.
                A logically coherent instruction question needs to meet at least one of the following conditions:
                    (1) The raised instruction may be a new request, suspection, confirmation or inquiry about the answers from ASSISTANT;
                    (2) The raised instruction may be a continued question about the concept or entity contained in the answers from ASSISTANT; 
                    (3) The raised instruction may be an extension of the previous instructions;
                    (4) The raised instruction may be a new instruction about the information related to the concept or entity contained in the answers from ASSISTANT. 
                The dialogue history is ###%s###, the existing instructions are ###%s###, and the raised instruction is ###%s###. Next, please carefully analyze whether the raised instruction is reasonable. Please be sure to output according to the following JSON format: {"analysis": "xxx", "result": "xxx"}, where “analysis” is the reason for whether it is reasonable, and “result” is "yes" or "no".'''%(message, instruction, output_text)
            verify_result = get_result(index, verify_prompt)
            if verify_result is None:
                result = False
            elif "\"result\": \"yes\"" in verify_result:
                result = True
            else:
                result = False
            is_effective &= result
    else:
        is_effective = False
    
    print("output{},output_text{}".format(output,output_text))
    return output, output_text, is_effective, selected_skill_list

def get_assistant_chatgpt_response(message, idx):
    message = remove_context(message, target_len = 120000, tokenizer=None)
    ai_completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
            )
    ai_response = ai_completion["choices"][0]["message"]["content"]
    return ai_response, len(enc.encode(' '.join(message['content']) + ai_response))

def get_session(index, model, device, tokenizer, trainer, reward_tokenizer, test_data, skills_list, skills_after_tokenizer, model_max_length, num_round=10):
    print(index)
    if not os.path.exists( f"{save_path}/{index}.jsonl" ):
        message_show = test_data[index]['message']
        message_infact = test_data[index]['message']
        topic = test_data[index]['topic'].lower().replace('.', '')
        selected_skill_list = []
        cost_tokens = 0
        verify_count = 0
        dialogue_verify_count_list = []
        i = 0
        while (len(message_show) // 2) < num_round:
            if verify_count > 5:
                break
            user_res_jieduan, user_res, verify_result, selected_skill_list= get_user_query(trainer, tokenizer, reward_tokenizer, selected_skill_list, skills_list, skills_after_tokenizer, index, topic, message_infact, model, device, model_max_length)
            if not verify_result:
                verify_count += 1
                continue
            else:
                dialogue_verify_count_list.append(verify_count)
                verify_count = 0
            message1 = message_show + [{'role': 'user', 'content': user_res_jieduan}]
            message_jieduan = message_infact + [{'role': 'user', 'content': user_res}]
            import pdb; pdb.set_trace()
            as_res, n_tokens = get_assistant_chatgpt_response(message_jieduan, str(index)+'_'+str(i))
            cost_tokens += n_tokens

            message_show = message1 + [{'role': 'assistant', 'content': as_res}]
            message_infact = message_jieduan + [{'role': 'assistant', 'content': as_res}]
            i += 1

        if 'question_id' in test_data[index]:
            question_id = test_data[index]['question_id']
        else:
            question_id = index

        if len(message_show) >= 4:
            with jsonlines.open(f"{save_path}/{index}.jsonl", 'w') as f:
                f.write( {'question_id': question_id, 'topic': topic, 'coversation':  message_infact, 'cost_tokens':cost_tokens})

        return {'question_id':question_id, 'topic': "", 'coversation': message_infact, 'cost_tokens':cost_tokens}

    else:
        print('exist!')

def main(model_name_or_path, tokenizer_path, model_max_length, data_path, category_path, save_path, data_part, rwmodel_name_or_path, device):
    os.makedirs(save_path, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path).half().to(device)
    model.eval()

    rewardmodel_name_or_path = rwmodel_name_or_path

    reward_tokenizer = BertTokenizer.from_pretrained(
        rewardmodel_name_or_path,
        use_fast=True,
    )
    reward_model = BertForSequenceClassification.from_pretrained(
        rewardmodel_name_or_path,   
    )
    trainer = Trainer(
        model=reward_model,  
    )

    test_data = []
    with open(data_path, 'r') as f_in:
        for line in f_in:
            test_data.append(json.loads(line))
    indexs = list(range(len(test_data)))[data_part*10000: (data_part+1)*10000]

    with open(category_path, 'r') as file:
        skills_list = json.load(file)
    
    skills_for_tokenizer = []
    for i in range(len(skills_list)):
        skills_for_tokenizer.append('\n' + skills_list[i])

    skills_after_tokenizer = reward_tokenizer(skills_for_tokenizer, padding=False, max_length=512, truncation=True)

    
    Parallel(n_jobs=1, backend = 'threading')(delayed(get_session)(i, model, device, tokenizer, trainer, reward_tokenizer, test_data, skills_list, skills_after_tokenizer, model_max_length) for i in tqdm(indexs))

if __name__ == '__main__':

    model_name_or_path = sys.argv[1]
    tokenizer_path = sys.argv[2]
    model_max_length = int(sys.argv[3])
    data_path = sys.argv[4]
    save_path = sys.argv[5]
    category_path = sys.argv[6]
    rwmodel_name_or_path = sys.argv[7]
    save_path = sys.argv[8]
    data_part = int(sys.argv[9])

    main(model_name_or_path, tokenizer_path, model_max_length, data_path, category_path, save_path, data_part, rwmodel_name_or_path, device)

