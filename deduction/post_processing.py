import json
from tqdm import tqdm
import jsonlines
import os
import transformers
import numpy as np
import glob
from torchmetrics.functional.text.rouge import rouge_score


def read_jsonlines(file):
    data = []
    with open(file, 'r')as f:
        for l in jsonlines.Reader(f):
            data.append(l)
    return data

raw_sessions=[]
for i in tqdm(range(0, 10000)):
    if os.path.exists('/data_path/{}.jsonl'.format(i)):
        data = read_jsonlines('/data_path/{}.jsonl'.format(i))
        raw_sessions.append(data)

raw_sessions = [x[0] for x in raw_sessions if len(x)!=0]

def judge_invalid_query(message, i):
    all_query = [x['content'].lstrip(': ').strip() for x in message if x['role'] == 'user']
    query = all_query[-1]
    query_dis = []
    if query == '':
        return False, 'Null'
    else:
        return True, 'valid'

def judge_invalid_response(message):
    all_res = [x['content'] for x in message if x['role'] == 'assistant']
    delete_keywords = ["语言模型", "抱歉", "我无法", "Sorry", "sorry", "apologize", "language model","UNKNOWN ERROR"]
    res = all_res[-1]
    if res == '': 
        return False, 'null'
    elif res == 'Omitted content due to a flag from our content filters':
        return False, 'omitted'
    elif 'UNKNOWN ERROR' in res:
        return False, 'delete_keywords'
    else:
        return True, 'valid'
    

p_chat_sharegpt_20k = []
role_map = {'user':'human', 'assistant':'gpt'}
for item in tqdm(raw_sessions):
    for i in range(2, len(item['coversation']) // 2 + 1):
        is_q_valid, wrong_type = judge_invalid_query(item['coversation'][:2*i], i)
            
        is_r_valid, wrong_type = judge_invalid_response(item['coversation'][:2*i])
        if not (is_q_valid and is_r_valid):
            break
    if not (is_q_valid and is_r_valid):
        conversations = [{"from":role_map[x['role']] , 'value':x['content']} for x in item['coversation'][:2*(i-1)]]
        
        p_chat_sharegpt_20k.append({'id':item['question_id'], 'topic':item['topic'], 'conversations':conversations})
    else:
        conversations = [{"from":role_map[x['role']] , 'value':x['content']} for x in item['coversation']]
        p_chat_sharegpt_20k.append({'id':item['question_id'], 'topic':item['topic'], 'conversations':conversations})

print(len(p_chat_sharegpt_20k))
with open(args.save_path, 'w', encoding='utf-8') as f:
    json.dump(p_chat_sharegpt_20k, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()   
    parser.add_argument("--save_path", type=str, default="/mmu_nlp_ssd/wujiayu03/parrot_v2/train/data/train_new_en_addcategory_new.json")  
    args = parser.parse_args()
    process_data_60k_classify(args)




