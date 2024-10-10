
import json
import re
import codecs
import copy

from tqdm import tqdm
from multiprocessing import Pool

import time
import random
import os

random.seed(42)

combined_lines = []
pattern = r'\((.*?)\)'
f_out = '/mmu_nlp_ssd/wujiayu03/parrot_v2/generate_data/conclude_results.json'

import json
import csv

import re
import codecs
import copy

from tqdm import tqdm
from multiprocessing import Pool

import time
import random
import os

openai.api_key = ""
enc = tiktoken.encoding_for_model("")
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

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
    

def remove_code_tags(string):
    if '```json' in string:
        start_tag = "```json"
        end_tag = "\n```"
        
        if string.startswith(start_tag):
            string = string[len(start_tag):]
        if string.endswith(end_tag):
            string = string[:-len(end_tag)]
    
    return string

def sbert_conclude_without_topic(filename):
    p = Pool(processes=200)
    result_data_list = []

    with open(filename, "r", encoding="UTF-8") as file:
        data = json.load(file)
    for inner_list in data:
        new_inner_list = []
        if len(inner_list) >= 1:
            for item in inner_list:
                item = item.split('idx:')[0]
                new_inner_list.append(item)
            new_inner_list = list(set(new_inner_list))
            prompt = '''
            You are very good at generalizing a series of original instructional strategies into a high-level instructional strategy. 
            High-level instructional strategies are general principles for thoughtfully, purposefully, and strategically generating instructions to obtain the desired information, create dialogues, stimulate thinking, or encourage others to share.   
            To better reuse instructional strategies in new dialogue scenarios, you need to generalize a series of similar original instructional strategies into a higher-level instructional strategy. Each original instructional strategy is a specification of a high-level instructional strategy.               
            We provide a list of similar original instructional strategies:{}, and ask you to generalize them into a high-level instructional strategy. The high-level instructional strategy need to be a simple phrase. Next, please directly output the high-level instructional strategy: xxx. 
            '''.format(new_inner_list)
            data_list = []
            result_data_list.append((inner_list, p.apply_async(get_result, args=(i, prompt))))
        else:
            data = {
                    "original instructional strategies":inner_list,
                    "High-level instructional strategy":inner_list[0].split('[')[0]
                }
            with open(args.save_path, 'a', encoding='utf-8') as jsonfile:
                jsonfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    count = 0
    flag = 1
    id_set = set()
    while flag:
        for id, process in enumerate(result_data_list):
            if process[1].ready() and id not in id_set:
                count += 1
                print('count = ', count)
                id_set.add(id)
                result_data = process[1].get()
                try:
                    try:
                        result_data = remove_code_tags(result_data)
                    except:
                        print(process[0])
                        print(result_data)
                    last_colon_index = result_data.rfind(":")  
                    if last_colon_index != -1:
                        result_data = result_data[last_colon_index + 1:].lstrip()  

                    data = {
                        "original instructional strategies":process[0],
                        "High-level instructional strategy":result_data
                    }
                    with open(args.save_path, 'a', encoding='utf-8') as jsonfile:
                        jsonfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {result_data}")
            if count == len(result_data_list):
                flag = 0
                break
    p.close()
    p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="") 
    parser.add_argument("--save_path", type=str, default="")  
    args = parser.parse_args()
    sbert_conclude_without_topic(args)

        
    



