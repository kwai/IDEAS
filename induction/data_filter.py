import sys
import os
import json
import argparse
from rouge_score import rouge_scorer
import numpy as np
import Levenshtein

def filter_sim(data):    
    results = []
    for item in data:
        query_list = []
        for utterance in item['conversations']:
            if utterance['from'] == 'human':
                query_list.append(utterance['value'])
        
        query_dis = []
        for i in range(len(query_list)):
            for j in range(i + 1, len(query_list)):
                query_dis.append(Levenshtein.ratio(query_list[i], query_list[j]))
        avg_query = np.max(query_dis)
	
	
        if avg_query < 0.8:
            results.append(item)

    return results

def filter_keyword(data):
    results = []
    for item in data:
        flag = 1
        message = item['conversations']
        all_res = [x['value'] for x in message if x['from'] == 'gpt']
        delete_keywords = ["语言模型", "抱歉", "我无法", "Sorry", "sorry", "apologize", "language model","UNKNOWN ERROR"]
        for res in all_res:
            if res == '' or res == 'Omitted content due to a flag from our content filters' or any(keyword in res for keyword in delete_keywords):          
                flag = 0
        if flag == 1:
            results.append(item)
    return results

def main(args):
    with open(args.data_path, 'r') as file:
        train_data = json.load(file)
    print('len(train_data)',len(train_data))

    div_data = filter_sim(train_data)
    print('len(div_data)',len(div_data))

    final_data = filter_keyword(div_data)
    print('len(final_data)',len(final_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")   
    parser.add_argument("--save_path", type=str, default="")   
    args = parser.parse_args()
    main(args)

