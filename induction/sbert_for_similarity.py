"""
This example uses Approximate Nearest Neighbor Search (ANN) with FAISS (https://github.com/facebookresearch/faiss).

Searching a large corpus with Millions of embeddings can be time-consuming. To speed this up,
ANN can index the existent vectors. For a new query vector, this index can be used to find the nearest neighbors.

This nearest neighbor search is not perfect, i.e., it might not perfectly find all top-k nearest neighbors.

In this example, we use FAISS with an inverse flat index (IndexIVFFlat). It learns to partition the corpus embeddings
into different cluster (number is defined by n_clusters). At search time, the matching cluster for query is found and only vectors
in this cluster must be search for nearest neighbors.

This script will compare the result from ANN with exact nearest neighbor search and output a Recall@k value
as well as the missing results in the top-k hits list.

See the FAISS repository, how to install FAISS.

As dataset, we use the Quora Duplicate Questions dataset, which contains about 500k questions (only 100k are used):
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs.

As embeddings model, we use the SBERT model 'quora-distilbert-multilingual',
that it aligned for 100 languages. I.e., you can type in a question in various languages and it will
return the closest questions in the corpus (questions in the corpus are mainly in English).
"""
from sentence_transformers import SentenceTransformer
import os
import argparse
import json
import csv
import pickle
import time
import faiss
from faiss import normalize_L2
from tqdm import tqdm 
import numpy as np
from collections import defaultdict

## skill count: 1593 for theshold = 0.5

def create_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def obtain_embeddings(embedding_cache_path, skill_list, model):
    #Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        # Check if the dataset exists. If not, download and extract
        # Download dataset if needed
        embeddings_list = []
        embeddings = model.encode(skill_list, show_progress_bar=True, convert_to_numpy=True)

        print("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump(embeddings, fOut)
    else:
        print("Load pre-computed embeddings from disc")
        with open(embedding_cache_path, "rb") as fIn:
            embeddings = pickle.load(fIn)
    return embeddings


def cluster(embeddings, skill_list):
    index = create_index(embeddings)

    mask = [-1] * embeddings.shape[0]
    for question_id, question_embedding in tqdm(enumerate(embeddings)):
        if mask[question_id] != -1:
            continue
        # Search in FAISS. It returns a matrix with distances and corpus ids.
        question_embedding = np.expand_dims(question_embedding, axis=0)
        question_embedding = question_embedding / np.linalg.norm(question_embedding, axis=1)
        distances, corpus_ids = index.search(question_embedding, index.ntotal)

        # We extract corpus ids and scores for the first query
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        count = 0
        for hit in hits:
            if hit['score'] > args.threshold:
                if mask[hit['corpus_id']] != -1:
                    prev_score = float(mask[hit['corpus_id']].split("###")[-1])
                    if prev_score < hit['score']:
                        mask[hit['corpus_id']] = str(question_id) + "###" + str(hit['score'])
                        count += 1
                else:
                    mask[hit['corpus_id']] = str(question_id) + "###" + str(hit['score'])
                    count += 1
        print(count)
    return mask

def obtain_skill(data):
    skill_list = []
    skill_with_idx_list = []
    for i, item in enumerate(data):
        try:
            skill = item['conversations'][-1]['value'].split('【instructional strategy】')[1].split('【instruction】')[0]
        except:
            print()
        skill_with_idx = skill + ' ' +'idx:{}'.format(i)
        skill_list.append(skill)
        skill_with_idx_list.append(skill_with_idx)
    return skill_list, skill_with_idx_list

def main(args):
    raw_data = []
    with open(args.data_path, 'r') as file:
        for f in file:
            raw_data.append(json.loads(f))
    
    skill_list, skill_with_idx_list = obtain_skill(raw_data)

    model = SentenceTransformer(args.model_path)

    
    embeddings = obtain_embeddings(args.embedding_cache_path, skill_list, model)

    new_skill_dict = {}
    new_skill_with_idx_dict = {}
    
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]
    embeddings_cluster = cluster(embeddings, skill_list)

    cluster_dict = defaultdict(list)
    cluster_with_idx_dict = defaultdict(list)
    for idx, label in enumerate(embeddings_cluster):
        label = int(label.split("###")[0])
        cluster_dict[label].append(skill_list[idx])
        cluster_with_idx_dict[label].append(skill_with_idx_list[idx])
    new_skill_list = list(cluster_dict.values())
    new_skill_with_idx_list = list(cluster_with_idx_dict.values())

    print('length of skill set: ', len(new_skill_list))

    with open(args.save_path, 'w') as file:
        file.write(json.dumps(new_skill_list, indent=4, ensure_ascii=False))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="")
    parser.add_argument('--threshold', default= 0.5)
    parser.add_argument('--data_path', default='')
    parser.add_argument('--embedding_cache_path', default="")
    parser.add_argument('--save_path', default="")
    parser.add_argument('--top_k_hits', type=int, default=50)
    args = parser.parse_args()
    
    main(args)

