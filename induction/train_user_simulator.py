# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
@Time: 2024-04-30
@Author: jiayuwu
"""

import copy
from dataclasses import dataclass, field 
import json
import pathlib
from typing import Dict, Optional, Sequence
import pickle

import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer, PreTrainedModel, HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, DistributedSampler

from fastchat.conversation import get_default_conv_template, SeparatorStyle
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from tqdm import tqdm
import random
import pprint
import numpy as np
# flash_attn加速
replace_llama_attn_with_flash_attn()

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "</s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    group_by_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

new_instruct = '''
    You are an expert at generating instructions in human-AI conversations. The following is a dialogue between a user and an AI assistant. User statements start with [USER] and AI assistant statements start with [ASSISTANT]. \
    To make it easier for you to ask for an insightful instruction, I will provide you a candidate of instructional strategies. Instructional strategies are general principles for thoughtfully, purposefully, and strategically generating instructions to obtain desired information, create conversation, stimulate thinking, or encourage others to share. \
    You first need to choose an instructional strategy that fits the current flow of the dialogue from the candidate, and then generate a specific instruction based on the high-level instructional strategy and the dialogue history. \
    The candidate of instructional strategies is: {}. \
    The dialogue history is: 
    '''

def preprocess(
    sources,
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    conv = get_default_conv_template("vicuna_v1.1").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    with open(candidate_path, 'r') as file:
        strategy_list = json.load(file)
    count = 0
    conversations = []
    for i, source in enumerate(sources):
        random.shuffle(strategy_list)
        strategy_list = strategy_list[:50] 
        if roles[source['conversations'][0]["from"]] != conv.roles[0]:
            source = source['conversations'][1:]

        conv.messages = []
        flag = True
        for j, sentence in enumerate(source['conversations']):
            sentence["value"] = sentence["value"].replace("</s>", " ")
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            
            if j != len(source['conversations'])-1:
                conv.append_message(role, sentence["value"])
            else:
                strategy, question = sentence["value"].split('【instructional strategy】')[1].split('【instruction】')
                if question == "":
                    flag = False
                if strategy not in strategy_list:
                    strategy_list[-1] = strategy
                random.shuffle(strategy_list)
                conv.system = new_instruct.format(strategy_list)
                sentence["value"] = sentence["value"] + DEFAULT_EOS_TOKEN
                conv.append_message(role, sentence["value"])
        conversation = conv.get_prompt()
        if '<unk>' in conversation or '<bos>' in conversation or '<pad>' in conversation or '<eos>' in conversation:
            continue

        prompt_len = len(tokenizer(conversation).input_ids)
        if prompt_len > tokenizer.model_max_length:
            count += 1
            continue

        if flag and 'USER: ASSISTANT' not in conversation and 'USER:ASSISTANT' not in conversation:
            conversations.append(conversation)
        
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        if rounds[-1] == ' ':
            rounds = rounds[:-1]
        cur_len = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            round_len = len(tokenizer(rou).input_ids)
            if i != len(rounds) - 1:
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                target[cur_len: cur_len+round_len] = (IGNORE_TOKEN_ID)
            else:
                target[cur_len] = (IGNORE_TOKEN_ID)
            

            cur_len += round_len
        target[cur_len+1:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length:
            if cur_len + 2 != total_len:
                rank0_print(f"WARNING: tokenization mismatch "
                            f"{cur_len} vs. {total_len}")
        else:
            rank0_print(f"WARNING: input is too long ")
    return dict(input_ids=input_ids, labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id))


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                tokenizer: PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        rank0_print("Loading data...")
        list_data_dict = []
        with open(data_path, "r", encoding="UTF-8") as file:
            data =file.readlines()
            for line in data:
                list_data_dict.append(json.loads(line))

        rank0_print("Formatting inputs...")
        sources = [example for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    attention_mask=self.attention_mask[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                tokenizer: PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Loading data...")

        list_data_dict = []
        with open(data_path, "r", encoding="UTF-8") as file:
            data =file.readlines()
            for line in data:
                list_data_dict.append(json.loads(line))
        print('len(list_data_dict)', len(list_data_dict))


        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        for _ in range(10):
            try:
                sources = self.list_data_dict[i]
                if isinstance(i, int):
                    sources = [sources]
                data_dict = preprocess([e for e in sources],
                    self.tokenizer)
                if isinstance(i, int):
                    data_dict = dict(input_ids=data_dict["input_ids"][0],
                                    labels=data_dict["labels"][0],
                                    attention_mask=data_dict["attention_mask"][0])
            except:
                i = random.choice(list(range(len(self) - 1)))
                continue
            else:
                break
        return data_dict


def make_supervised_data_module(tokenizer: PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path)
    return dict(train_dataset=train_dataset,
                eval_dataset=None)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    dataset = data_module['train_dataset']
    sample=dataset[0]
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )
    model.config.max_position_embeddings = training_args.model_max_length
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      **data_module)

    print('begin train')
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                    output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
