import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, pipeline, get_scheduler, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import BertTokenizer, BertForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch, LengthSampler
from tqdm import tqdm

print(">>>> Starting code")

config = PPOConfig(
    model_name='gpt2',
    batch_size=16,
    optimize_cuda_cache=True
)

print(">>>> Importing models")

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
reference_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

print(">>>> Loading train_dataset")

train_dataset = load_dataset("squad_v2", cache_dir='/scratch4/danielk/<wchoi20>')
train_dataset = train_dataset['train']

# segmenting dataset to be smaller
train_dataset = train_dataset.shard(num_shards=50, index=0)

def build_dataset(tokenizer, dataset_name="squad_v2"):

    ds = load_dataset(dataset_name, cache_dir='/scratch4/danielk/<wchoi20>')
    ds = ds['train']
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)
    ds.set_format(type="torch")

    return ds

print(">>>> Building dataset")

dataset = build_dataset(tokenizer)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

reward_model = BertForSequenceClassification.from_pretrained("./bert-reward-model", num_labels=2)
reward_tokenizer = BertTokenizer.from_pretrained("./bert-reward-tokenizer")

reward_pipe = pipeline("text-classification", model=reward_model, tokenizer=reward_tokenizer)

reward_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
reward_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

ppo_trainer = PPOTrainer(config, model, ref_model=reference_model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)

output_min_length = 8
output_max_length = 64
output_length_sampler = LengthSampler(output_min_length, output_max_length)

print(">>>> Started fine-tuning using RLHF")

for epoch in range(1):
  for batch in tqdm(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    # response from gpt2
    response_tensors = []
    for query in query_tensors:
      response = ppo_trainer.generate(query, **generation_kwargs, max_new_tokens=64)
      response_tensors.append(response.squeeze())
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # get reward from reward model
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = reward_pipe(texts, **reward_kwargs)
    rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

    # run a PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)


model.save_pretrained("./rlhf-model-A")