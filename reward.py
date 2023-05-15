import torch
import random
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, get_scheduler

random.seed(42)

is_fake_subset = False
batch_size=32
device = "cuda"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

print(">>>> Preprocessing Dataset")

train_dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", cache_dir="/scratch4/danielk/<wchoi20>")
train_dataset = train_dataset['train']

# for half size (Model B)
# train_dataset = train_dataset.shard(num_shards=2, index=0)

class AnthropicDataset(torch.utils.data.Dataset):

    def __init__(self, contents, chosen, tokenizer):
        self.contents = contents
        self.chosen = chosen
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, index):
        content = str(self.contents[index])
        is_chosen = self.chosen[index]

        encoded_review = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length = 128,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
            )

        return {
            'input_ids': encoded_review['input_ids'][0],
            'attention_mask': encoded_review['attention_mask'][0],
            'labels': torch.tensor(is_chosen, dtype=torch.long)
            }

train_subsets = []
train_labels = []

# for inaccurate data (Model C)
# random_fake = random.randint(0, 20)
# count = 1

for passage in train_dataset:

    train_subsets.append(passage['chosen'])
    train_labels.append(True)

    # if random_fake % count == 0:
    #     train_labels.append(False)
    # else:
    #     train_labels.append(True)

    train_subsets.append(passage['rejected'])
    train_labels.append(False)

    # if random_fake % count == 0:
    #     train_labels.append(True)
    # else:
    #     train_labels.append(False)

    # count += 1

train_dataset = AnthropicDataset(
      contents=train_subsets,
      chosen=train_labels,
      tokenizer=tokenizer
  )

print(">>>> Dataset to DataLoader")
dataloader = DataLoader(train_dataset, batch_size=batch_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
num_epochs = 1

lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(dataloader) * num_epochs
    )

loss = torch.nn.CrossEntropyLoss()

print(">>>> Begin Training BERT Model")

for epoch in range(num_epochs):

    model.train()

    for i, batch in enumerate(dataloader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predictions = output.logits
        model_loss = loss(predictions, labels)

        model_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

print(">>>> Finished Training BERT Model")

model.save_pretrained("./bert-reward-model")
tokenizer.save_pretrained("./bert-reward-tokenizer")