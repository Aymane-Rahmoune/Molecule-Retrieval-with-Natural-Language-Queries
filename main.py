import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader import GraphTextDataset, GraphDataset, TextDataset
from losses import (
    contrastive_loss,
    negative_sampling_contrastive_loss,
    info_nce_loss,
    nt_xent_loss,
)
from model import Model


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root="./data/", gt=gt, split="val", tokenizer=tokenizer)
train_dataset = GraphTextDataset(
    root="./data/", gt=gt, split="train", tokenizer=tokenizer
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 60
batch_size = 32
lr_text = 3e-5
lr_graph = 1e-4

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Model(
    model_name=model_name,
    num_node_features=300,
    nout=300,
    nhid=300,
    graph_hidden_channels=300,
)
model.to(device)

warmup_steps = int(1000 * nb_epochs / 40)
total_steps = len(train_loader) * nb_epochs

optimizer = optim.AdamW(
    [
        {"params": model.text_encoder.parameters(), "lr": lr_text},
        {"params": model.graph_encoder.parameters(), "lr": lr_graph},
    ]
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[lr_text, lr_graph],
    total_steps=total_steps,
    pct_start=warmup_steps / total_steps,
    anneal_strategy="linear",
)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = len(train_loader)
best_validation_loss = 1000000

for i in range(epoch, epoch + nb_epochs):
    print("-----EPOCH{}-----".format(i + 1))
    model.train()
    for batch in tqdm(train_loader):
        input_ids = batch.input_ids
        batch.pop("input_ids")
        attention_mask = batch.attention_mask
        batch.pop("attention_mask")
        graph_batch = batch

        x_graph, x_text = model(
            graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
        )
        current_loss = nt_xent_loss(x_graph, x_text)
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        scheduler.step()
        loss += current_loss.item()

        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print(
                "Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(
                    count_iter, time2 - time1, loss / printEvery
                )
            )
            losses.append(loss)
            loss = 0
    model.eval()
    val_loss = 0
    for batch in tqdm(val_loader):
        input_ids = batch.input_ids
        batch.pop("input_ids")
        attention_mask = batch.attention_mask
        batch.pop("attention_mask")
        graph_batch = batch
        x_graph, x_text = model(
            graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
        )
        current_loss = nt_xent_loss(x_graph, x_text)
        val_loss += current_loss.item()
    best_validation_loss = min(best_validation_loss, val_loss)
    print(
        "-----EPOCH" + str(i + 1) + "----- done.  Validation loss: ",
        str(val_loss / len(val_loader)),
    )
    if best_validation_loss == val_loss:
        print("validation loss improved saving checkpoint...")
        save_path = os.path.join("./", "model" + str(i) + ".pt")
        torch.save(
            {
                "epoch": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_accuracy": val_loss,
                "loss": loss,
            },
            save_path,
        )
        print("checkpoint saved to: {}".format(save_path))


print("loading best model...")
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root="./data/", gt=gt, split="test_cids")
test_text_dataset = TextDataset(file_path="./data/test_text.txt", tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(
    test_text_dataset, batch_size=batch_size, shuffle=False
)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(
        batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
    ):
        text_embeddings.append(output.tolist())


similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution["ID"] = solution.index
solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
solution.to_csv("submission.csv", index=False)
