from model import AttentionModel
import helpers
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import numpy as np

BATCH_SIZE = 256
UNIQUE_FRA_WORDS = 6754
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eng_data, fra_data = helpers.LoadPickle(
    "pickles/source_final.pkl", "pickles/target_final.pkl")

input = torch.Tensor(eng_data).unsqueeze(-1).to(device)  # to device!!!
labels = torch.LongTensor(fra_data).unsqueeze(-1).to(device)  # to device!!!
dataset = TensorDataset(input, labels)

train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = AttentionModel().to(device)  # to device!!!
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lossFn = torch.nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding token
n_epoch = 10


def tfrScheduler(TFR=0.75, m=100):
    newTFR = TFR/np.exp(1/(m))
    return newTFR


for epoch in range(n_epoch):
    model.train()

    epochLoss = 0.
    for xb, yb in train_loader:
        optimizer.zero_grad()
        # print(yb.size())
        logits, _ = model(xb, yb)

        logits = logits.view(-1, UNIQUE_FRA_WORDS)
        yb = yb.view(-1)

        loss = lossFn(logits, yb)
        epochLoss += loss

        loss.backward()
        optimizer.step()

    print(
        f"Training Loss on epoch: {epoch+1} is: {epochLoss/len(train_loader)}")
    torch.save(model.state_dict(), r"architecture/saved_models/model.pt")

    model.eval()
    totalTestLoss = 0.
    with torch.no_grad():
        for xb, yb in test_loader:
            optimizer.zero_grad()
            logits, _ = model(xb, yb)

            logits = logits.view(-1, UNIQUE_FRA_WORDS)
            yb = yb.view(-1)

            loss = lossFn(logits, yb)
            totalTestLoss += loss
    print(
        f"Testing Loss on epoch: {epoch+1} is: {totalTestLoss/len(test_loader)}")

    newTFR = tfrScheduler(model.TFR)
    model.TFR = newTFR
    print(newTFR)
