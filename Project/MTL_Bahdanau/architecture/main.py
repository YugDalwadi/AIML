from model import AttentionModel
import helpers
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import numpy as np

BATCH_SIZE = 128
UNIQUE_FRA_WORDS = 7576
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eng_data, fra_data = helpers.LoadPickle(
    "pickles/source_final.pkl", "pickles/target_final.pkl")

input = torch.Tensor(eng_data).unsqueeze(-1).to(device)  # to device!!!
labels = torch.LongTensor(fra_data).unsqueeze(-1).to(device)  # to device!!!
dataset = TensorDataset(input, labels)

sampleInput, sampleOutput = input[100000:100002], labels[100000:100002]

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = AttentionModel().to(device)  # to device!!!
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
lossFn = torch.nn.CrossEntropyLoss(
    # label smoothing helps with rarer words.
    ignore_index=0, label_smoothing=0.1)
n_epoch = 30


def tfrScheduler(TFR=0.95, m=500):
    newTFR = TFR/np.exp(1/(m))
    return newTFR


leastTestLoss = 10.
for epoch in range(n_epoch):
    model.train()

    epochLoss = 0.
    for xb, yb in train_loader:
        optimizer.zero_grad()
        # print(yb.size())
        logits = model(xb, yb)

        logits = logits.view(-1, UNIQUE_FRA_WORDS)
        yb = yb.view(-1)
        loss = lossFn(logits, yb)
        epochLoss += loss

        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    print(
        f"Training Loss on epoch: {epoch+1} is: {epochLoss/len(train_loader)}")

    model.eval()
    with torch.no_grad():
        totalTestLoss = 0.

        for xb, yb in test_loader:
            optimizer.zero_grad()
            logits = model(xb, yb)

            logits = logits.view(-1, UNIQUE_FRA_WORDS)
            yb = yb.view(-1)

            loss = lossFn(logits, yb)
            totalTestLoss += loss

        predSampleOutput = model.predict(sampleInput)
        sampleOutputList = sampleOutput.tolist()
        sampleOutputList = [output for output in sampleOutputList]

        # sampleOutputList is weird, this flattens it
        trueOutputFlat = [[item[0] for item in output]
                          for output in sampleOutputList]
        print(
            f"Predicted output: {predSampleOutput} \nTrue output: {trueOutputFlat}\n")

        totalTestLoss = totalTestLoss/len(test_loader)
        print(
            f"Testing Loss on epoch: {epoch+1} is: {totalTestLoss}")
        scheduler.step(totalTestLoss)

        if (totalTestLoss < leastTestLoss):
            print(f"New best model saved!")
            torch.save(model.state_dict(),
                       r"architecture/saved_models/model.pt")
            leastTestLoss = totalTestLoss

    newTFR = tfrScheduler(model.TFR)
    model.TFR = newTFR
    print(newTFR)
