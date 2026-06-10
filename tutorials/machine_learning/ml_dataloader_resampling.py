### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### Example of resampling when one class is underrepresented in the dataset.
###
### \macro_code
### \macro_output
### \author Jonah Ascoli

import ROOT
import torch
torch.manual_seed(42)

# Create a skewed dataset with two classes, one of which is underrepresented. 
# Here, we'll create two files, one with even numbers and one with odd numbers, 
# and then merge them to form a dataset with underrepresented odd numbers.
ROOT.RDataFrame(100000).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1%2").Snapshot("tree", "major.root")
ROOT.RDataFrame(100).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1%2").Snapshot("tree", "minor.root")
df_major = ROOT.RDataFrame("tree", "major.root")
df_minor = ROOT.RDataFrame("tree", "minor.root")

batch_size = 16
batches_in_memory = 10
num_epochs = 10

loss_fn = torch.nn.BCEWithLogitsLoss()

def train_model(model, optimizer, dataloader):
    train, val = dataloader.train_test_split(test_size=0.2)
    for _ in range(num_epochs):
        model.train()
        for X, y in train.as_torch():
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
    losses = []
    for X, y in val.as_torch():
        with torch.no_grad():
            loss = loss_fn(model(X), y)
        losses.append(loss.item())
    print(f"Validation Loss: {sum(losses)/len(losses)}")

# First, let's try to create a dataloader without resampling and see how it handles the underrepresented class.
dl = ROOT.Experimental.ML.RDataLoader(
                [df_major, df_minor],
                batch_size=batch_size,
                batches_in_memory=batches_in_memory,
                target="b2",
                set_seed=42,
            )

basic_model = torch.nn.Linear(1, 1)  # Simple linear model for binary classification
basic_optimizer = torch.optim.Adam(basic_model.parameters())

print("Training without resampling:")
train_model(basic_model, basic_optimizer, dl)

# Now, let's try the same thing with oversampling
# Strategy: more batches of the underrepresented class
# Takes more time per epoch, but each epoch is more effective
dl_oversampled = ROOT.Experimental.ML.RDataLoader(
                [df_major, df_minor],
                batch_size=batch_size,
                batches_in_memory=batches_in_memory,
                target="b2",
                set_seed=42,
                load_eager=True, # Must be enabled for resampling
                sampling_type="oversampling", # Can also be "undersampling"
            )

oversampling_model = torch.nn.Linear(1, 1)
oversampling_optimizer = torch.optim.Adam(oversampling_model.parameters())

print("Training with oversampling:")
train_model(oversampling_model, oversampling_optimizer, dl_oversampled)

import os
os.remove("major.root")
os.remove("minor.root")
