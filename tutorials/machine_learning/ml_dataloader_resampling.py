### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### Example of resampling when one class is underrepresented in the dataset.
###
### \macro_code
### \macro_output
### \author Jonah Ascoli

from pyexpat import model

import ROOT
import torch

# Create a skewed dataset with two classes, one of which is underrepresented. 
# Here, we'll create two files, one with even numbers and one with odd numbers, 
# and then merge them to form a dataset with underrepresented odd numbers.
file_name1 = "major.root"
file_name2 = "minor.root"
tree_name = "tree"

# Helpers
def define_rdf_even(num_of_entries):
    df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_").Define("b2", "(int) b1%2")

    return df

def define_rdf_odd(num_of_entries):
    df = ROOT.RDataFrame(num_of_entries).Define("b1", "(int) 2 * rdfentry_ + 1").Define("b2", "(int) b1%2")

    return df

def create_file_major(num_of_entries=10000):
    define_rdf_even(num_of_entries).Snapshot(tree_name, file_name1)

def create_file_minor(num_of_entries=100):
    define_rdf_odd(num_of_entries).Snapshot(tree_name, file_name2)

create_file_major()
create_file_minor()

df_major = ROOT.RDataFrame(tree_name, file_name1)
df_minor = ROOT.RDataFrame(tree_name, file_name2)

loss_fn = torch.nn.BCEWithLogitsLoss()
num_epochs = 2

batch_size = 16
batches_in_memory = 10

# First, let's try to create a dataloader without resampling and see how it handles the underrepresented class.
dl = ROOT.Experimental.ML.RDataLoader(
                [df_major, df_minor],
                batch_size=batch_size,
                batches_in_memory=batches_in_memory,
                target="b2",
            )

basic_model = torch.nn.Linear(1, 1)  # Simple linear model for binary classification
basic_optimizer = torch.optim.Adam(basic_model.parameters())

# Training time
train, val = dl.train_test_split(test_size=0.2)
for epoch in range(num_epochs):
    basic_model.train()
    for X, y in train.as_torch():
        basic_optimizer.zero_grad()
        loss = loss_fn(basic_model(X), y)
        loss.backward()
        basic_optimizer.step()
    basic_model.eval()
    losses = []
    for X, y in val.as_torch():
        with torch.no_grad():
            loss = loss_fn(basic_model(X), y)
        losses.append(loss.item())
    print(f"Basic DataLoader Epoch {epoch + 1}, Validation Loss: {sum(losses)/len(losses)}")

# Now, let's use the three resampling techniques: 

# Oversampling
overSampling_model = torch.nn.Linear(1, 1)
overSampling_optimizer = torch.optim.Adam(overSampling_model.parameters())
# Undersampling
underSampling_model = torch.nn.Linear(1, 1)
underSampling_optimizer = torch.optim.Adam(underSampling_model.parameters())
# Weighted sampling
weightedSampling_model = torch.nn.Linear(1, 1)
weightedSampling_optimizer = torch.optim.Adam(weightedSampling_model.parameters())

dl_over = ROOT.Experimental.ML.RDataLoader(
                [df_major, df_minor],
                batch_size=batch_size,
                batches_in_memory=batches_in_memory,
                target="b2",
                load_eager=True,
                sampling_type="oversampling",
            )
dl_under = ROOT.Experimental.ML.RDataLoader(
                [df_major, df_minor],
                batch_size=batch_size,
                batches_in_memory=batches_in_memory,
                target="b2",
                load_eager=True,
                sampling_type="undersampling",
            )
dl_weighted = ROOT.Experimental.ML.RDataLoader(
                [df_major, df_minor],
                batch_size=batch_size,
                batches_in_memory=batches_in_memory,
                target="b2",
                load_eager=True,
                sampling_ratio=0.001,  # Set the sampling ratio to give more importance to the underrepresented class
            )

# Training time with oversampling
train_over, val_over = dl_over.train_test_split(test_size=0.2)
for epoch in range(num_epochs):
    overSampling_model.train()
    for X, y in train_over.as_torch():
        overSampling_optimizer.zero_grad()
        loss = loss_fn(overSampling_model(X), y)
        loss.backward()
        overSampling_optimizer.step()
    overSampling_model.eval()
    losses = []
    for X, y in val_over.as_torch():
        with torch.no_grad():
            loss = loss_fn(overSampling_model(X), y)
        losses.append(loss.item())
    print(f"Oversampling DataLoader Epoch {epoch + 1}, Validation Loss: {sum(losses)/len(losses)}")

# Training time with undersampling
train_under, val_under = dl_under.train_test_split(test_size=0.2)
for epoch in range(num_epochs):
    underSampling_model.train()
    for X, y in train_under.as_torch():
        underSampling_optimizer.zero_grad()
        loss = loss_fn(underSampling_model(X), y)
        loss.backward()
        underSampling_optimizer.step()
    underSampling_model.eval()
    losses = []
    for X, y in val_under.as_torch():
        with torch.no_grad():
            loss = loss_fn(underSampling_model(X), y)
        losses.append(loss.item())
    print(f"Undersampling DataLoader Epoch {epoch + 1}, Validation Loss: {sum(losses)/len(losses)}")

# Training time with weighted sampling
train_weighted, val_weighted = dl_weighted.train_test_split(test_size=0.2)
for epoch in range(num_epochs):
    weightedSampling_model.train()
    for X, y in train_weighted.as_torch():
        weightedSampling_optimizer.zero_grad()
        loss = loss_fn(weightedSampling_model(X), y)
        loss.backward()
        weightedSampling_optimizer.step()
    weightedSampling_model.eval()
    losses = []
    for X, y in val_weighted.as_torch():
        with torch.no_grad():
            loss = loss_fn(weightedSampling_model(X), y)
        losses.append(loss.item())
    print(f"Weighted Sampling DataLoader Epoch {epoch + 1}, Validation Loss: {sum(losses)/len(losses)}")