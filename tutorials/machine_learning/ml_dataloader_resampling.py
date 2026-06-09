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

seed = 42
torch.manual_seed(seed)


# Create an imbalanced dataset with two classes, one of which is underrepresented.
# Here, we'll create two files, one with even numbers and one with odd numbers,
# and then merge them to form a dataset with underrepresented odd numbers.
def make_df(b1_expr, num_events):
    return ROOT.RDataFrame(num_events).Define("b1", b1_expr).Define("b2", "(int) b1%2")


df_major = make_df("(int) 2 * rdfentry_", 100000)
df_minor = make_df("(int) 2 * rdfentry_ + 1", 1000)

batch_size = 256
num_epochs = 10

loss_fn = torch.nn.BCEWithLogitsLoss()


def train_model(model, optimizer, dataloader):
    train, val = dataloader.train_test_split(test_size=0.2)
    for _ in range(num_epochs):
        train_correct = 0
        train_total = 0
        train_losses = []
        model.train()
        for X, y in train.as_torch():
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            train_losses.append(loss.item())
        print(
            f"Training   => Accuracy: {int(train_correct / train_total * 100000) / 100000}; Loss: {int(sum(train_losses) / len(train_losses) * 100000) / 100000}"
        )
    val_losses = []
    val_correct = 0
    val_total = 0
    for X, y in val.as_torch():
        with torch.no_grad():
            outputs = model(X)
            loss = loss_fn(outputs, y)

            preds = (outputs > 0.5).float()
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)
            val_losses.append(loss.item())

    print(
        f"Validation => Accuracy: {int(val_correct / val_total * 100000) / 100000}; Loss: {int(sum(val_losses) / len(val_losses) * 100000) / 100000}\n"
    )


# First, let's try to create a dataloader without resampling and see how it handles the underrepresented class.
dl = ROOT.Experimental.ML.RDataLoader(
    [df_major, df_minor],
    batch_size=batch_size,
    target="b2",
    set_seed=seed,
    load_eager=True,
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
    target="b2",
    set_seed=seed,
    load_eager=True,  # Must be enabled for resampling
    sampling_type="oversampling",  # Can also be "undersampling"
    sampling_ratio=0.1,  # ~10% of the data will be from the underrepresented class
)

oversampling_model = torch.nn.Linear(1, 1)
oversampling_optimizer = torch.optim.Adam(oversampling_model.parameters())

print("Training with oversampling:")
train_model(oversampling_model, oversampling_optimizer, dl_oversampled)
