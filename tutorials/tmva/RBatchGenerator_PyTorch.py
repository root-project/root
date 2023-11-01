### \file
### \ingroup tutorial_tmva
### \notebook -nodraw
###
### Example of getting batches of events from a ROOT dataset into a basic
### PyTorch workflow.
###
### \macro_code
### \macro_output
### \author Dante Niewenhuis

import torch
import ROOT

tree_name = "sig_tree"
file_name = "http://root.cern/files/Higgs_data.root"

batch_size = 128
chunk_size = 5_000

target = "Type"

# Returns two generators that return training and validation batches
# as PyTorch tensors.
gen_train, gen_validation = ROOT.TMVA.Experimental.CreatePyTorchGenerators(
    tree_name,
    file_name,
    batch_size,
    chunk_size,
    target=target,
    validation_split=0.3,
)

# Get a list of the columns used for training
input_columns = gen_train.train_columns
num_features = len(input_columns)


def calc_accuracy(targets, pred):
    return torch.sum(targets == pred.round()) / pred.size(0)


# Initialize PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(num_features, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 300),
    torch.nn.Tanh(),
    torch.nn.Linear(300, 1),
    torch.nn.Sigmoid(),
)
loss_fn = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# Loop through the training set and train model
for i, (x_train, y_train) in enumerate(gen_train):
    # Make prediction and calculate loss
    pred = model(x_train).view(-1)
    loss = loss_fn(pred, y_train)

    # improve model
    model.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    accuracy = calc_accuracy(y_train, pred)

    print(f"Training => accuracy: {accuracy}")

#################################################################
# Validation
#################################################################

# Evaluate the model on the validation set
for i, (x_train, y_train) in enumerate(gen_validation):
    # Make prediction and calculate accuracy
    pred = model(x_train).view(-1)
    accuracy = calc_accuracy(y_train, pred)

    print(f"Validation => accuracy: {accuracy}")
