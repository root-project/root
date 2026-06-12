## \file
## \ingroup tutorial_dataframe
## The Higgs to four lepton analysis from the ATLAS Open Data release of 2020, with RDataFrame.
##
## This tutorial is a continuation of the HiggsToFourLeptons tutorial.
## We will build a model to classify the data as Higgs or not Higgs.
##
##
## \macro_code
## \macro_output
##
## \date June 2026
## \authors Jonah Ascoli (CERN), Martin Foll (CERN, University of Oslo (UiO)), Silia Taider (CERN)

import matplotlib.pyplot as plt
import ROOT
import sklearn.metrics as skl
import torch
from matplotlib import use
from torch import nn

print("Loading dataframes...")
data_dir = ROOT.gROOT.GetTutorialDir().Data() + "/machine_learning/data/"
df_train = ROOT.RDataFrame("tree", data_dir + "ml_dataloader_Higgs_Classification_train.root")
df_val = ROOT.RDataFrame("tree", data_dir + "ml_dataloader_Higgs_Classification_val.root")
df_test = ROOT.RDataFrame("tree", data_dir + "ml_dataloader_Higgs_Classification_test.root")


# Classifier model with adjustable hidden layers
class Classifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_layers: list[int],
        p: float = 0.2,
        use_dropout: bool = False,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        layers = []
        in_dim = num_features

        for out_dim in hidden_layers:
            block = [nn.Linear(in_dim, out_dim)]

            if use_batchnorm:
                block.append(nn.BatchNorm1d(out_dim))

            block.append(nn.ReLU())

            if use_dropout:
                block.append(nn.Dropout(p))

            layers.append(nn.Sequential(*block))
            in_dim = out_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


batch_size = 1000
batches_in_memory = 1000
drop_remainder = True
columns = ["m4l", "good_lep", "goodlep_E", "goodlep_eta", "goodlep_phi", "goodlep_pt", "goodlep_type", "isHiggsRef"]
target = "isHiggsRef"
max_vec_sizes = {"good_lep": 4, "goodlep_E": 4, "goodlep_eta": 4, "goodlep_phi": 4, "goodlep_pt": 4, "goodlep_type": 4}
shuffle = True
set_seed = 42

# Normalize the data!
print("Normalizing data...")
for var in columns[:-1]:
    if var == "m4l":  # The only non-vector column
        mean = df_train.Mean(var).GetValue()
        stddev = df_train.StdDev(var).GetValue()
        df_train = df_train.Redefine(var, f"({var} - {mean}) / {stddev}")
        # The validation and testing data should be normalized based on the
        # mean and standard deviation calculated from the training data.
        df_val = df_val.Redefine(var, f"({var} - {mean}) / {stddev}")
        df_test = df_test.Redefine(var, f"({var} - {mean}) / {stddev}")
    else:
        # Each vector event has 4 columns, and we need to take a column-wise mean and stddev
        means = []
        stddevs = []
        for i in range(max_vec_sizes[var]):
            scalar_column = f"{var}_{i}"
            df_train = df_train.Define(scalar_column, f"{var}[{i}]")
            means.append(df_train.Mean(scalar_column).GetValue())
            stddevs.append(df_train.StdDev(scalar_column).GetValue())
        mean_vec = ROOT.RVec("double")(means)
        stddev_vec = ROOT.RVec("double")(stddevs)
        for i in range(len(stddevs)):
            if stddevs[i] == 0:
                stddevs[i] = 0.01  # Avoids division by 0
        expr = ", ".join(f"(({var}[{i}] - {means[i]}) / {stddevs[i]})" for i in range(max_vec_sizes[var]))
        df_train = df_train.Redefine(var, f"ROOT::RVec<double>{{{expr}}}")
        # The validation and testing data should be normalized based on the
        # mean and standard deviation calculated from the training data.
        df_val = df_val.Redefine(var, f"ROOT::RVec<double>{{{expr}}}")
        df_test = df_test.Redefine(var, f"ROOT::RVec<double>{{{expr}}}")

print("Creating dataloaders...")
train = ROOT.Experimental.ML.RDataLoader(
    df_train,
    batch_size=batch_size,
    batches_in_memory=batches_in_memory,
    drop_remainder=drop_remainder,
    columns=columns,
    target=target,
    max_vec_sizes=max_vec_sizes,
    shuffle=shuffle,
    set_seed=set_seed,
)
val = ROOT.Experimental.ML.RDataLoader(
    df_val,
    batch_size=batch_size,
    batches_in_memory=batches_in_memory,
    drop_remainder=drop_remainder,
    columns=columns,
    target=target,
    max_vec_sizes=max_vec_sizes,
    shuffle=shuffle,
    set_seed=set_seed,
)
test = ROOT.Experimental.ML.RDataLoader(
    df_test,
    batch_size=batch_size,
    batches_in_memory=batches_in_memory,
    drop_remainder=drop_remainder,
    columns=columns,
    target=target,
    max_vec_sizes=max_vec_sizes,
    shuffle=shuffle,
    set_seed=set_seed,
)

# num_features must be calculated manually since the train.training_columns includes condensed vector columns.
# Vector columns are lazily expanded while receiving batches, unless eager_loading is enabled.
num_features = sum(max_vec_sizes.values()) + len([0 for i in train.train_columns if i not in max_vec_sizes])

torch.manual_seed(set_seed)
hidden_layers = [60, 60]
model = Classifier(num_features=num_features, hidden_layers=hidden_layers, p=0.2, use_dropout=False)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def print_epoch_summary(epoch: int, val_loss: float, val_accuracy: float):
    print(f"Epoch {epoch} summary ==> Validation loss: {val_loss:.2f}; Validation accuracy: {val_accuracy:.2f}")


epochs = 1000
last_val_losses = [float("inf")] * 6
# Early stopping criterion: most recent 3 avg. losses are worse than the 3 before that
avg_val_losses = []
print("Starting training...")
for epoch in range(epochs):
    # training
    model.train()

    for i, (x_train, y_train) in enumerate(train.as_torch()):
        outputs = model(x_train)
        loss = loss_fn(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for j, (x_val, y_val) in enumerate(val.as_torch()):
            outputs = model(x_val)
            loss = loss_fn(outputs, y_val)
            val_loss += loss.item()

            preds = (outputs > 0.5).float()
            val_correct += (preds == y_val).sum().item()
            val_total += y_val.size(0)

    avg_val_loss = val_loss / (j + 1)
    avg_val_losses.append(avg_val_loss)
    val_accuracy = val_correct / val_total

    if epoch % 10 == 9:
        print_epoch_summary(epoch + 1, val_loss, val_accuracy)
    del last_val_losses[0]
    last_val_losses.append(avg_val_loss)
    # Early stopping check
    if min(last_val_losses[-3:]) > max(last_val_losses[:3]):
        print(f"Validation loss has not improved for 6 epochs, stopping training after {epoch + 1} epochs.")
        epochs = epoch + 1
        break

# Testing
model.eval()
test_loss = 0
test_correct = 0
test_total = 0

test_preds = []
test_true = []
with torch.no_grad():
    for j, (x_test, y_test) in enumerate(test.as_torch()):
        outputs = model(x_test)
        loss = loss_fn(outputs, y_test)
        test_loss += loss.item()
        test_preds += outputs.tolist()
        test_true += y_test.tolist()

        preds = (outputs > 0.5).float()
        test_correct += (preds == y_test).sum().item()
        test_total += y_test.size(0)

avg_test_loss = test_loss / (j + 1)
test_accuracy = test_correct / test_total

print(f"Testing Loss: {avg_test_loss:.4f}  Accuracy: {test_accuracy:.4f}\n")

# Analysis
use("Agg")  # Non-interactive backend for writing to files

fig = plt.figure()
ax = plt.axes()
ax.plot([i for i in range(epochs)], avg_val_losses)
plt.title("Loss curve")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.savefig("loss_curve")
print("Loss curve saved to loss_curve.png")

fpr, tpr, thresholds = skl.roc_curve(test_true, test_preds)
fig = plt.figure()
ax = plt.axes()
ax.plot(fpr[:-1], tpr[:-1])
plt.title("ROC curve")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig("ROC_curve")
print("ROC curve saved to ROC_curve.png")
