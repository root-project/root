\defgroup Py_ML RDataLoader
\ingroup Python
\brief Feed ROOT data directly into models for machine learning training.


`RDataLoader` streams ROOT data into machine learning frameworks as batches ready for training. It takes any @ref dataframe as input, giving you access to the full ROOT ecosystem for filtering, defining new variables and applying selections; it delivers batches of your dataset for [NumPy](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html), [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and [PyTorch](https://docs.pytorch.org/docs/main/tensors.html) through a simple iteration interface.

\note `RDataLoader` is part of `ROOT.Experimental.ML` and is currently experimental. The API may change between ROOT releases.

## Cheat Sheet

A one-page quick reference covering the API.

\htmlonly
<object data="rdataloader-cheatsheet.pdf"
        type="application/pdf"
        width="100%"
        height="520px"
        style="border:1px solid #ccc;border-radius:6px;">
  <p>PDF preview not available in your browser.</p>
</object>
<a href="rdataloader-cheatsheet.pdf"
   style="display:inline-block;margin-top:8px;padding:6px 14px;background:#1a73e8;
          color:#fff;border-radius:4px;text-decoration:none;font-size:13px;">
  ⬇ Download cheat sheet (PDF)
</a>
\endhtmlonly

## Getting your data ready

`RDataLoader` takes an @ref dataframe as input. This means your data preparation (selecting events, computing
new variables, applying cuts, etc.) all happens before the loader is created, using the full power of `RDataFrame`:

~~~{.py}
import math
import ROOT

# Open a ROOT file and create an RDataFrame
rdf = ROOT.RDataFrame("events", "file.root")

# Apply selections and compute derived features
rdf = rdf.Filter("nMuons >= 2") \
         .Define("inv_mass", "sqrt(E*E - p*p)")
~~~

Then pass your `RDataFrame` to `RDataLoader`:

~~~{.py}
from ROOT.Experimental.ML import RDataLoader

dl = RDataLoader(rdf,
                 columns=["inv_mass", "label"],
                 batch_size=64,
                 batches_in_memory=1000,
                 target="label")

# Iterate your batches as PyTorch tensors: X contains inv_mass, y contains label
for X, y in dl.as_torch():
    ...
~~~

The sections below explain how to configure the loader and get the most out of it.

## Configuring the RDataLoader

### Selecting columns and target

`columns` selects which branches to load. `target` names the label column, it is returned separately as `y` when you iterate, so you don't need to split it manually:

~~~{.py}
dl = RDataLoader(
    rdf,
    columns=["inv_mass", "pt", "eta", "label"],
    target="label",
    batch_size=256,
    batches_in_memory=1000
)
~~~

You can also pass multiple targets:

~~~{.py}
dl = RDataLoader(rdf,
                 columns=["x1", "x2", "x3", "y1", "y2"],
                 target=["y1", "y2"])

for X, y in dl.as_torch():
    # X.shape: (256, 3)
    # y.shape: (256, 2)
    ...
~~~

\warning `target` must appear in the `columns` list.

### Batch size and memory

`batch_size` controls how many events are in each batch. `batches_in_memory`
controls how many batches are held in the shuffle buffer at any time:

~~~{.py}
dl = RDataLoader(rdf,
                 batch_size=256,
                 batches_in_memory=20)  # default: 10
~~~

- **`batches_in_memory` ↑** - larger shuffle buffer, better randomisation, higher memory use
- **`batches_in_memory` ↓** - lower memory use, limited shuffle


### Shuffling and reproducibility

Shuffling is enabled by default. To make runs reproducible, fix the seed:

~~~{.py}
dl = RDataLoader(rdf, batch_size=256,
                 shuffle=True,
                 set_seed=42) # same order every run
~~~


### RVec / variable-length branches

ROOT branches that store variable-length arrays must be declared with a maximum size. Shorter entries are zero-padded and the branch is expanded into numbered columns:

~~~{.py}
dl = RDataLoader(
    rdf,
    columns=["jets_pt", "jets_eta", "label"],
    max_vec_sizes={"jets_pt": 10, "jets_eta": 10},
    vec_padding=0.0,
    target="label",
)
# jets_pt expands to jets_pt_0, jets_pt_1, … jets_pt_9
# events with fewer than 10 jets are zero-padded
~~~

\warning Every vector column in `columns` must appear in `max_vec_sizes`.

## Iterating Batches

### as_torch()

Yields `torch.Tensor` batches:

~~~{.py}
loss_fn   = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for X, y in dl.as_torch():
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
~~~

Move tensors to GPU by passing a device:

~~~{.py}
for X, y in dl.as_torch(device="cuda"):
    ...
~~~

### as_tensorflow()

Returns a `tf.data.Dataset` of `tf.Tensor` batches:

~~~{.py}
model.fit(dl.as_tensorflow(), epochs=10)
~~~

### as_numpy()

Yields `np.ndarray` batches:

~~~{.py}
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier()
for X, y in dl.as_numpy():
    clf.partial_fit(X, y, classes=[0, 1])
~~~

## Train / Validation Split

Pass `test_size` to split the dataset into two loaders each representing a fraction of the original dataset (no data is duplicated):

~~~{.py}
train, val = dl.train_test_split(test_size=0.2)

for epoch in range(num_epochs):
    model.train()
    for X, y in train.as_torch(device):
        ...

    model.eval()
    for X, y in val.as_torch(device):
        ...
~~~

\note Need a three-way train / val / test split? Call `train_test_split` twice:

~~~{.py}
train_val, test = dl.train_test_split(test_size=0.15)
train, val = train_val.train_test_split(test_size=0.176)
# 0.176 × 0.85 ≈ 15% of the total
~~~

## Advanced Features

### Eager loading

By default the loader reads data lazily, one chunk of data at a time. For small datasets that fit in memory and will be iterated many times, eager loading pays a one-time cost at construction and then serves batches every epoch from memory:

~~~{.py}
dl = RDataLoader(rdf, batch_size=256, load_eager=True)
~~~

### Resampling

Correct class imbalance by oversampling the minority or undersampling the majority. You can do this by passing two RDataFrames:

~~~{.py}
dl = RDataLoader(
    [rdf_signal, rdf_background],
    columns=["inv_mass", "label"],
    target="label",
    batch_size=256,
    batches_in_memory=1000,
    load_eager=True,
    sampling_type="oversampling", # or "undersampling"
    sampling_ratio=1.0,
)
~~~

\warning This feature is only available in eager loading mode (`load_eager=True`).

### Event weights

If your dataset has a weight column, pass its name to `weights`. It is returned as a third value `w` alongside `X` and `y`:

~~~{.py}
dl = RDataLoader(rdf,
                 columns=["inv_mass", "label", "weight"],
                 target="label",
                 weights="weight")

for X, y, w in dl.as_torch():
    loss = (loss_fn(model(X), y) * w).mean()
~~~
