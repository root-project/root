\defgroup Py_ML RDataLoader
\ingroup Python
\brief Feed ROOT data directly into models for machine learning training.


`RDataLoader` streams ROOT data into machine learning frameworks as batches ready for training. It reads any TTree or RNTuple, from local or remote files or from an @ref dataframe, giving you access to the full ROOT ecosystem for filtering, defining new variables and applying selections; it delivers batches of your dataset for [NumPy](https://numpy.org/devdocs/reference/generated/numpy.ndarray.html), [PyTorch](https://docs.pytorch.org/docs/main/tensors.html), [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and [JAX](https://docs.jax.dev/en/latest/_autosummary/jax.Array.html) through a simple iteration interface.

\note `RDataLoader` is part of `ROOT.Experimental.ML` and is currently experimental. The API may change between ROOT releases, and feedback is much appreciated!

\htmlonly
<style>
/* "Tip" boxes styled like the note and warning boxes */
dl.section.user {
  background: var(--invariant-color-bg);
  border-left: 8px solid var(--invariant-color-hl);
  color: var(--invariant-color-text);
  padding: 8px 12px;
  border-radius: 4px;
}
dl.section.user dt { color: var(--invariant-color-hl); }
dl.section.user dd { margin-inline-start: 0; }
</style>
\endhtmlonly

\anchor ml_quickstart
# Quickstart

~~~{.py}
from ROOT.Experimental.ML import RDataLoader

# read the TTree or RNTuple "events" from a file, and return batches of 256 entries
dl = RDataLoader(("events", "data.root"), batch_size=256, target="label")

# 80% of the entries for training, 20% for validation
train, val = dl.train_test_split(test_size=0.2)

for X, y in train.as_torch():
    ...  # X holds all feature columns, y the "label" column
~~~

The loader reads the file in the background while you train, shuffles the entries every epoch and hands them over as tensors. The rest of this page goes through each step in detail.


\anchor ml_cheatsheet
# Cheat sheet

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


\anchor ml_walkthrough
# Walkthrough: from ROOT files to a trained model

This walkthrough trains a small PyTorch classifier to separate a signal from a background sample, stored in two files. Each step points to the section of this page that explains it further.

### 1. Prepare the data with RDataFrame

Selections, new variables and labels are defined with @ref dataframe, before the data reaches the loader. Here, each sample gets a `label` column: 1 for signal, 0 for background.

~~~{.py}
import ROOT

def prepare(filename, label):
    df = ROOT.RDataFrame("Events", filename)
    return (
        df.Filter("nMuon >= 2", "At least two muons")
        .Define("Muon_pt_1", "Muon_pt[0]")
        .Define("Muon_pt_2", "Muon_pt[1]")
        .Define("label", f"{label}.0")
    )

df_sig = prepare("signal.root", 1)
df_bkg = prepare("background.root", 0)
~~~

\par Tip
Only the columns you give to the loader are read from the files, and the `Filter`s run before any data is copied internally. Selecting events and columns in RDataFrame is therefore the cheapest way to reduce how much data the loader processes.

### 2. Create the loader

Both samples are combined into one stream of batches. `columns` lists the features, `target` the label (see [Columns, targets and weights](#ml_columns)):

~~~{.py}
from ROOT.Experimental.ML import RDataLoader

dl = RDataLoader(
    [df_sig, df_bkg],
    columns=["Muon_pt_1", "Muon_pt_2"],
    target="label",
    batch_size=512,
    set_seed=42,  # same split and same shuffling at every run
)
~~~

### 3. Split into training and validation datasets

~~~{.py}
train, val = dl.train_test_split(test_size=0.2)
print(f"{train.num_batches} training and {val.num_batches} validation batches per epoch")
~~~

See [Training and validation](#ml_split).

### 4. Train

Each `for` loop over a loader is one epoch, and the entries are re-shuffled at the start of every epoch:

~~~{.py}
import torch

model = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    model.train()
    for X, y in train.as_torch(): # here we choose the output format of the batches (`as_torch()`)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)  # y has the shape (batch_size, 1), like the model output
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(X), y).item() for X, y in val.as_torch()) / val.num_batches
    print(f"epoch {epoch}: validation loss {val_loss:.4f}")
~~~

\par Tip
Batches are `float32`, which is what most models expect. For a loss that needs integer class indices, like `torch.nn.CrossEntropyLoss`, convert the target with `y.long().squeeze(1)` (this casts the Pytorch tensor to a 64-bit integer and removes a dimension of size 1 at index 1).

### 5. Save the datasets for later (optional)

The training and validation datasets can be written to disk exactly as they were loaded, e.g. to share them or to train other models on the same split (see [Saving datasets](#ml_saving)):

~~~{.py}
train.save("Events", "train.root")
val.save("Events", "val.root")
~~~

The [tutorials](#ml_tutorials) show complete examples, with NumPy, PyTorch, TensorFlow and XGBoost.


\anchor ml_inputs
# Inputs

`RDataLoader` takes the dataset to read as its first argument. All the other arguments are keyword-only.

| Input | Reads |
|-------|-------|
| `("events", "data.root")` | the TTree or RNTuple `events` from one file |
| `("events", ["data_1.root", "data_2.root"])`, `("events", "data_*.root")` | `events` from several files, as one dataset |
| `ROOT.RDataFrame(...)`, with any `Filter` and `Define` | the entries and columns of the dataframe |
| `[("signal", "sig.root"), ("background", "bkg.root")]`, `[df1, df2]` | several datasets, one after the other in the same stream of batches |

A `(dataset_name, files)` tuple is opened like `ROOT.RDataFrame(dataset_name, files)`. The files can be given as strings or `pathlib.Path` objects, and can be remote:

~~~{.py}
from pathlib import Path

dl = RDataLoader(("events", Path("data") / "sample.root"), batch_size=256)
dl = RDataLoader(("Events", "root://eospublic.cern.ch//eos/opendata/sample.root"), batch_size=256)
~~~

Pass an RDataFrame instead as soon as you need to prepare the data, e.g. to apply selections or to define the target column:

~~~{.py}
df = ROOT.RDataFrame("events", "data.root").Filter("nJets >= 2").Define("inv_mass", "sqrt(E*E - p*p)")
dl = RDataLoader(df, columns=["inv_mass", "pt"], target="label", batch_size=256)
~~~

\warning All the datasets given in a list must provide the requested columns, with the same types: the columns are resolved on the first one.

\par Tip
Tuples and RDataFrames can be mixed in the same list, e.g. a file read as it is together with a dataframe that defines extra columns.


\anchor ml_columns
# Columns, targets and weights

### Features and targets

`columns` selects the feature columns to load, and `target` names the column (or list of columns) to predict. When iterating, the targets are returned separately as `y`, so you don't need to split them manually:

~~~{.py}
dl = RDataLoader(df, columns=["x1", "x2", "x3"], target=["y1", "y2"], batch_size=256)

for X, y in dl.as_numpy():
    # X.shape: (256, 3)
    # y.shape: (256, 2)
    ...
~~~

If `columns` is not given, all the columns of the dataset are used as features, except for the target and weights columns, you can inspect the loader to check yourself:

~~~{.py}
dl.feature_columns  # ["x1", "x2", "x3"]
dl.target_columns   # ["y1", "y2"]
dl.columns          # all of them, in the order they appear in the batches
~~~

Without a target, each batch is a single array holding all the columns.

### Event weights

If your dataset has a weight column, pass its name to `weights`. It is returned as a third value `w`, with the shape `(batch_size, 1)`:

~~~{.py}
dl = RDataLoader(df, columns=["inv_mass", "pt"], target="label", weights="weight", batch_size=256)

for X, y, w in dl.as_torch():
    loss = (loss_fn(model(X), y) * w).mean()
    ...
~~~

\warning `weights` can only be used together with a `target`.

### Vector columns

Columns holding a variable number of values per entry, such as `ROOT::RVec` or `std::vector`, are loaded as a fixed number of features. Give their maximum size in `max_vec_sizes`: shorter vectors are padded with `vec_padding` (0 by default) and longer ones are truncated.

~~~{.py}
dl = RDataLoader(
    df,
    columns=["jets_pt", "jets_eta", "nJets"],
    target="label",
    max_vec_sizes={"jets_pt": 4, "jets_eta": 4},
    vec_padding=-999.0,
    batch_size=256,
)

train, val = dl.train_test_split(test_size=0.2)
train.feature_columns  # ["jets_pt_0", ..., "jets_pt_3", "jets_eta_0", ..., "jets_eta_3", "nJets"]
~~~

\warning Every vector column that is loaded must appear in `max_vec_sizes`.

\par Tip
Choose a padding value that cannot be mistaken for a real value, and keep the number of entries in a separate column (like `nJets` above) so that the model can tell real values from padding.

### Preparing the features

Any transformation of the features can be done with RDataFrame before loading, e.g. to normalize them:

~~~{.py}
df_train = ROOT.RDataFrame("events", "train.root")
df_test = ROOT.RDataFrame("events", "test.root")

# the constants are computed on the training data, and applied to both datasets
mean = df_train.Mean("pt").GetValue()
std = df_train.StdDev("pt").GetValue()
df_train = df_train.Redefine("pt", f"(pt - {mean}) / {std}")
df_test = df_test.Redefine("pt", f"(pt - {mean}) / {std}")
~~~

\par Tip
Compute the normalization constants on the training data only, so that no information from the validation or test data leaks into the training. The ml_dataloader_Higgs_Classification.py tutorial normalizes scalar and vector columns this way.


\anchor ml_split
# Training and validation datasets

`train_test_split()` splits the dataset into two loaders, each holding a fraction of the entries:

~~~{.py}
train, val = dl.train_test_split(test_size=0.2)

for epoch in range(num_epochs):
    model.train()
    for X, y in train.as_torch():
        ...

    model.eval()
    for X, y in val.as_torch():
        ...
~~~

The validation entries are rounded down: with 1001 entries and `test_size=0.2`, 200 entries are used for validation and 801 for training. Without shuffling, the last entries are used for validation.

\par Tip
The split is drawn at random unless `set_seed` is given. Fix the seed whenever you split, so that the validation entries are the same every time you run, e.g. when resuming a training or evaluating a model later.

### Three datasets: training, validation and test

To also keep a test dataset that is never seen during the training, there are two options. In both examples below, 60% of the events are used for training, 20% for validation and 20% for the test.

**With a filter.** Select the test events with a `Filter`, e.g. on the event number, and give them their own loader:

~~~{.py}
df = ROOT.RDataFrame("events", "data.root")

dl = RDataLoader(df.Filter("event % 5 != 0"), target="label", batch_size=256, set_seed=42)
train, val = dl.train_test_split(test_size=0.25)

test = RDataLoader(df.Filter("event % 5 == 0"), target="label", batch_size=256, shuffle=False)
~~~

The test events are then always the same, whatever the seed, and nothing is written to disk.

**By splitting twice.** Split off the test dataset first, save the rest with `save()`, and split the saved dataset again into training and validation:

~~~{.py}
# 1. keep 20% of the events for the test, and save the other 80%
dl = RDataLoader(df, target="label", batch_size=256, set_seed=42, drop_remainder=False)
train_val, test = dl.train_test_split(test_size=0.2)
train_val.save("events", "train_val.root")

# 2. split the saved 80% into training (60%) and validation (20%)
dl = RDataLoader(("events", "train_val.root"), target="label", batch_size=256, set_seed=42)
train, val = dl.train_test_split(test_size=0.25)
~~~

The events are then selected at random, and no column is needed to select them.

\warning The second split must be done on a new loader. Calling `train_test_split()` on a loader returned by `train_test_split()` returns the same two datasets again, whatever the `test_size`.

\par Tip
Create the first loader with `drop_remainder=False`, otherwise `save()` drops the entries of the last incomplete batch (see [Saving datasets](#ml_saving)). The `test` dataset can be saved the same way, to evaluate the model later or with other tools. Vector columns are saved padded to their maximum size, so give the same `max_vec_sizes` to the second loader.

Training, validation and test can also simply be read from different files.


\anchor ml_iterating
# Iterating batches

The batches can be returned in four formats.

| Method | Batches |
|--------|---------|
| `as_numpy()` | `numpy.ndarray` |
| `as_torch(device=None)` | `torch.Tensor`, optionally moved to a device such as `"cuda"` |
| `as_tensorflow()` | a `tf.data.Dataset` of `tf.Tensor` |
| `as_jax(device=None)` | `jax.Array`, optionally moved to a device such as `"gpu"` |

All the values are `float32`. With a single target or weights column, `y` and `w` have the shape `(batch_size, 1)`.

### PyTorch

~~~{.py}
device = "cuda" if torch.cuda.is_available() else "cpu"

for X, y in train.as_torch(device=device):
    ...
~~~

### TensorFlow and Keras

`as_tensorflow()` returns a `tf.data.Dataset`, which can be passed to Keras directly:

~~~{.py}
model.fit(train.as_tensorflow(), validation_data=val.as_tensorflow(), epochs=10)
~~~

\warning The TensorFlow dataset has a fixed batch shape: with `drop_remainder=False`, the last batch is completed with rows of zeros. Keep the default `drop_remainder=True`, or use a `weights` column so that the added rows get a weight of zero.

### JAX

~~~{.py}
for X, y in train.as_jax(device="gpu"):
    ...
~~~

### NumPy, scikit-learn and XGBoost

~~~{.py}
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier()
for X, y in train.as_numpy():
    clf.partial_fit(X, y.ravel(), classes=[0, 1])
~~~

\par Tip
Models that need all the data at once, like XGBoost, can get the whole dataset as a single batch: set `batch_size` to the number of entries and `drop_remainder=False`, then take it with `X, y = next(iter(train.as_numpy()))`. See the ml_dataloader_XGBoost.py tutorial.

### Transforming batches

Transform each batch with Python's `map`:

~~~{.py}
def transform(batch):
    X, y = batch
    return X * 2, y

for X, y in map(transform, train.as_numpy()):
    ...
~~~

For a TensorFlow dataset, use `Dataset.map`:

~~~{.py}
ds = train.as_tensorflow().map(lambda X, y: (X * 2, y))
model.fit(ds, ...)
~~~

### The last batch

When the number of entries is not a multiple of `batch_size`, the remaining entries are dropped by default. Pass `drop_remainder=False` to get them in a smaller last batch. `num_batches` and `last_batch_no_of_rows` tell how many batches an epoch has and how many rows the last one holds; they are available once the loader is split or iterated.


\anchor ml_loading
# Lazy or eager loading

By default, the loader reads the data lazily: it loads a few clusters of entries at a time in a background thread while you train, so that the memory usage does not depend on the size of the dataset. With `load_eager=True`, the whole dataset is loaded into memory at the start, and every epoch is served from memory.

| | Lazy (default) | Eager (`load_eager=True`) |
|-|----------------|---------------------------|
| Memory | about `batch_size × batches_in_memory` entries | the whole dataset |
| First epoch | starts right away | starts when the dataset is loaded |
| Next epochs | read subsets of the files again | served from memory |
| Shuffling | within a buffer of clusters | across the whole dataset |
| Resampling | not available | available |

\par Tip
If your dataset fits in memory and you train for many epochs, eager loading is usually the fastest option.

The memory used in lazy loading is controlled with `batches_in_memory`, the approximate number of batches held in memory at a time (10 by default):

- **`batches_in_memory` ↑**: larger shuffle buffer, better randomization, higher memory usage
- **`batches_in_memory` ↓**: lower memory usage, less randomization


\anchor ml_shuffling
# Shuffling and reproducibility

The entries are shuffled by default, and shuffled differently at every epoch. The seed makes runs reproducible:

~~~{.py}
dl = RDataLoader(df, batch_size=256, set_seed=42)  # same split and same order at every run
~~~

With the default `set_seed=0`, a random seed is drawn at every run.

To make reading efficient, lazy loading follows the layout of the data on disk: at every epoch, it reads the clusters of entries in a random order, and shuffles the entries of a few clusters together, in a buffer of about `batches_in_memory` batches. Eager loading shuffles the entries across the whole dataset.

\par Tip
If the files are sorted, e.g. when all the signal entries come before the background ones, increase `batches_in_memory` in lazy loading so that each buffer mixes more clusters, or use eager loading.

Pass `shuffle=False` to read the entries in their order on disk, e.g. for the evaluation of a model.


\anchor ml_resampling
# Class imbalance: resampling

When one class has many more entries than the other, the loader can rebalance them at every epoch, by drawing fewer entries of the majority class (undersampling) or more entries of the minority class (oversampling). Give the two classes as two datasets:

~~~{.py}
dl = RDataLoader(
    [df_signal, df_background],
    columns=["inv_mass", "pt"],
    target="label",
    batch_size=256,
    load_eager=True,               # required for resampling
    sampling_type="oversampling",  # or "undersampling"
    sampling_ratio=1.0,            # number of minority / majority entries in each epoch
)
~~~

The dataset with the fewest entries is taken as the minority class. With `sampling_type="undersampling"`, `replacement=True` lets the majority entries be drawn more than once.

\warning Resampling is only available with eager loading (`load_eager=True`), and needs exactly two input datasets.

\note After a split, the validation dataset is resampled too. To evaluate a model on the real class proportions, use a separate loader without resampling.

\par Tip
Instead of resampling, the classes can also be balanced with a weight column, which works in both loading modes, as in the ml_dataloader_XGBoost.py tutorial.


\anchor ml_saving
# Saving datasets

`save()` writes a dataset to disk exactly as it would be loaded for training: after the filters, the split, the shuffling and the padding of the vector columns. The output is a TTree by default, or an RNTuple:

~~~{.py}
train, val = dl.train_test_split(test_size=0.2)

train.save("events", "train.root")
val.save("events", "val.root", output_format="rntuple")
~~~

The column names are kept, and vector columns are written with their fixed, padded size, so a saved file can be loaded back by a new `RDataLoader` with the same `columns` and `target`:

~~~{.py}
train = RDataLoader(("events", "train.root"), columns=["jets_pt", "nJets"], target="label",
                    max_vec_sizes={"jets_pt": 4}, batch_size=256)
~~~

\par Tip
`save()` follows the `batch_size` and `drop_remainder` of the loader: create it with `drop_remainder=False` to write all the entries. Saving is also a good way to run an expensive preparation of the data once, and train many times on the result.


\anchor ml_performance
# Performance tips

- **Read only what you need.** Only the selected columns are read from the files: list them in `columns` instead of loading all the columns of a dataset.
- **Prepare expensive inputs once.** In lazy loading, the `Filter`s and `Define`s of the RDataFrame run again at every epoch. If they are expensive, use eager loading, or save the prepared dataset with `save()` (or `Snapshot`) and train on the saved file.
- **Use eager loading for small datasets.** If the dataset fits in memory, eager loading pays a one-time cost and then serves every epoch from memory.
- **Tune the batch size.** Larger batches mean fewer Python iterations per epoch. `batches_in_memory` trades memory for shuffling quality.

\warning `RDataLoader` does not support `ImplicitMT` yet, and raises a `RuntimeError` if it is enabled. You can still use multi-threading in the rest of your script, e.g. to prepare and save the data with `Snapshot`, as long as you call `ROOT.DisableImplicitMT()` before creating the loader and its inputs.

\anchor ml_tutorials
# Tutorials

| Tutorial | Shows |
|----------|-------|
| ml_dataloader_NumPy.py | Loading batches as NumPy arrays. |
| ml_dataloader_PyTorch.py | A basic PyTorch training loop. |
| ml_dataloader_TensorFlow.py | A basic TensorFlow training loop. |
| ml_dataloader_filters_vectors.py | Filtered datasets and vector columns. |
| ml_dataloader_resampling.py | Balancing an imbalanced dataset with oversampling and undersampling. |
| ml_dataloader_Higgs_Classification.py | Normalizing the features and training a classifier on training, validation and test files. |
| ml_dataloader_XGBoost.py | Training an XGBoost classifier from remote files, with class-balancing weights. |

The full API is described below.
