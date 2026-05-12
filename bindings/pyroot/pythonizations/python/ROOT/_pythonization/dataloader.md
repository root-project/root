\defgroup Py_ML Machine Learning Training
\ingroup Python
\brief Feed ROOT data directly into models for machine learning training.

# RDataLoader

`RDataLoader` is ROOT's bridge between HEP data and machine learning frameworks.
It lets you stream batches from any [RDataFrame](@ref Py_RDataFrame) directly into your models for training with no intermediate conversion or copies.

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

## Import

You can directly import `RDataLoader` from ROOT:

~~~{.py}
from ROOT.Experimental.ML import RDataLoader
~~~

## Pipeline

~~~
ROOT file (TTree / RNTuple)
    → RDataFrame (Filter, Define)
        → RDataLoader (batch, shuffle)
            → as_torch() / as_numpy() / as_tensorflow()
                → your model
~~~

## Quickstart

~~~{.py}
import ROOT
from ROOT.Experimental.ML import RDataLoader

rdf = ROOT.RDataFrame("events", "file.root")

dl = RDataLoader(rdf,
                 columns=["px", "py", "pz", "label"],
                 batch_size=64,
                 target="label",
                 shuffle=True)

for X, y in dl.as_torch():
    ...
~~~

## Two Loading Modes

### Lazy loading (default)

Data is read chunk by chunk from disk. Low memory usage, suitable for any dataset size.

~~~{.py}
dl = RDataLoader(rdf, batch_size=512)  # load_eager=False by default
~~~

### Eager loading

The full dataset is loaded into RAM upfront. Best when the dataset fits in memory
and you train for many epochs - eliminates per-epoch I/O overhead.
Required for resampling.

~~~{.py}
dl = RDataLoader(rdf, batch_size=512, load_eager=True)
~~~

## Iterating Batches

### as_torch()

Yields `torch.Tensor` batches.

~~~{.py}
# with target - classification
loss_fn = torch.nn.CrossEntropyLoss()
for X, y in dl.as_torch():
    optimizer.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    optimizer.step()

# move directly to GPU
for X, y in dl.as_torch(device="cuda"):
    loss = loss_fn(model(X), y)

# with event weights
for X, y, w in dl.as_torch():
    loss = (loss_fn(model(X), y) * w).mean()
~~~

### as_tensorflow()

Returns a `tf.data.Dataset` of `tf.Tensor` batches.

~~~{.py}
ds = dl.as_tensorflow()
model.fit(ds, epochs=10)

# chain tf.data transforms
ds = dl.as_tensorflow().prefetch(tf.data.AUTOTUNE).cache()
~~~

### as_numpy()
Yields `np.ndarray` batches.

~~~{.py}
for X, y in dl.as_numpy():
    print(X.shape)  # (64, 3) - np.ndarray
    clf.partial_fit(X, y)

# with event weights
for X, y, w in dl.as_numpy():
    clf.partial_fit(X, y, sample_weight=w)
~~~

## Train / Validation Split

~~~{.py}
train, val = dl.train_test_split(test_size=0.2)

for X, y in train.as_torch(device):
    ...  # training loop

for X, y in val.as_numpy():
    ...  # validation loop
~~~

\note Need a train / val / test three-way split? Call `train_test_split` twice:

~~~{.py}
# 20% for test
train_val, test = dl.train_test_split(test_size=0.2)

# 10% of total for val (0.125 × 0.8 = 0.10)
train, val = train_val.train_test_split(test_size=0.125)
~~~

## Resampling

Correct class imbalance by oversampling the minority or undersampling the majority.
Requires exactly two RDataFrames (minority first) and `load_eager=True`.

~~~{.py}
dl = RDataLoader(
    [rdf_signal, rdf_background],
    load_eager=True,
    sampling_type="oversampling",
    sampling_ratio=1.0,
)
~~~

## RVec / Variable-length Branches

Variable-length branches are flattened into fixed-width columns.
Declare the maximum size per branch - shorter vectors are zero-padded.

~~~{.py}
dl = RDataLoader(
    rdf,
    columns=["jets_pt", "jets_eta", "label"],
    max_vec_sizes={"jets_pt": 10, "jets_eta": 10},
    vec_padding=0.0,
    target="label",
)
# jets_pt expands to jets_pt_0 … jets_pt_9
# jets_eta expands to jets_eta_0 … jets_eta_9
~~~

\warning Every RVec column listed in `columns` must appear in `max_vec_sizes`.

## Tips

- **`batches_in_memory` ↑** - larger shuffle buffer, better randomisation across cluster boundaries, higher RAM use
- **`batches_in_memory` ↓** - smaller memory usage, shuffle limited to within one cluster
- **`load_eager=True`** - eliminates per-epoch I/O overhead; best when the dataset fits in RAM and you train for many epochs
- **`set_seed`** - set to a fixed integer for reproducible train/val splits and epoch shuffling
- **`drop_remainder=False`** - keep the last incomplete batch, useful when dataset size matters for validation metrics
- **`shuffle=False`** - deterministic order, useful for debugging or when data order carries meaning
- **`weights`** - requires `target` to be set, otherwise a `ValueError` is raised