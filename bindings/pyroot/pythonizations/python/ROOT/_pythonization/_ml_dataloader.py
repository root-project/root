# Author: Dante Niewenhuis, VU Amsterdam 07/2023
# Author: Kristupas Pranckietis, Vilnius University 05/2024
# Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
# Author: Vincenzo Eduardo Padulano, CERN 10/2024
# Author: Martin Føll, University of Oslo (UiO) & CERN 01/2026
# Author: Silia Taider, CERN 02/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING, Any, Callable, Tuple

if TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf
    import torch

    import ROOT


class _RDataLoader:
    def get_template(
        self,
        x_rdf: ROOT.RDF.RNode,
        columns: list[str] | None = None,
        max_vec_sizes: dict[str, int] | None = None,
    ) -> Tuple[str, list[int]]:
        """
        Generate a template for the DataLoader based on the given
        RDataFrame and columns.

        Args:
            x_rdf (RNode): RDataFrame or RNode object.
            columns (list[str], optional): Columns that should be loaded.
                                 Defaults to loading all columns
                                 in the given RDataFrame
            max_vec_sizes (dict[str, int], optional):
                                 Mapping from vector column name
                                 to the maximum size of the vector.
                                 Required when using vector based columns.

        Returns:
            Tuple[str, list[int]]: Template string for the DataLoader and list of max vector sizes
        """
        if not columns:
            columns = x_rdf.GetColumnNames()
        if max_vec_sizes is None:
            max_vec_sizes = {}

        template_string = ""

        self.given_columns = []
        self.all_columns = []

        max_vec_sizes_list = []

        for name in columns:
            name_str = str(name)
            self.given_columns.append(name_str)
            column_type = x_rdf.GetColumnType(name_str)
            template_string = f"{template_string}{column_type},"

            if "RVec" in column_type:
                # Add column for each element if column is a vector
                if name_str in max_vec_sizes:
                    max_vec_sizes_list.append(max_vec_sizes[name_str])
                    for i in range(max_vec_sizes[name_str]):
                        self.all_columns.append(f"{name_str}_{i}")

                else:
                    raise ValueError(
                        f"No max size given for feature {name_str}. \
                        Given max sizes: {max_vec_sizes}"
                    )

            else:
                self.all_columns.append(name_str)

        return template_string[:-1], max_vec_sizes_list

    def __init__(
        self,
        rdataframes: ROOT.RDF.RNode | list[ROOT.RDF.RNode] | None = None,
        batch_size: int = 0,
        batches_in_memory: int = 1,
        columns: list[str] | None = None,
        max_vec_sizes: dict[str, int] | None = None,
        vec_padding: int = 0,
        target: str | list[str] | None = None,
        weights: str = "",
        test_size: float = 0.0,
        shuffle: bool = True,
        drop_remainder: bool = True,
        set_seed: int = 0,
        load_eager: bool = False,
        sampling_type: str = "",
        sampling_ratio: float = 1.0,
        replacement: bool = False,
    ) -> None:
        """Wrapper around the C++ DataLoader

        Args:
            rdataframes (ROOT.RDF.RNode | list[ROOT.RDF.RNode] | None):
                RDataFrame or list of RDataFrames to load from.
            batch_size (int):
                Number of entries per batch returned by the generator.
            batches_in_memory (int):
                Approximate number of batches that should be kept in memory at
                the same time. Higher value results in faster loading, but
                also higher memory usage. Defaults to 1.
            columns (list[str] | None):
                Names of columns to load. If not given, all columns are used.
            max_vec_sizes (dict[str, int] | None):
                Mapping from vector column name to the maximum size of the vector.
                Required when using vector based columns.
            vec_padding (int):
                Value used to pad vectors with if the vector is smaller
                than the given max vector length. Defaults to 0.
            target (str | list[str] | None):
                Name or list of names of target column(s).
            weights (str):
                Column used to weight events.
                Can only be used when a target is given.
            test_size (float):
                The ratio of batches being kept for validation.
                Value has to be between 0 and 1. Defaults to 0.0.
            shuffle (bool):
                Batches consist of random events and are shuffled every epoch.
                Defaults to True.
            drop_remainder (bool):
                Drop the remainder of data that is too small to compose full batch.
                Defaults to True.
            set_seed (int):
                For reproducibility: Set the seed for the random number generator used
                to split the dataset into training and validation as well as shuffing of the entries.
                Defaults to 0 which means that the seed is set to the random device.
            load_eager (bool):
                If True, load the full dataset(s) into memory.
                If False, load data lazily in clusters. Defaults to False.
            sampling_type (str):
                Describes the mode of sampling from the minority and majority dataframes.
                Supported values are ``"undersampling"`` and ``"oversampling"``. Requires ``load_eager=True``.
                Defaults to ``""``.
                For 'undersampling' and 'oversampling' it requires a list of exactly two dataframes as input,
                where the dataframe with the most entries is the majority dataframe
                and the dataframe with the fewest entries is the minority dataframe.
            sampling_ratio (float):
                Ratio of minority and majority entries in the resampled dataset.
                Requires ``load_eager=True`` and ``sampling_type="undersampling"`` or ``"oversampling"``. Defaults to 1.0.
            replacement (bool):
                Whether the sampling is with (True) or without (False) replacement.
                Requires ``load_eager=True`` and ``sampling_type="undersampling"``. Defaults to False.
        """

        from ROOT import RDF

        if rdataframes is None:
            rdataframes = []
        if columns is None:
            columns = []
        if max_vec_sizes is None:
            max_vec_sizes = {}
        if target is None or target == "":
            target = []

        if not hasattr(rdataframes, "__iter__"):
            rdataframes = [rdataframes]
        self.noded_rdfs = [RDF.AsRNode(rdf) for rdf in rdataframes]

        if isinstance(target, str):
            target = [target]

        self.target_columns = target
        self.weights_column = weights

        template, max_vec_sizes_list = self.get_template(self.noded_rdfs[0], columns, max_vec_sizes)

        self.num_columns = len(self.all_columns)
        self.batch_size = batch_size

        # Handle target
        self.target_given = len(self.target_columns) > 0
        self.weights_given = len(self.weights_column) > 0
        if self.target_given:
            for target in self.target_columns:
                if target not in self.all_columns:
                    raise ValueError(
                        f"Provided target not in given columns: \ntarget => \
                            {target}\ncolumns => {self.all_columns}"
                    )

            self.target_indices = [self.all_columns.index(target) for target in self.target_columns]

            # Handle weights
            if self.weights_given:
                if weights in self.all_columns:
                    self.weights_index = self.all_columns.index(self.weights_column)
                    self.train_indices = [
                        c for c in range(len(self.all_columns)) if c not in self.target_indices + [self.weights_index]
                    ]
                else:
                    raise ValueError(
                        f"Provided weights not in given columns: \nweights => \
                            {weights}\ncolumns => {self.all_columns}"
                    )
            else:
                self.train_indices = [c for c in range(len(self.all_columns)) if c not in self.target_indices]

        elif self.weights_given:
            raise ValueError("Weights can only be used when a target is provided")
        else:
            self.train_indices = [c for c in range(len(self.all_columns))]

        self.train_columns = [c for c in self.all_columns if c not in self.target_columns + [self.weights_column]]

        import ROOT

        # The DataLoader will create a separate C++ thread for I/O.
        # Enable thread safety in ROOT from here, to make sure there is no
        # interference between the main Python thread (which might call into
        # cling via cppyy) and the I/O thread.
        ROOT.EnableThreadSafety()

        self.engine = ROOT.Experimental.Internal.ML.RDataLoaderEngine(template)(
            self.noded_rdfs,
            batch_size,
            batches_in_memory,
            self.given_columns,
            max_vec_sizes_list,
            vec_padding,
            test_size,
            shuffle,
            drop_remainder,
            set_seed,
            load_eager,
            sampling_type,
            sampling_ratio,
            replacement,
        )

        atexit.register(self.DeActivate)

    @property
    def isActive(self):
        return self.engine.IsActive()

    def isTrainingActive(self):
        return self.engine.IsTrainingActive()

    def isValidationActive(self):
        return self.engine.IsValidationActive()

    def Activate(self):
        """Initialize the generator to be used for a loop, this spawns the loading thread"""
        self.engine.Activate()

    def DeActivate(self):
        """Deactivate the generator"""
        self.engine.DeActivate()

    def ActivateTrainingEpoch(self):
        """Activate the training epoch of the generator"""
        self.engine.ActivateTrainingEpoch()

    def ActivateValidationEpoch(self):
        """Activate the validation epoch of the generator"""
        self.engine.ActivateValidationEpoch()

    def DeActivateTrainingEpoch(self):
        """Deactivate the training epoch of the generator"""
        self.engine.DeActivateTrainingEpoch()

    def DeActivateValidationEpoch(self):
        """Deactivate the validation epoch of the generator"""
        self.engine.DeActivateValidationEpoch()

    def CreateTrainBatches(self):
        """Create the first training batches from the first cluster"""
        self.engine.CreateTrainBatches()

    def CreateValidationBatches(self):
        """Create the first validation batches from the first cluster"""
        self.engine.CreateValidationBatches()

    @property
    def num_training_batches(self) -> int:
        return self.engine.NumberOfTrainingBatches()

    @property
    def num_validation_batches(self) -> int:
        return self.engine.NumberOfValidationBatches()

    @property
    def train_remainder_rows(self) -> int:
        return self.engine.TrainRemainderRows()

    @property
    def val_remainder_rows(self) -> int:
        return self.engine.ValidationRemainderRows()

    def GetSample(self):
        """
        Return a sample of data that has the same size and types as the actual
        result. This sample can be used to define the shape and size of the
        output

        Returns:
            np.ndarray: data sample
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("Failed to import numpy needed for the ML dataloader")

        # Split the target and weight
        if not self.target_given:
            return np.zeros((self.batch_size, self.num_columns))

        if not self.weights_given:
            if len(self.target_indices) == 1:
                return np.zeros((self.batch_size, self.num_columns - 1)), np.zeros((self.batch_size)).reshape(-1, 1)

            return np.zeros((self.batch_size, self.num_columns - 1)), np.zeros(
                (self.batch_size, len(self.target_indices))
            )

        if len(self.target_indices) == 1:
            return (
                np.zeros((self.batch_size, self.num_columns - 2)),
                np.zeros((self.batch_size)).reshape(-1, 1),
                np.zeros((self.batch_size)).reshape(-1, 1),
            )

        return (
            np.zeros((self.batch_size, self.num_columns - 2)),
            np.zeros((self.batch_size, len(self.target_indices))),
            np.zeros((self.batch_size)).reshape(-1, 1),
        )

    def ConvertBatchToNumpy(self, batch) -> np.ndarray:
        """Convert a RTensor into a NumPy array

        Args:
            batch (RTensor): Batch returned from the DataLoader

        Returns:
            np.ndarray: converted batch
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("Failed to import numpy needed for the ML dataloader")

        data = batch.GetData()
        batch_size, num_columns = tuple(batch.GetShape())

        data.reshape((batch_size * num_columns,))

        return_data = np.asarray(data).reshape(batch_size, num_columns)

        # Splice target column from the data if target is given
        if self.target_given:
            train_data = return_data[:, self.train_indices]
            target_data = return_data[:, self.target_indices]

            # Splice weight column from the data if weight is given
            if self.weights_given:
                weights_data = return_data[:, self.weights_index]

                if len(self.target_indices) == 1:
                    return train_data, target_data.reshape(-1, 1), weights_data.reshape(-1, 1)

                return train_data, target_data, weights_data.reshape(-1, 1)

            if len(self.target_indices) == 1:
                return train_data, target_data.reshape(-1, 1)

            return train_data, target_data

        return return_data

    def ConvertBatchToPyTorch(self, batch: Any, device=None) -> torch.Tensor:
        """Convert a RTensor into a PyTorch tensor

        Args:
            batch (RTensor): Batch returned from the DataLoader

        Returns:
            torch.Tensor: converted batch
        """
        import numpy as np
        import torch

        data = batch.GetData()
        batch_size, num_columns = tuple(batch.GetShape())

        data.reshape((batch_size * num_columns,))

        return_data = torch.as_tensor(np.asarray(data), device=device).reshape(batch_size, num_columns)

        # Splice target column from the data if target is given
        if self.target_given:
            train_data = return_data[:, self.train_indices]
            target_data = return_data[:, self.target_indices]

            # Splice weight column from the data if weight is given
            if self.weights_given:
                weights_data = return_data[:, self.weights_index]

                if len(self.target_indices) == 1:
                    return train_data, target_data.reshape(-1, 1), weights_data.reshape(-1, 1)

                return train_data, target_data, weights_data.reshape(-1, 1)

            if len(self.target_indices) == 1:
                return train_data, target_data.reshape(-1, 1)

            return train_data, target_data

        return return_data

    def ConvertBatchToTF(self, batch: Any) -> Any:
        """
        Convert a RTensor into a TensorFlow tensor

        Args:
            batch (RTensor): Batch returned from the DataLoader

        Returns:
            tensorflow.Tensor: converted batch
        """
        import tensorflow as tf

        data = batch.GetData()
        batch_size, num_columns = tuple(batch.GetShape())

        data.reshape((batch_size * num_columns,))

        return_data = tf.constant(data, shape=(batch_size, num_columns))

        if batch_size != self.batch_size:
            return_data = tf.pad(return_data, tf.constant([[0, self.batch_size - batch_size], [0, 0]]))

        # Splice target column from the data if weight is given
        if self.target_given:
            train_data = tf.gather(return_data, indices=self.train_indices, axis=1)
            target_data = tf.gather(return_data, indices=self.target_indices, axis=1)

            # Splice weight column from the data if weight is given
            if self.weights_given:
                weights_data = tf.gather(return_data, indices=[self.weights_index], axis=1)

                return train_data, target_data, weights_data

            return train_data, target_data

        return return_data

    # Return a batch when available
    def GetTrainBatch(self) -> Any:
        """Return the next training batch of data from the given RDataFrame

        Returns:
            (np.ndarray): Batch of data of size.
        """

        batch = self.engine.GetTrainBatch()
        return batch if (batch and batch.GetSize() > 0) else None

    def GetValidationBatch(self) -> Any:
        """Return the next training batch of data from the given RDataFrame

        Returns:
            (np.ndarray): Batch of data of size.
        """

        batch = self.engine.GetValidationBatch()
        return batch if (batch and batch.GetSize() > 0) else None


# context managers for the loading thread
class _TrainingEpochContext:
    def __init__(self, internal: _RDataLoader):
        self._internal = internal
        # init loading thread
        internal.Activate()
        internal.CreateTrainBatches()

    def __enter__(self):
        self._internal.ActivateTrainingEpoch()
        return self

    def __exit__(self, type, value, traceback):
        self._internal.DeActivateTrainingEpoch()


class _ValidationEpochContext:
    def __init__(self, internal: _RDataLoader):
        self._internal = internal
        internal.Activate()
        internal.CreateValidationBatches()

    def __enter__(self):
        self._internal.ActivateValidationEpoch()
        return self

    def __exit__(self, type, value, traceback):
        self._internal.DeActivateValidationEpoch()


# formatted iterator (returned by as_torch / as_numpy / as_tensorflow)
class FormattedLoader:
    """
    Iterable that converts each batch to the requested format.
    Returned by the as_torch / as_numpy / as_tensorflow methods on RDataLoader.
    """

    def __init__(
        self,
        internal: _RDataLoader,
        conversion_fn: Callable,
        is_training: bool,
    ):
        self._internal = internal
        self._conversion_fn = conversion_fn
        self._is_training = is_training

    def _make_gen(self):
        ctx_cls = _TrainingEpochContext if self._is_training else _ValidationEpochContext
        get_batch = self._internal.GetTrainBatch if self._is_training else self._internal.GetValidationBatch

        with ctx_cls(self._internal):
            while True:
                batch = get_batch()
                if batch is None:
                    break
                yield self._conversion_fn(batch)

    def __iter__(self):
        return self._make_gen()


class RDataLoader:
    """
    Entry point for ML batch loading from a ROOT RDataFrame.

    Usage without a validation split::

        train = ROOT.Experimental.ML.RDataLoader(df, batch_size=1000, ...)
        for x, y in train.as_torch():
            ...

    Usage with a validation split::

        dl = ROOT.Experimental.ML.RDataLoader(df, batch_size=1000, ...)
        train, val = dl.train_test_split(test_size=0.2)
        for x, y in train.as_torch():
            ...
        for x, y in val.as_numpy():
            ...
    """

    def __init__(
        self,
        rdataframes: ROOT.RDF.RNode | list[ROOT.RDF.RNode],
        batch_size: int = 64,
        batches_in_memory: int = 10,
        columns: list[str] | None = None,
        max_vec_sizes: dict[str, int] | None = None,
        vec_padding: float = 0.0,
        target: str | list[str] | None = None,
        weights: str = "",
        shuffle: bool = True,
        drop_remainder: bool = True,
        set_seed: int = 0,
        load_eager: bool = False,
        sampling_type: str = "",
        sampling_ratio: float = 1.0,
        replacement: bool = False,
    ) -> None:
        """
        Args:
            rdataframes:
                RDataFrame or list of RDataFrames to load from.
            batch_size:
                Number of entries per batch.
            batches_in_memory:
                Approximate number of batches held in the shuffle buffer at any
                time. Larger values improve shuffle quality across cluster
                boundaries at the cost of higher memory usage. Acts as a soft
                cap: the buffer may temporarily exceed this. Defaults to 10.
            columns:
                Names of columns to load. If not given, all columns are used.
            max_vec_sizes:
                Maximum size per vector column. Required for RVec columns.
            vec_padding:
                Padding value for vectors shorter than their max size. Defaults to 0.
            target:
                Name or list of names of target column(s).
            weights:
                Column to use for event weighting. Requires a target.
            shuffle:
                Whether to shuffle data across cluster boundaries every epoch.
                Defaults to True.
            drop_remainder:
                Drop the last batch if smaller than batch_size. Defaults to True.
            set_seed:
                Seed for the random number generator. 0 means a random seed is
                drawn from the system. Defaults to 0.
            load_eager:
                If True, load the full dataset into memory before training.
                If False (default), load lazily in chunks.
            sampling_type:
                Resampling strategy: "undersampling" or "oversampling".
                Requires load_eager=True and exactly two input dataframes.
            sampling_ratio:
                Ratio of minority to majority entries in the resampled dataset.
                Requires load_eager=True and sampling_type set.
            replacement:
                Whether undersampling is with replacement. Requires load_eager=True
                and sampling_type="undersampling".
        """
        # Store all constructor parameters. The C++ backend (_RDataLoader) is
        # created lazily on the first call to as_torch/as_numpy/as_tensorflow or
        # train_test_split.
        self._params = dict(
            rdataframes=rdataframes,
            batch_size=batch_size,
            batches_in_memory=batches_in_memory,
            columns=columns,
            max_vec_sizes=max_vec_sizes,
            vec_padding=vec_padding,
            target=target,
            weights=weights,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
            set_seed=set_seed,
            load_eager=load_eager,
            sampling_type=sampling_type,  # TODO(staider) consider turning into an enum
            sampling_ratio=sampling_ratio,
            replacement=replacement,
        )
        self._internal: _RDataLoader | None = None
        self._test_size: float | None = None
        self._is_training: bool = True  # default: full dataset treated as training

    @classmethod
    def _from_internal(cls, internal: _RDataLoader, is_training: bool) -> RDataLoader:
        """
        Internal factory that creates a split bound RDataLoader sharing an
        already-constructed C++ backend, train_test_split uses this to return two
        RDataLoader instances that both point at the same _RDataLoader object.
        """
        obj = cls.__new__(cls)
        obj._params = None
        obj._internal = internal
        obj._test_size = None
        obj._is_training = is_training
        return obj

    def _ensure_created(self, test_size: float = 0.0) -> None:
        """
        Construct the C++ backend if not already done.
        """
        if self._internal is not None:
            # Already constructed, guard against accidentally calling with a different split
            if self._params is not None and test_size != self._test_size:
                raise RuntimeError(
                    f"RDataLoader was already initialised with test_size="
                    f"{self._test_size}. Create a new RDataLoader to use a different split."
                )
            return

        self._internal = _RDataLoader(**self._params, test_size=test_size)
        self._test_size = test_size

    def train_test_split(self, test_size: float = 0.2) -> Tuple[RDataLoader, RDataLoader]:
        """
        Partition the dataset into training and validation splits.
        Returns two RDataLoader instances that share the same underlying C++
        backend and can each be iterated independently.
        """
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be in (0.0, 1.0), got {test_size}")

        self._ensure_created(test_size)
        return (
            RDataLoader._from_internal(self._internal, is_training=True),
            RDataLoader._from_internal(self._internal, is_training=False),
        )

    def as_numpy(self) -> FormattedLoader:
        """
        Return an iterable that yields batches as NumPy arrays.
        """
        self._ensure_created()
        return FormattedLoader(self._internal, self._internal.ConvertBatchToNumpy, self._is_training)

    def as_torch(self, device: str | torch.device | None = None) -> FormattedLoader:
        """
        Return an iterable that yields batches as PyTorch tensors.

        Args:
            device: If given, the returned tensors are moved to the specified device.
        """
        self._ensure_created()
        conversion_fn = lambda batch: self._internal.ConvertBatchToPyTorch(batch, device)  # noqa: E731
        return FormattedLoader(self._internal, conversion_fn, self._is_training)

    def as_tensorflow(self) -> tf.data.Dataset:
        """
        Return a tf.data.Dataset over batches as TensorFlow tensors.
        """
        import tensorflow as tf

        self._ensure_created()

        batch_size = self._internal.batch_size
        num_train_columns = len(self._internal.train_columns)
        num_target_columns = len(self._internal.target_columns)

        # No target and weights given
        if not self._internal.target_given:
            batch_signature = tf.TensorSpec(shape=(batch_size, num_train_columns), dtype=tf.float32)

        # Target given, no weights given
        elif not self._internal.weights_given:
            batch_signature = (
                tf.TensorSpec(shape=(batch_size, num_train_columns), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, num_target_columns), dtype=tf.float32),
            )

        # Target and weights given
        else:
            batch_signature = (
                tf.TensorSpec(shape=(batch_size, num_train_columns), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, num_target_columns), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32),
            )

        loader = FormattedLoader(self._internal, self._internal.ConvertBatchToTF, self._is_training)
        return tf.data.Dataset.from_generator(lambda: loader, output_signature=batch_signature)

    @property
    def columns(self) -> list[str]:
        """All column names as they appear in each batch tensor."""
        if self._internal is None:
            return self._params["columns"]
        return self._internal.all_columns

    @property
    def train_columns(self) -> list[str]:
        """Feature column names (columns minus target and weights)."""
        if self._internal is None:
            target = self._params["target"] if self._params["target"] is not None else []
            weights = self._params["weights"] if self._params["weights"] is not None else []
            return [col for col in self._params["columns"] if col not in target and col not in weights]
        return self._internal.train_columns

    @property
    def target_columns(self) -> list[str]:
        """Target column names."""
        if self._internal is None:
            return self._params["target"] if self._params["target"] is not None else []
        return self._internal.target_columns

    @property
    def weights_column(self) -> str:
        """Weights column name, or empty string if not set."""
        if self._internal is None:
            return self._params["weights"] if self._params["weights"] is not None else ""
        return self._internal.weights_column

    @property
    def num_batches(self) -> int:
        """Total number of batches in this split for one epoch."""
        if self._internal is None:
            raise RuntimeError(
                "num_batches is available after the first call to "
                "as_torch / as_numpy / as_tensorflow / train_test_split."
            )
        if self._is_training:
            return self._internal.num_training_batches
        return self._internal.num_validation_batches

    @property
    def last_batch_no_of_rows(self) -> int:
        """Number of rows in the last (remainder) batch, 0 if no remainder."""
        if self._internal is None:
            raise RuntimeError(
                "last_batch_no_of_rows is available after the first call to "
                "as_torch / as_numpy / as_tensorflow / train_test_split."
            )
        if self._is_training:
            return self._internal.train_remainder_rows
        return self._internal.val_remainder_rows


def _inject_dataloader_api(parentmodule):
    """
    Inject the public Python API into the ROOT.Experimental.ML namespace.
    Only RDataLoader is part of the public surface.
    """
    for cls in [RDataLoader, FormattedLoader]:
        setattr(parentmodule, cls.__name__, cls)
