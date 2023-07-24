from __future__ import annotations

from typing import Any, Callable, Tuple, TYPE_CHECKING
import atexit

if TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf
    import torch


class BaseGenerator:
    def get_template(
        self,
        tree_name: str,
        file_name: str,
        columns: list[str] = list(),
        max_vec_sizes: dict[str, int] = dict(),
    ) -> Tuple[str, list[int]]:
        """
        Generate a template for the RBatchGenerator based on the given
        RDataFrame and columns.

        Args:
            file_name (str): name of the root file.
            tree_name (str): name of the tree in the root file.
            columns (list[str]): Columns that should be loaded.
                                 Defaults to loading all columns
                                 in the given RDataFrame
            max_vec_sizes (list[int]): The length of each vector based column.

        Returns:
            template (str): Template for the RBatchGenerator
        """

        # from cppyy.gbl.ROOT import RDataFrame
        from ROOT import RDataFrame

        x_rdf = RDataFrame(tree_name, file_name)

        if not columns:
            columns = x_rdf.GetColumnNames()

        template_dict = {
            "Int_t": "int&",
            "Float_t": "float&",
            "Bool_t": "bool&",
            "ROOT::VecOps::RVec<int>": "ROOT::RVec<int>",
            "ROOT::VecOps::RVec<float>": "ROOT::RVec<float>",
            "ROOT::VecOps::RVec<bool>": "ROOT::RVec<bool>",
        }

        template_string = ""

        self.given_columns = []
        self.all_columns = []
        # Get the types of the different columns

        max_vec_sizes_list = []

        for name in columns:
            name_str = str(name)
            self.given_columns.append(name_str)
            column_type = template_dict[str(x_rdf.GetColumnType(name_str))]
            template_string += column_type + ","

            if column_type in [
                "ROOT::RVec<int>",
                "ROOT::RVec<float>",
                "ROOT::RVec<bool>",
            ]:
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
        tree_name: str,
        file_name: str,
        batch_size: int,
        chunk_size: int,
        columns: list[str] = list(),
        filters: list[str] = list(),
        max_vec_sizes: dict[str, int] = dict(),
        vec_padding: int = 0,
        target: str = "",
        weights: str = "",
        validation_split: float = 0.0,
        max_chunks: int = 0,
        shuffle: bool = True,
    ):
        """Wrapper around the Cpp RBatchGenerator

        Args:
            tree_name (str): Name of the tree in the ROOT file
            file_name (str): Path to the ROOT file
            batch_size (int): Size of the returned chunks.
            chunk_size (int):
                The size of the chunks loaded from the ROOT file. Higher chunk
                size results in better randomization, but higher memory usage.
            columns (list[str], optional):
                Columns to be returned. If not given, all columns are used.
            filters (list[str], optional):
                Filters to apply during loading. If not given, no filters
                are applied.
            max_vec_sizes (dict[std, int], optional):
                Size of each column that consists of vectors.
                Required when using vector based columns.
            vec_padding (int):
                Value to pad vectors with if the vector is smaller
                than the given max vector length. Defaults is 0
            target (str, optional): Column that is used as target.
            weights (str, optional):
                Column used to weight events.
                Can only be used when a target is given.
            validation_split (float, optional):
                The ratio of batches being kept for validation.
                Value has to be between 0 and 1. Defaults to 0.0.
            max_chunks (int, optional):
                The number of chunks that should be loaded for an epoch.
                If not given, the whole file is used.
            shuffle (bool):
                Batches consist of random events and are shuffled every epoch.
                Defaults to True.
        """

        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "Failed to import NumPy during init. NumPy is required when \
                    using RBatchGenerator"
            )

        if chunk_size < batch_size:
            raise ValueError(
                f"chunk_size cannot be smaller than batch_size: chunk_size: \
                    {chunk_size}, batch_size: {batch_size}"
            )

        if validation_split < 0.0 or validation_split > 1.0:
            raise ValueError(
                f"The validation_split has to be in range [0.0, 1.0] \n \
                    given value is {validation_split}"
            )

        # TODO: better linking when importing into ROOT
        # ROOT.gInterpreter.ProcessLine(
        #     f'#include "{main_folder}Cpp_files/RBatchGenerator.cpp"')

        self.target_column = target
        self.weights_column = weights

        template, max_vec_sizes_list = self.get_template(
            tree_name, file_name, columns, max_vec_sizes
        )

        self.num_columns = len(self.all_columns)
        self.batch_size = batch_size

        # Handle target
        self.target_given = len(self.target_column) > 0
        if self.target_given:
            if target in self.all_columns:
                self.target_index = self.all_columns.index(self.target_column)
            else:
                raise ValueError(
                    f"Provided target not in given columns: \ntarget => \
                        {target}\ncolumns => {self.all_columns}"
                )

        # Handle weights
        self.weights_given = len(self.weights_column) > 0
        if self.weights_given and not self.target_given:
            raise ValueError("Weights can only be used when a target is provided")
        if self.weights_given:
            if weights in self.all_columns:
                self.weights_index = self.all_columns.index(self.weights_column)
            else:
                raise ValueError(
                    f"Provided weights not in given columns: \nweights => \
                        {weights}\ncolumns => {self.all_columns}"
                )

        self.train_columns = [c for c in self.all_columns if c not in [target, weights]]

        from ROOT import TMVA, EnableThreadSafety

        # The RBatchGenerator will create a separate C++ thread for I/O.
        # Enable thread safety in ROOT from here, to make sure there is no
        # interference between the main Python thread (which might call into
        # cling via cppyy) and the I/O thread.
        EnableThreadSafety()

        expanded_filter = " && ".join(["(" + fltr + ")" for fltr in filters])

        self.generator = TMVA.Experimental.Internal.RBatchGenerator(template)(
            tree_name,
            file_name,
            chunk_size,
            batch_size,
            self.given_columns,
            expanded_filter,
            max_vec_sizes_list,
            vec_padding,
            validation_split,
            max_chunks,
            self.num_columns,
            shuffle,
        )

        atexit.register(self.DeActivate)

    def StartValidation(self):
        self.generator.StartValidation()

    @property
    def is_active(self):
        return self.generator.IsActive()

    def Activate(self):
        """Initialize the generator to be used for a loop"""
        self.generator.Activate()

    def DeActivate(self):
        """Initialize the generator to be used for a loop"""
        self.generator.DeActivate()

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
            raise ImportError("Failed to import numpy in batchgenerator init")

        # Split the target and weight
        if not self.target_given:
            return np.zeros((self.batch_size, self.num_columns))

        if not self.weights_given:
            return np.zeros((self.batch_size, self.num_columns - 1)), np.zeros(
                (self.batch_size)
            )

        return (
            np.zeros((self.batch_size, self.num_columns - 2)),
            np.zeros((self.batch_size)),
            np.zeros((self.batch_size)),
        )

    def ConvertBatchToNumpy(self, batch: "RTensor") -> np.ndarray:
        """Convert a RTensor into a NumPy array

        Args:
            batch (RTensor): Batch returned from the RBatchGenerator

        Returns:
            np.ndarray: converted batch
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("Failed to import numpy in batchgenerator init")

        data = batch.GetData()
        data.reshape((self.batch_size * self.num_columns,))

        return_data = np.array(data).reshape(self.batch_size, self.num_columns)

        # Splice target column from the data if weight is given
        if self.target_given:
            target_data = return_data[:, self.target_index]
            return_data = np.column_stack(
                (
                    return_data[:, : self.target_index],
                    return_data[:, self.target_index + 1 :],
                )
            )

            # Splice weights column from the data if weight is given
            if self.weights_given:
                if self.target_index < self.weights_index:
                    self.weights_index -= 1

                weights_data = return_data[:, self.weights_index]
                return_data = np.column_stack(
                    (
                        return_data[:, : self.weights_index],
                        return_data[:, self.weights_index + 1 :],
                    )
                )
                return return_data, target_data, weights_data

            return return_data, target_data

        return return_data

    def ConvertBatchToPyTorch(self, batch: Any) -> torch.Tensor:
        """Convert a RTensor into a PyTorch tensor

        Args:
            batch (RTensor): Batch returned from the RBatchGenerator

        Returns:
            torch.Tensor: converted batch
        """
        import torch

        data = batch.GetData()
        data.reshape((self.batch_size * self.num_columns,))

        return_data = torch.Tensor(data).reshape(self.batch_size, self.num_columns)

        # Splice target column from the data if weight is given
        if self.target_given:
            target_data = return_data[:, self.target_index]
            return_data = torch.column_stack(
                (
                    return_data[:, : self.target_index],
                    return_data[:, self.target_index + 1 :],
                )
            )

            # Splice weights column from the data if weight is given
            if self.weights_given:
                if self.target_index < self.weights_index:
                    self.weights_index -= 1

                weights_data = return_data[:, self.weights_index]
                return_data = torch.column_stack(
                    (
                        return_data[:, : self.weights_index],
                        return_data[:, self.weights_index + 1 :],
                    )
                )
                return return_data, target_data, weights_data

            return return_data, target_data

        return return_data

    def ConvertBatchToTF(self, batch: Any) -> np.ndarray:
        """
        PLACEHOLDER: at this moment this function only calls the
        ConvertBatchToNumpy function. In the Future this function can be
        used to convert to TF tensors directly

        Args:
            batch (RTensor): Batch returned from the RBatchGenerator

        Returns:
            np.ndarray: converted batch
        """
        # import tensorflow as tf

        batch = self.ConvertBatchToNumpy(batch)

        # TODO: improve this by returning tensorflow tensors
        return batch

    # Return a batch when available
    def GetTrainBatch(self) -> Any:
        """Return the next training batch of data from the given RDataFrame

        Returns:
            (np.ndarray): Batch of data of size.
        """

        batch = self.generator.GetTrainBatch()

        if batch and batch.GetSize() > 0:
            return batch

        return None

    def GetValidationBatch(self) -> Any:
        """Return the next training batch of data from the given RDataFrame

        Returns:
            (np.ndarray): Batch of data of size.
        """

        batch = self.generator.GetValidationBatch()

        if batch and batch.GetSize() > 0:
            return batch

        return None


# Context that activates and deactivates the loading thread of the Cpp class
# This ensures that the thread will always be deleted properly
class LoadingThreadContext:
    def __init__(self, base_generator: BaseGenerator):
        self.base_generator = base_generator

    def __enter__(self):
        self.base_generator.Activate()

    def __exit__(self, type, value, traceback):
        self.base_generator.DeActivate()
        return True


class TrainRBatchGenerator:
    def __init__(self, base_generator: BaseGenerator, conversion_function: Callable):
        """
        A generator that returns the training batches of the given
        base generator

        Args:
            base_generator (BaseGenerator):
                The base connection to the Cpp code
            conversion_function (Callable[RTensor, np.NDArray|torch.Tensor]):
                Function that converts a given RTensor into a python batch
        """
        self.base_generator = base_generator
        self.conversion_function = conversion_function

    def Activate(self):
        """Start the loading of training batches"""
        self.base_generator.Activate()

    def DeActivate(self):
        """Stop the loading of batches"""

        self.base_generator.DeActivate()

    @property
    def columns(self) -> list[str]:
        return self.base_generator.all_columns

    @property
    def train_columns(self) -> list[str]:
        return self.base_generator.train_columns

    @property
    def target_column(self) -> str:
        return self.base_generator.target_column

    @property
    def weights_column(self) -> str:
        return self.base_generator.weights_column

    def __iter__(self):
        self._callable = self.__call__()

        return self

    def __next__(self):
        batch = self._callable.__next__()

        if batch is None:
            raise StopIteration

        return batch

    def __call__(self) -> Any:
        """Start the loading of batches and Yield the results

        Yields:
            Union[np.NDArray, torch.Tensor]: A batch of data
        """

        with LoadingThreadContext(self.base_generator):
            while True:
                batch = self.base_generator.GetTrainBatch()

                if not batch:
                    break

                yield self.conversion_function(batch)

        return None


class ValidationRBatchGenerator:
    def __init__(self, base_generator: BaseGenerator, conversion_function: Callable):
        """
        A generator that returns the validation batches of the given base
        generator. NOTE: The ValidationRBatchGenerator only returns batches
        if the training has been run.

        Args:
            base_generator (BaseGenerator):
                The base connection to the Cpp code
            conversion_function (Callable[RTensor, np.NDArray|torch.Tensor]):
                Function that converts a given RTensor into a python batch
        """
        self.base_generator = base_generator
        self.conversion_function = conversion_function

    @property
    def columns(self) -> list[str]:
        return self.base_generator.all_columns

    @property
    def train_columns(self) -> list[str]:
        return self.base_generator.train_columns

    @property
    def target_column(self) -> str:
        return self.base_generator.target_column

    @property
    def weights_column(self) -> str:
        return self.base_generator.weights_column

    def __iter__(self):
        self._callable = self.__call__()

        return self

    def __next__(self):
        batch = self._callable.__next__()

        if batch is None:
            raise StopIteration

        return batch

    def __call__(self) -> Any:
        """Loop through the validation batches

        Yields:
            Union[np.NDArray, torch.Tensor]: A batch of data
        """
        if self.base_generator.is_active:
            self.base_generator.DeActivate()

        self.base_generator.StartValidation()

        while True:
            batch = self.base_generator.GetValidationBatch()

            if not batch:
                break

            yield self.conversion_function(batch)


def CreateNumPyGenerators(
    tree_name: str,
    file_name: str,
    batch_size: int,
    chunk_size: int,
    columns: list[str] = list(),
    filters: list[str] = list(),
    max_vec_sizes: dict[str, int] = dict(),
    vec_padding: int = 0,
    target: str = "",
    weights: str = "",
    validation_split: float = 0.0,
    max_chunks: int = 0,
    shuffle: bool = True,
) -> Tuple[TrainRBatchGenerator, ValidationRBatchGenerator]:
    """
    Return two batch generators based on the given ROOT file and tree.
    The first generator returns training batches, while the second generator
    returns validation batches

    Args:
        tree_name (str): Name of the tree in the ROOT file
        file_name (str): Path to the ROOT file
        batch_size (int): Size of the returned chunks.
        chunk_size (int):
            The size of the chunks loaded from the ROOT file. Higher chunk size
            results in better randomization, but also higher memory usage.
        columns (list[str], optional):
            Columns to be returned. If not given, all columns are used.
        filters (list[str], optional):
            Filters to apply. If not given, no filters are applied.
        max_vec_sizes (list[int], optional):
            Size of each column that consists of vectors.
            Required when using vector based columns
        target (str, optional):
            Column that is used as target.
        weights (str, optional):
            Column used to weight events.
            Can only be used when a target is given
        validation_split (float, optional):
            The ratio of batches being kept for validation.
            Value has to be from 0.0 to 1.0. Defaults to 0.0.
        max_chunks (int, optional):
            The number of chunks that should be loaded for an epoch.
            If not given, the whole file is used
        shuffle (bool):
            randomize the training batches every epoch. Defaults to True

    Returns:
        Tuple[TrainRBatchGenerator, ValidationRBatchGenerator]:
            Two generators are returned. One used to load training batches,
            and one to load validation batches. NOTE: the validation batches
            are loaded during the training. Before training, the validation
            generator will return no batches.
    """
    base_generator = BaseGenerator(
        tree_name,
        file_name,
        batch_size,
        chunk_size,
        columns,
        filters,
        max_vec_sizes,
        vec_padding,
        target,
        weights,
        validation_split,
        max_chunks,
        shuffle,
    )

    train_generator = TrainRBatchGenerator(
        base_generator, base_generator.ConvertBatchToNumpy
    )
    validation_generator = ValidationRBatchGenerator(
        base_generator, base_generator.ConvertBatchToNumpy
    )

    return train_generator, validation_generator


def CreateTFDatasets(
    tree_name: str,
    file_name: str,
    batch_size: int,
    chunk_size: int,
    columns: list[str] = list(),
    filters: list[str] = list(),
    max_vec_sizes: dict[str, int] = dict(),
    vec_padding: int = 0,
    target: str = "",
    weights: str = "",
    validation_split: float = 0.0,
    max_chunks: int = 0,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Return two Tensorflow Datasets based on the given ROOT file and tree
    The first generator returns training batches, while the second generator
    returns validation batches

    Args:
        tree_name (str): Name of the tree in the ROOT file
        file_name (str): Path to the ROOT file
        batch_size (int): Size of the returned chunks.
        chunk_size (int):
            The size of the chunks loaded from the ROOT file. Higher chunk size
            results in better randomization, but also higher memory usage.
        columns (list[str], optional):
            Columns to be returned. If not given, all columns are used.
        filters (list[str], optional):
            Filters to apply. If not given, no filters are applied.
        max_vec_sizes (list[int], optional):
            Size of each column that consists of vectors.
            Required when using vector based columns
        target (str, optional):
            Column that is used as target.
        weights (str, optional):
            Column used to weight events.
            Can only be used when a target is given
        validation_split (float, optional):
            The ratio of batches being kept for validation.
            Value has to be from 0.0 to 1.0. Defaults to 0.0.
        max_chunks (int, optional):
            The number of chunks that should be loaded for an epoch.
            If not given, the whole file is used
        shuffle (bool):
            randomize the training batches every epoch. Defaults to True

    Returns:
        Tuple[TrainRBatchGenerator, ValidationRBatchGenerator]:
            Two generators are returned. One used to load training batches,
            and one to load validation batches. NOTE: the validation batches
            are loaded during the training. Before training, the validation
            generator will return no batches.
    """
    import tensorflow as tf

    base_generator = BaseGenerator(
        tree_name,
        file_name,
        batch_size,
        chunk_size,
        columns,
        filters,
        max_vec_sizes,
        vec_padding,
        target,
        weights,
        validation_split,
        max_chunks,
        shuffle,
    )

    train_generator = TrainRBatchGenerator(
        base_generator, base_generator.ConvertBatchToTF
    )
    validation_generator = ValidationRBatchGenerator(
        base_generator, base_generator.ConvertBatchToTF
    )

    num_columns = len(train_generator.train_columns)

    # No target and weights given
    if target == "":
        batch_signature = tf.TensorSpec(
            shape=(batch_size, num_columns), dtype=tf.float32
        )

    # Target given, no weights given
    elif weights == "":
        batch_signature = (
            tf.TensorSpec(shape=(batch_size, num_columns), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
        )

    # Target and weights given
    else:
        batch_signature = (
            tf.TensorSpec(shape=(batch_size, num_columns), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
        )

    ds_train = tf.data.Dataset.from_generator(
        train_generator, output_signature=batch_signature
    )

    # Give access to the columns function of the training set
    setattr(ds_train, "columns", train_generator.columns)
    setattr(ds_train, "train_columns", train_generator.train_columns)
    setattr(ds_train, "target_column", train_generator.target_column)
    setattr(ds_train, "weights_column", train_generator.weights_column)

    ds_validation = tf.data.Dataset.from_generator(
        validation_generator, output_signature=batch_signature
    )

    # Give access to the columns function of the validation set
    setattr(ds_validation, "columns", train_generator.columns)
    setattr(ds_validation, "train_columns", train_generator.train_columns)
    setattr(ds_validation, "target_column", train_generator.target_column)
    setattr(ds_validation, "weights_column", train_generator.weights_column)

    return ds_train, ds_validation


def CreatePyTorchGenerators(
    tree_name: str,
    file_name: str,
    batch_size: int,
    chunk_size: int,
    columns: list[str] = list(),
    filters: list[str] = list(),
    max_vec_sizes: dict[str, int] = dict(),
    vec_padding: int = 0,
    target: str = "",
    weights: str = "",
    validation_split: float = 0.0,
    max_chunks: int = 0,
    shuffle: bool = True,
) -> Tuple[TrainRBatchGenerator, ValidationRBatchGenerator]:
    """
    Return two Tensorflow Datasets based on the given ROOT file and tree
    The first generator returns training batches, while the second generator
    returns validation batches

    Args:
        tree_name (str): Name of the tree in the ROOT file
        file_name (str): Path to the ROOT file
        batch_size (int): Size of the returned chunks.
        chunk_size (int):
            The size of the chunks loaded from the ROOT file. Higher chunk size
            results in better randomization, but also higher memory usage.
        columns (list[str], optional):
            Columns to be returned. If not given, all columns are used.
        filters (list[str], optional):
            Filters to apply. If not given, no filters are applied.
        max_vec_sizes (list[int], optional):
            Size of each column that consists of vectors.
            Required when using vector based columns
        target (str, optional):
            Column that is used as target.
        weights (str, optional):
            Column used to weight events.
            Can only be used when a target is given
        validation_split (float, optional):
            The ratio of batches being kept for validation.
            Value has to be from 0.0 to 1.0. Defaults to 0.0.
        max_chunks (int, optional):
            The number of chunks that should be loaded for an epoch.
            If not given, the whole file is used
        shuffle (bool):
            randomize the training batches every epoch. Defaults to True

    Returns:
        Tuple[TrainRBatchGenerator, ValidationRBatchGenerator]:
            Two generators are returned. One used to load training batches,
            and one to load validation batches. NOTE: the validation batches
            are loaded during the training. Before training, the validation
            generator will return no batches.
    """
    base_generator = BaseGenerator(
        tree_name,
        file_name,
        batch_size,
        chunk_size,
        columns,
        filters,
        max_vec_sizes,
        vec_padding,
        target,
        weights,
        validation_split,
        max_chunks,
        shuffle,
    )

    train_generator = TrainRBatchGenerator(
        base_generator, base_generator.ConvertBatchToPyTorch
    )
    validation_generator = ValidationRBatchGenerator(
        base_generator, base_generator.ConvertBatchToPyTorch
    )

    return train_generator, validation_generator
