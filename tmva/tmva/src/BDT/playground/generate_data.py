import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import bench

DATA_FOLDER = "./data/"

bench.create_model_gaussian(
    num_samples=1000,
    num_features=5,
    num_trees=100,
    max_depth=3,
    data_folder=DATA_FOLDER,
)


# end
