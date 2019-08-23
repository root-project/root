import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import bench

DATA_FOLDER = "./data/"

print("Generating data!")
bench.create_model_gaussian(
    num_samples=100000,
    num_features=5,
    num_trees=1000,
    max_depth=10,
    data_folder=DATA_FOLDER,
)
print("Data generated!")

# end
