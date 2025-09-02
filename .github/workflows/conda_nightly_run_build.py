import argparse
from contextlib import contextmanager
import os
import re
import shutil
import pathlib
import shlex
import subprocess


@contextmanager
def change_directory(path):
    """Change to 'path' directory and restore it when done."""
    old_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_dir)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-l",
    help="Build from local feedstocks.",
    action="store_true"
)
parser.add_argument(
    "--clean",
    help="Removes feedstock directories.",
    action="store_true"
)
parser.add_argument(
    "-j",
    help="Number of cores to use for building the feedstocks.",
    type=int
)

FEEDSTOCK_DIRNAMES = ["llvmdev-feedstock",
                      "clangdev-feedstock", "root-feedstock"]


def clean_directories():
    for feedstock_dir in FEEDSTOCK_DIRNAMES:
        if os.path.exists(feedstock_dir):
            print(f"Cleaning directory {feedstock_dir}")
            shutil.rmtree(feedstock_dir)


def patch_conda_build_config():
    """
    Patch the `conda_build_config.yaml` file of this feedstock with a root_master variant.

    Assumes we are already in the right directory.
    """
    config_file = pathlib.Path("recipe/conda_build_config.yaml")
    config_text = config_file.read_text()
    match = re.search(r"(root_6\d\d\d\d)", config_text)
    assert match is not None
    config_text = config_text.replace(match.group(0), "root_master")
    with open(config_file, "w") as f:
        f.write(config_text)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.clean:
        clean_directories()
        print("Directories cleaned, exiting now.")
        raise SystemExit()

    # Copy repositories to local directory if none is present
    if not any(os.path.exists(p) for p in FEEDSTOCK_DIRNAMES):
        if args.l:
            feedstock_paths = [pathlib.Path(f"../{name}")
                               for name in FEEDSTOCK_DIRNAMES]
            assert all(os.path.exists(p)
                       for p in feedstock_paths), "A local feedstock was not found"

            for p in feedstock_paths:
                shutil.copytree(p, p.name)
        else:
            for p in FEEDSTOCK_DIRNAMES:
                subprocess.run(shlex.split(
                    f"git clone --single-branch --branch nightlies https://github.com/root-project/{p}.git"
                ))

    with change_directory(FEEDSTOCK_DIRNAMES[0]):
        print("Starting building LLVM feedstock")
        env = os.environ
        env.update(
            {"CONDA_FORGE_DOCKER_RUN_ARGS": "--network host --security-opt label=disable --privileged --userns=keep-id"})
        subprocess.run(shlex.split(
            "./build-locally.py linux_64_variantroot_master"), env=env, check=True)
        print("Finished building LLVM feedstock")

    with change_directory(FEEDSTOCK_DIRNAMES[1]):
        print("Starting building clang feedstock")
        env = os.environ
        env.update(
            {"CONDA_FORGE_DOCKER_RUN_ARGS": "--network host --security-opt label=disable --privileged --userns=keep-id"})
        subprocess.run(shlex.split(
            "./build-locally.py linux_64_variantroot_master"), env=env)
        print("Finished building clang feedstock")

    subprocess.run(shlex.split(
        f"cp -r {FEEDSTOCK_DIRNAMES[0]}/build_artifacts {FEEDSTOCK_DIRNAMES[2]}"))
    subprocess.run(shlex.split(
        f"cp -r {FEEDSTOCK_DIRNAMES[1]}/build_artifacts {FEEDSTOCK_DIRNAMES[2]}"))

    with change_directory(FEEDSTOCK_DIRNAMES[2]):
        print("Starting building ROOT feedstock")
        env = os.environ
        env.update(
            {"CONDA_FORGE_DOCKER_RUN_ARGS": "--network host --security-opt label=disable --privileged --userns=keep-id"})
        subprocess.run(shlex.split(
            "./build-locally.py linux_64_python3.10.____cpython"), env=env)
        print("Finished building ROOT feedstock")
