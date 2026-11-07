#!/usr/bin/env python3

# pylint: disable=missing-function-docstring,line-too-long

"""Check that building an already built tree does not rebuild anything.

Generator agnostic: rather than parsing build tool output, it fingerprints every
file in the build tree, builds again, and reports what was written. ROOT's one
accepted exception is etc/gitinfo.txt, which the gitinfotxt target in the top
level CMakeLists.txt rewrites on every build by design.

Standalone usage:

    python3 null_build_check.py <builddir> [<extra cmake --build args>...]
"""

import argparse
import collections
import fnmatch
import os
import subprocess
import sys

# Globs matched against the '/'-separated path relative to the build directory.
# fnmatch's '*' matches '/' too, so unanchored patterns match at any depth.
ALLOWED_PATTERNS = (
    "etc/gitinfo.txt",
    # Build tool logs and dependency databases, not build products.
    ".ninja_log",
    ".ninja_deps",
    ".ninja_lock",
    ".cmake/api/*",
    # Refreshed by every make.
    "*/compiler_depend.*",
    "*/depend.internal",
    "*/depend.make",
    "CMakeFiles/progress.marks",
    # MSBuild file trackers and up-to-date markers.
    "*.tlog",
    "*.tlog/*",
    "*.lastbuildstate",
    "*.unsuccessfulbuild",
    "*.CopyComplete",
    # MSBuild writes the link recipe of every project on every build, whether
    # or not the project is relinked.
    "*.recipe",
    # The ZERO_CHECK project of the Visual Studio generator rewrites the CMake
    # generation stamps on every build to record that CMake need not re-run. A
    # CMake run that really did regenerate would still be caught, since it
    # rewrites the project files as well.
    "CMakeFiles/generate.stamp",
    "*/CMakeFiles/generate.stamp",
    # ctest leftovers, in case the tree has been tested before.
    "Testing/*",
)

# ExternalProject keeps git clones of externals in the build tree.
PRUNED_DIRS = (".git",)

MAX_REPORTED = 100


def snapshot(builddir: str) -> dict:
    fingerprints = {}

    for dirpath, dirnames, filenames in os.walk(builddir):
        dirnames[:] = [name for name in dirnames if name not in PRUNED_DIRS]

        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                stat = os.lstat(path)
            except OSError:
                continue  # vanished under us, or a dangling symlink
            key = os.path.relpath(path, builddir).replace(os.sep, "/")
            fingerprints[key] = (stat.st_mtime_ns, stat.st_size)

    return fingerprints


def is_allowed(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in ALLOWED_PATTERNS)


def compare(before: dict, after: dict) -> list:
    touched = []

    for path, fingerprint in after.items():
        if path not in before:
            touched.append((path, "created"))
        elif before[path] != fingerprint:
            touched.append((path, "modified"))

    for path in before:
        if path not in after:
            touched.append((path, "deleted"))

    return sorted(entry for entry in touched if not is_allowed(entry[0]))


def ninja_explain(builddir: str) -> str:
    if not os.path.exists(os.path.join(builddir, "build.ninja")):
        return ""

    try:
        result = subprocess.run(
            ["ninja", "-C", builddir, "-n", "-d", "explain"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError:
        return ""

    return result.stderr


def check_null_build(builddir: str, rebuild) -> list:
    """`rebuild` runs `cmake --build builddir` and returns its exit code."""

    print(f"Recording the state of {builddir}")
    before = snapshot(builddir)
    print(f"{len(before)} files")

    returncode = rebuild()
    if returncode != 0:
        raise RuntimeError(f"rebuilding an already built tree failed with exit code {returncode}")

    return compare(before, snapshot(builddir))


def summarize(touched: list) -> list:
    """Count the reported paths per file type, most numerous first."""

    counts = collections.Counter()

    for path, _ in touched:
        name = path.rsplit("/", 1)[-1]
        stem, dot, extension = name.rpartition(".")
        counts["*." + extension if stem and dot else name] += 1

    return counts.most_common()


def report(builddir: str, touched: list) -> None:
    if not touched:
        print("No spurious rebuilds: building again left the build tree untouched.")
        return

    print(f"{len(touched)} file(s) were written by a build that had nothing to do:")

    for path, what in touched[:MAX_REPORTED]:
        print(f"  {what:<8}  {path}")

    if len(touched) > MAX_REPORTED:
        print(f"  ... and {len(touched) - MAX_REPORTED} more")
        # The listing is truncated and sorted by path, so a category that only
        # shows up late would go unnoticed without this.
        print("\nBy file type:")
        for kind, count in summarize(touched):
            print(f"  {count:>6}  {kind}")

    explanation = ninja_explain(builddir)
    if explanation:
        print("\nWhy ninja thinks there is work left to do:")
        print(explanation)

    print("""
Usual causes are a custom command that does not declare its OUTPUT or BYPRODUCTS,
an output that ends up newer than what it is compared against, or a generated
file embedding a timestamp. If a file really has to be rewritten on every build,
add it to ALLOWED_PATTERNS above.""")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("builddir", help="a ROOT build directory that is fully built")
    parser.add_argument("build_args", nargs="*", help="extra arguments for `cmake --build`, e.g. --config Release")
    args = parser.parse_args()

    builddir = os.path.abspath(args.builddir)
    command = ["cmake", "--build", builddir, "--parallel", str(os.cpu_count())] + args.build_args

    def rebuild() -> int:
        print("+ " + " ".join(command))
        return subprocess.run(command, check=False).returncode

    touched = check_null_build(builddir, rebuild)
    report(builddir, touched)

    return 1 if touched else 0


if __name__ == "__main__":
    sys.exit(main())
