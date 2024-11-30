#!/usr/bin/env python3

"""!
@brief  Generate index of tutorials grouped by categories
@author Vít Kučera <vit.kucera@cern.ch>
@date   2024-11-27
"""

import argparse
import glob
import pathlib
import re
import sys


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate index of tutorials grouped by categories")
    parser.add_argument("path", type=str, help="directory to scan")
    args = parser.parse_args()

    # supported file extensions and their Doxygen string patterns
    prefixes_doxygen = {
        ".C": r"/// \\",
        ".py": r"# +\\",
    }
    suffixes_files = prefixes_doxygen.keys()

    # parsed Doxygen fields and their titles
    fields_doxygen = {
        "keywords": "Keywords",
        "classes": "Classes"
    }

    # dictionary with categorised tutorial file paths
    dic_categorised = {
        cat: {} for cat in fields_doxygen
    }

    path_base = args.path
    len_path_base = len(path_base)
    paths_all = []
    for suffix in suffixes_files:
        paths_all += glob.glob(f"{path_base}/**/*{suffix}")
    # print(paths_all)

    # Process files.
    for path in paths_all:
        suffix = pathlib.Path(path).suffix
        if suffix not in suffixes_files:
            continue
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line in file.readlines():
                    line = line.strip()
                    # Look for Doxygen fields.
                    for field in fields_doxygen:
                        if not (match := re.match(rf"{prefixes_doxygen[suffix]}{field} (.*)", line)):
                            continue
                        value = match.group(1)
                        # Split the field value into items.
                        items = [v.strip() for v in value.split(",")]
                        # Add the file path in the matching categories.
                        for item in items:
                            if item not in dic_categorised[field]:
                                dic_categorised[field][item] = []
                            dic_categorised[field][item].append(path[len_path_base:])
        except IOError:
            print(f'Failed to open file "{path}".')
            sys.exit(1)
    # print(dic_categorised)

    # Generate the Markdown index.

    header = "\\defgroup Tutorials Tutorials" \
        "\n\n## Grouped by categories"

    print(header)
    for field, items in dic_categorised.items():
        if not items:
            continue
        print(f"\n### {fields_doxygen[field]}\n")
        for item, list_files in items.items():
            print(f"- {item}")
            print("  - " + ", ".join(list_files))


if __name__ == "__main__":
    main()
