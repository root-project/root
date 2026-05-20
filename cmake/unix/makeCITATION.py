# Author: axel@cern.ch, 2021-07-08
# License: LGPL2.1+, see file `LICENSE` in this folder.

import pypandoc
import argparse
import json
import datetime
import os, sys

import ROOT  # for ROOT.__version__

"""
Generate a CITATION.cff-style YAML block for the ROOT project based on a Markdown input file.

This script reads a Markdown file (e.g. `makeCITATION_example_input.md`, but
typically the ROOT release notes) containing a bullet list of authors and
optional ORCIDs, converts it to a Pandoc JSON AST using pypandoc, extracts
author information.
"""


def printNames(nameBlock):
    # print(json.dumps(nameBlock))
    nameParts = [c for c in nameBlock]
    firstLastNames = [n["c"].replace("\u00a0", " ") for n in nameParts if n["t"] == "Str"]
    firstName = firstLastNames[0]
    lastName = firstLastNames[1]
    if lastName[-1] == ",":
        lastName = lastName[0:-1]
    print("  - family-names:", lastName)
    print("    given-names:", firstName)


parser = argparse.ArgumentParser()
parser.add_argument("releasenotes")
parser.add_argument("--dump-ast", action="store_true")
args = parser.parse_args()

jsonRN = json.loads(pypandoc.convert_file(args.releasenotes, "json"))
if args.dump_ast:
    print(json.dumps(jsonRN, indent=1))
    sys.exit(0)


print(
    """cff-version: 1.1.0
message: "If you use ROOT, please cite it as below."
authors:"""
)

for block in jsonRN["blocks"]:
    if "t" in block and block["t"] == "BulletList":
        for item in block["c"]:
            itemC = item[0]["c"]
            if itemC[0]["t"] == "Link":
                # 0th element is ["", [], []] ?
                printNames(itemC[0]["c"][1])
                orcid = itemC[0]["c"][2][0]
                print("    orcid:", orcid)
            elif itemC[0]["t"] == "Str":
                printNames(itemC)
            else:
                print('ERROR: expected "Link" or "Str" in', json.dumps(itemC[0], indent=1))
                sys.exit(1)
        break

print(
    f"""title: "ROOT: analyzing, storing and visualizing big data, scientifically"
version: {ROOT.__version__}
doi: 10.5281/zenodo.848818
date-released: {datetime.datetime.now().strftime("%Y-%m-%d")}
"""
)
