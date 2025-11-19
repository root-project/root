import difflib
import json
import os
import shutil
import sys
import tempfile


# Replace the criterion according to which a line shall be skipped
def should_keep_line(line):
    # Skip the banner and empty lines
    skip_patterns = [
        "Info in <TUnixSystem::ACLiC",
        "Info in <TMacOSXSystem::ACLiC",
        "FAILED TO establish the default connection to the WindowServer",
        '"version": ',
        '"pygments_lexer": "ipython',
        '     "execution_count":',
        "libclang_rt.asan-",
    ]
    for pattern in skip_patterns:
        if pattern in line:
            return False
    return True


def removeCellMetadata(lines):
    filteredLines = []
    discardLine = False
    for line in lines:
        if '   "metadata": {' in line:
            if line.endswith("}," + os.linesep):  # empty metadata
                continue
            discardLine = True

        if not discardLine:
            filteredLines.append(line)

        if discardLine and "   }," in line:  # end of metadata
            discardLine = False

    return filteredLines


def getFilteredLines(fileName):
    with open(fileName) as f:
        filteredLines = list(filter(should_keep_line, f.readlines()))

    # Sometimes the jupyter server adds a new line at the end of the notebook
    # and nbconvert does not.
    lastLine = filteredLines[-1]
    if lastLine[-1] != "\n":
        filteredLines[-1] += "\n"

    # Remove the metadata field of cells (contains specific execution timestamps)
    filteredLines = removeCellMetadata(filteredLines)

    return filteredLines


# Workaround to support nbconvert versions >= 7.14 . See #14303
def patchForNBConvert714(outNBLines):
    newOutNBLines = []
    toReplace = """      "1\\n"\n"""
    replacement = [
        """      "1"\n""",
        """     ]\n""",
        """    },\n""",
        """    {\n""",
        """     "name": "stdout",\n""",
        """     "output_type": "stream",\n""",
        """     "text": [\n""",
        """      "\\n"\n""",
    ]

    for line in outNBLines:
        if line == toReplace:
            newOutNBLines.extend(replacement)
        else:
            newOutNBLines.append(line)
    return newOutNBLines


def compareNotebooks(inNBName, outNBName):
    inNBLines = getFilteredLines(inNBName)
    inNBLines = patchForNBConvert714(inNBLines)
    outNBLines = getFilteredLines(outNBName)
    outNBLines = patchForNBConvert714(outNBLines)
    areDifferent = False
    for line in difflib.unified_diff(inNBLines, outNBLines, fromfile=inNBName, tofile=outNBName):
        areDifferent = True
        sys.stdout.write(line)
    if areDifferent:
        print("\n")
    return areDifferent


def createKernelSpec():
    """Create a root kernel spec with the right python interpreter name
    and puts it in a tmp directory. Return the name of such directory."""
    tmpd = tempfile.mkdtemp(suffix="_nbdiff_ipythondir")
    kernelsPath = os.path.join(tmpd, "kernels")
    os.mkdir(kernelsPath)
    rootKernelPath = os.path.join(kernelsPath, "root")
    os.mkdir(rootKernelPath)
    with open(os.path.join(rootKernelPath, "kernel.json"), "w") as kernel_file:
        kernel_file.write(
            """{
         "language": "c++",
         "display_name": "ROOT C++",
         "argv": [
          "%s",
          "-m",
          "JupyROOT.kernel.rootkernel",
          "-f",
          "{connection_file}"
         ]
        }
        """
            % sys.executable
        )

    return tmpd


def addEtcToEnvironment(inNBDirName):
    """Add the etc directory of root to the environment under the name of
    JUPYTER_PATH in order to pick up the kernel specs.
    """
    ipythondir = createKernelSpec()
    os.environ["JUPYTER_PATH"] = ipythondir
    os.environ["IPYTHONDIR"] = ipythondir
    return ipythondir


def getKernelName(inNBName):
    with open(inNBName) as f:
        nbj = json.load(f)
    return nbj["metadata"]["kernelspec"]["name"]


def canReproduceNotebook(inNBName, needsCompare):
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    tmpDir = addEtcToEnvironment(os.path.dirname(inNBName))
    outNBName = inNBName.replace(".ipynb", "_out.ipynb")

    # Load input notebook
    with open(inNBName, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Configure execution
    ep = ExecutePreprocessor(
        kernel_name=getKernelName(inNBName),
        timeout=3600,
        startup_timeout=180,
        allow_errors=False,
    )

    # Run the notebook
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(inNBName)}})

    # Export executed notebook
    with open(outNBName, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    # Compare or return success
    if needsCompare:
        return compareNotebooks(inNBName, outNBName)
    else:
        return 0  # success

    shutil.rmtree(tmpDir)


def isInputNotebookFileName(filename):
    if not filename.endswith(".ipynb"):
        print("Notebook files shall have the .ipynb extension")
        return False
    return True


if __name__ == "__main__":
    import sys

    needsCompare = True
    if len(sys.argv) < 2:
        print("Usage: nbdiff.py myNotebook.ipynb [compare_output]")
        sys.exit(1)
    elif len(sys.argv) == 3 and sys.argv[2] == "OFF":
        needsCompare = False

    nbFileName = sys.argv[1]
    if not isInputNotebookFileName(nbFileName):
        sys.exit(1)

    retCode = canReproduceNotebook(nbFileName, needsCompare)
    sys.exit(retCode)
