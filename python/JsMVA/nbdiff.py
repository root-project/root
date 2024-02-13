import difflib
import subprocess
import shutil
import os

import sys
if sys.version_info >= (3, 0):
    kernelName = 'python3'
else:
    kernelName = 'python2'
nbExtension=".ipynb"
convCmdTmpl = "%s nbconvert  --to notebook --ExecutePreprocessor.kernel_name=%s --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=3600 %s --output %s"

# Replace the criterion according to which a line shall be skipped
def customLineJunkFilter(line):
    # Skip the banner and empty lines
    junkLines =["Info in <TUnixSystem::ACLiC",
                "Info in <TMacOSXSystem::ACLiC"]
    for junkLine in junkLines:
        if junkLine in line: return False
    return True

def getFilteredLines(fileName):
    filteredLines = list(filter(customLineJunkFilter, open(fileName).readlines()))
    # Sometimes the jupyter server adds a new line at the end of the notebook
    # and nbconvert does not.
    lastLine = filteredLines[-1]
    if lastLine[-1] != "\n": filteredLines[-1] += "\n"
    return filteredLines

def compareNotebooks(inNBName,outNBName):
    inNBLines = getFilteredLines(inNBName)
    outNBLines = getFilteredLines(outNBName)
    areDifferent = False
    for line in difflib.unified_diff(inNBLines, outNBLines, fromfile=inNBName, tofile=outNBName):
        areDifferent = True
        sys.stdout.write(line)
    if areDifferent: print("\n")
    return areDifferent

def getInterpreterName():
    """Find if the 'jupyter' executable is available on the platform. If
    yes, return its name else return 'ipython'
    """
    ret = subprocess.call("type jupyter",
                          shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return "jupyter" if ret == 0 else "ipython"

def canReproduceNotebook(inNBName):
    outNBName = inNBName.replace(nbExtension,"_out"+nbExtension)
    interpName = getInterpreterName()
    convCmd = convCmdTmpl %(interpName, kernelName, inNBName, outNBName)
    subprocess.check_output(convCmd.split(), env = os.environ)
    return compareNotebooks(inNBName,outNBName)

def isInputNotebookFileName(filename):
    if not filename.endswith(".ipynb"):
        print("Notebook files shall have the %s extension" %nbExtension)
        return False
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: nbdiff.py myNotebook.ipynb")
        sys.exit(1)
    nbFileName = sys.argv[1]
    if not isInputNotebookFileName(nbFileName):
        sys.exit(1)
    retCode = canReproduceNotebook(nbFileName)
    sys.exit(retCode)
