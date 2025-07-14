import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile

nbExtension=".ipynb"
convCmdTmpl = "%s nbconvert " \
"--to notebook " \
"--ExecutePreprocessor.kernel_name=%s " \
"--ExecutePreprocessor.enabled=True " \
"--ExecutePreprocessor.timeout=3600 " \
"--ExecutePreprocessor.startup_timeout=180 " \
"%s " \
"--output %s"
pythonInterpName = 'python3'

rootKernelFileContent = '''{
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
''' %pythonInterpName



# Replace the criterion according to which a line shall be skipped
def customLineJunkFilter(line):
    # Skip the banner and empty lines
    junkLines =['Info in <TUnixSystem::ACLiC',
                'Info in <TMacOSXSystem::ACLiC',
                'FAILED TO establish the default connection to the WindowServer',
                '"version": ',
                '"pygments_lexer": "ipython',
                '     "execution_count":',
                'libclang_rt.asan-']
    for junkLine in junkLines:
        if junkLine in line: return False
    return True

def removeCellMetadata(lines):
    filteredLines = []
    discardLine = False
    for line in lines:
        if '   "metadata": {' in line:
            if line.endswith('},' + os.linesep): # empty metadata
                continue
            discardLine = True

        if not discardLine:
            filteredLines.append(line)

        if discardLine and '   },' in line: # end of metadata
            discardLine = False

    return filteredLines

def getFilteredLines(fileName):
    with open(fileName) as f:
        filteredLines = list(filter(customLineJunkFilter, f.readlines()))

    # Sometimes the jupyter server adds a new line at the end of the notebook
    # and nbconvert does not.
    lastLine = filteredLines[-1]
    if lastLine[-1] != "\n": filteredLines[-1] += "\n"

    # Remove the metadata field of cells (contains specific execution timestamps)
    filteredLines = removeCellMetadata(filteredLines)

    return filteredLines

# Workaround to support nbconvert versions >= 7.14 . See #14303
def patchForNBConvert714(outNBLines):
    newOutNBLines = []
    toReplace = '''      "1\\n"\n'''
    replacement = [
'''      "1"\n''',
'''     ]\n''',
'''    },\n''',
'''    {\n''',
'''     "name": "stdout",\n''',
'''     "output_type": "stream",\n''',
'''     "text": [\n''',
'''      "\\n"\n''']

    for line in outNBLines:
        if line == toReplace:
            newOutNBLines.extend(replacement)
        else:
            newOutNBLines.append(line)
    return newOutNBLines

def compareNotebooks(inNBName,outNBName):
    inNBLines = getFilteredLines(inNBName)
    inNBLines = patchForNBConvert714(inNBLines)
    outNBLines = getFilteredLines(outNBName)
    outNBLines = patchForNBConvert714(outNBLines)
    areDifferent = False
    for line in difflib.unified_diff(inNBLines, outNBLines, fromfile=inNBName, tofile=outNBName):
        areDifferent = True
        sys.stdout.write(line)
    if areDifferent: print("\n")
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
        kernel_file.write(rootKernelFileContent)

    return tmpd

def addEtcToEnvironment(inNBDirName):
    """Add the etc directory of root to the environment under the name of
    JUPYTER_PATH in order to pick up the kernel specs.
    """
    ipythondir = createKernelSpec()
    os.environ["JUPYTER_PATH"] = ipythondir
    os.environ["IPYTHONDIR"] = ipythondir
    return ipythondir

def getInterpreterName():
    """Find if the 'jupyter' executable is available on the platform. If
    yes, return its name else return 'ipython'
    """
    ret = subprocess.call("type jupyter",
                          shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return "jupyter" if ret == 0 else "i%s" %pythonInterpName

def getKernelName(inNBName):
    with open(inNBName) as f:
        nbj = json.load(f)
    if nbj["metadata"]["kernelspec"]["language"] == "python":
        return pythonInterpName
    else: # we support only Python and C++
        return 'root'


def canReproduceNotebook(inNBName, kernelName, needsCompare):
    tmpDir = addEtcToEnvironment(os.path.dirname(inNBName))
    outNBName = inNBName.replace(nbExtension,"_out"+nbExtension)
    interpName = getInterpreterName()
    convCmd = convCmdTmpl %(interpName, kernelName, inNBName, outNBName)
    exitStatus = os.system(convCmd) # we use system to inherit the environment in os.environ
    shutil.rmtree(tmpDir)
    if needsCompare:
        return compareNotebooks(inNBName,outNBName)
    else:
        return exitStatus

def isInputNotebookFileName(filename):
    if not filename.endswith(".ipynb"):
        print("Notebook files shall have the %s extension" %nbExtension)
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

    try:
        # If jupyter is there, ipython is too
        import jupyter
    except:
        raise ImportError("Cannot import jupyter")

    kernelName = getKernelName(nbFileName)

    retCode = canReproduceNotebook(nbFileName, kernelName, needsCompare)
    sys.exit(retCode)
