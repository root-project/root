import difflib
import subprocess

nbExtension=".ipynb"
convCmdTmpl = "ipython nbconvert  --to notebook --ExecutePreprocessor.enabled=True %s --output %s"

# Replace the criterion according to which a line shall be skipped
def customLineJunkFilter(line):
    # Skip the banner and empty lines
    junkLines =["Info in <TUnixSystem::ACLiC",
                "Welcome to ROOTaaS 6."]
    for junkLine in junkLines:
        if junkLine in line: return False
    return True

def getFilteredLines(fileName):
    filteredLines =  filter(customLineJunkFilter, open(fileName).readlines())
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
    if areDifferent: print "\n"
    return areDifferent

def canReproduceNotebook(inNBName):
    outNBName = inNBName.replace(nbExtension,"_out"+nbExtension)
    convCmd = convCmdTmpl %(inNBName,outNBName)
    subprocess.check_output(convCmd.split())
    return compareNotebooks(inNBName,outNBName)

def isInputNotebookFileName(filename):
    if not filename.endswith(".ipynb"):
        print "Notebook files shall have the %s extension" %nbExtension
        return False
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print "Usage: nbdiff.py myNotebook.ipynb"
        sys.exit(1)
    nbFileName = sys.argv[1]
    if not isInputNotebookFileName(nbFileName):
        sys.exit(1)
    retCode = canReproduceNotebook(nbFileName)
    sys.exit(retCode)
