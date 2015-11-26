import difflib
import subprocess
import shutil
import os

nbExtension=".ipynb"
convCmdTmpl = "ipython nbconvert  --to notebook --ExecutePreprocessor.enabled=True %s --output %s"

# Replace the criterion according to which a line shall be skipped
def customLineJunkFilter(line):
    # Skip the banner and empty lines
    junkLines =["Info in <TUnixSystem::ACLiC",
                "Info in <TMacOSXSystem::ACLiC",
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

class tmpDirCreator:
    def __init__(self,nbName):
       tmpDirName = nbName.replace(nbExtension,"") + "_profileDir"
       self.dirname = tmpDirName
    def __enter__(self):
       if not os.path.exists(self.dirname):
           print "[tmpDirCreator] Creating tmp directory %s" %self.dirname
           os.makedirs(self.dirname)
       return self
    def __exit__(self, type, value, traceback):
       if os.path.exists(self.dirname):
           shutil.rmtree(self.dirname)
           print "[tmpDirCreator] Deleting tmp directory %s" %self.dirname

def canReproduceNotebook(inNBName):
    outNBName = inNBName.replace(nbExtension,"_out"+nbExtension)
    convCmd = convCmdTmpl %(inNBName,outNBName)
    with tmpDirCreator(inNBName) as creator:    
        subprocess.check_output(convCmd.split(), env = dict(os.environ, IPYTHONDIR=creator.dirname))
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
