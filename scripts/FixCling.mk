#
# Set of macro passed to root.exe to tell the scripts to
# avoid features not yet implemented by cling.
#

CALLROOTEXE = root.exe
CALLROOTEXE += -e "\#define ClingWorkAroundMissingImplicitAuto"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingDynamicScope"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingUnloading"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingAutoLoading"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingSmartInclude"
CALLROOTEXE += -e "\#define ClingWorkAroundErracticValuePrinter"
CALLROOTEXE += -e "\#define ClingWorkAroundBrokenUnnamedReturn"
CALLROOTEXE += -e "\#define ClingWorkAroundUnnamedIncorrectInitOrder"
CALLROOTEXE += -e "\#define ClingWorkAroundUnnamedIncorrectFileLoc"
CALLROOTEXE += -e "\#define ClingWorkAroundPrintfIssues"

# variable to be used in Makefiles.

ClingWorkAroundMissingImplicitAuto = yes
ClingWorkAroundMissingDynamicScope = yes
ClingWorkAroundMissingUnloading = yes
ClingWorkAroundMissingAutoLoading = yes
ClingWorkAroundMissingSmartInclude = yes
ClingWorkAroundErracticValuePrinter = yes      # See https://savannah.cern.ch/bugs/index.php?98725
ClingWorkAroundBrokenUnnamedReturn = yes       # See https://savannah.cern.ch/bugs/index.php?99032
ClingWorkAroundUnnamedIncorrectInitOrder = yes # See https://savannah.cern.ch/bugs/index.php?99210
ClingWorkAroundUnnamedIncorrectFileLoc = yes   # see https://savannah.cern.ch/bugs/index.php?99236
ClingWorkAroundPrintfIssues = yes              # see https://savannah.cern.ch/bugs/index.php?99234
