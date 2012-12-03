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

# variable to be used in Makefiles.

ClingWorkAroundMissingImplicitAuto = yes
ClingWorkAroundMissingDynamicScope = yes
ClingWorkAroundMissingUnloading = yes
ClingWorkAroundMissingAutoLoading = yes
ClingWorkAroundMissingSmartInclude = yes
ClingWorkAroundErracticValuePrinter = yes
ClingWorkAroundBrokenUnnamedReturn = yes
ClingWorkAroundUnnamedIncorrectInitOrder = yes # See https://savannah.cern.ch/bugs/index.php?99210
