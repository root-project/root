#
# Set of macro passed to root.exe to tell the scripts to
# avoid features not yet implemented by cling.
#

CALLROOTEXE = root.exe
CALLROOTEXE += -e "\#define ClingWorkAroundMissingDynamicScope"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingUnloading"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingAutoLoading"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingSmartInclude"

# variable to be used in Makefiles.

ClingWorkAroundMissingDynamicScope = yes
ClingWorkAroundMissingUnloading = yes
ClingWorkAroundMissingAutoLoading = yes
ClingWorkAroundMissingSmartInclude = yes
