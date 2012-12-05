#
# Set of macro passed to root.exe to tell the scripts to
# avoid features not yet implemented by cling.
#

CINT_VERSION := cling

CALLROOTEXE = root.exe
# Essential problems, must be fixed before the release.
# (incorrect behavior of C++ compliant code)
CALLROOTEXE += -e "\#define ClingWorkAroundUnnamedIncorrectInitOrder"
CALLROOTEXE += -e "\#define ClingWorkAroundPrintfIssues"
# Major features/issues
CALLROOTEXE += -e "\#define ClingWorkAroundMissingDynamicScope"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingAutoLoading"
CALLROOTEXE += -e "\#define ClingWorkAroundJITandInline"
CALLROOTEXE += -e "\#define ClingWorkAroundCallfuncAndInline"
CALLROOTEXE += -e "\#define ClingWorkAroundScriptClassDef"
# Convenience features, would be nice to have.
CALLROOTEXE += -e "\#define ClingWorkAroundBrokenRecovery"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingImplicitAuto"
CALLROOTEXE += -e "\#define ClingWorkAroundMissingSmartInclude"
CALLROOTEXE += -e "\#define ClingWorkAroundErracticValuePrinter"
CALLROOTEXE += -e "\#define ClingWorkAroundBrokenUnnamedReturn"
CALLROOTEXE += -e "\#define ClingWorkAroundUnnamedIncorrectFileLoc"
CALLROOTEXE += -e "\#define ClingWorkAroundUnnamedInclude"
CALLROOTEXE += -e "\#define ClingWorkAroundNoDotNamespace"

# Not for 6.0
CALLROOTEXE += -e "\#define ClingWorkAroundMissingUnloading"

# Fixes used when building library via ACLiC
CALLROOTEXEBUILD += -e "\#define ClingWorkAroundCallfuncAndInline"


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
ClingWorkAroundCallfuncAndInline = yes         # see https://savannah.cern.ch/bugs/index.php?98425
ClingWorkAroundUnnamedInclude = yes            # See https://savannah.cern.ch/bugs/index.php?99246
ClingWorkAroundBrokenRecovery = yes
ClingWorkAroundNoDotNamespace = yes            # See https://savannah.cern.ch/bugs/index.php?99288
ClingWorkAroundJITandInline = yes              # JIT does not instantiate inline even-though they are used (but not actually inlined)

