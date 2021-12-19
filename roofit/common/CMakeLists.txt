############################################################################
# CMakeLists.txt file for building ROOT roofit/common package
# @author Jonas Rembser, CERN
############################################################################

# The RooFitCommon library is at the root of the dependency graph of the RooFit
# libraries. Is is not allowed to depend on other RooFit libraries, but any
# other RooFit library can depend on it.

ROOT_LINKER_LIBRARY(RooFitCommon
    src/Common.cxx
  DEPENDENCIES
    Core
)
