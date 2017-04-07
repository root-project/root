############################################################################
# CMakeLists.txt file for building ROOT proof/pq2 package
# @author Pere Mato, CERN
############################################################################

ROOT_EXECUTABLE(pq2 *.cxx LIBRARIES Core Proof Hist Matrix Tree
  RIO Net Thread MathCore)
