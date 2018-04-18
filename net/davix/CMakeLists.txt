############################################################################
# CMakeLists.txt file for building ROOT net/davix package
# @author Pere Mato, CERN
############################################################################

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated-declarations) # Davix getAllReplicas
ROOT_STANDARD_LIBRARY_PACKAGE(RDAVIX LIBRARIES Davix::Davix DEPENDENCIES RIO Net)
