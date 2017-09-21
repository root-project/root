############################################################################
# CMakeLists.txt file for building ROOT gui/qtroot package
# @author Pere Mato, CERN
############################################################################

include(${QT_USE_FILE})

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated-register)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-uninitialized)

ROOT_STANDARD_LIBRARY_PACKAGE(QtRoot
                              DEPENDENCIES Core Gui GQt)
