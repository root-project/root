############################################################################
# CMakeLists.txt file for building ROOT gui/qtroot package
# @author Pere Mato, CERN
############################################################################

include(${QT_USE_FILE})

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated-register)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-uninitialized)

ROOT_GENERATE_DICTIONARY(G__QtRoot *.h MODULE QtRoot LINKDEF LinkDef.h DEPENDENCIES Gui GQt)

ROOT_LINKER_LIBRARY(QtRoot *.cxx G__QtRoot.cxx LIBRARIES Core DEPENDENCIES Gui GQt)

ROOT_INSTALL_HEADERS()
