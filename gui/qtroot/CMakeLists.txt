############################################################################
# CMakeLists.txt file for building ROOT gui/qtroot package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(gui/gui)
ROOT_USE_PACKAGE(graf2d/qt)

include(${QT_USE_FILE})

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-Wno-deprecated-register -Wno-uninitialized)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  add_definitions(-Wno-deprecated -Wno-uninitialized)
endif()

ROOT_GENERATE_DICTIONARY(G__QtRoot *.h MODULE QtRoot LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(QtRoot *.cxx G__QtRoot.cxx LIBRARIES Core DEPENDENCIES Gui GQt)

ROOT_INSTALL_HEADERS()
