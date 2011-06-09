############################################################################
# CMakeLists.txt file for building ROOT gui/qtroot package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(gui/gui)
ROOT_USE_PACKAGE(graf2d/qt)

include(${QT_USE_FILE})

ROOT_GENERATE_DICTIONARY(G__QtRoot *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(QtRoot LINKDEF LinkDef.h DEPENDENCIES Gui GQt )
ROOT_LINKER_LIBRARY(QtRoot *.cxx G__QtRoot.cxx LIBRARIES Core DEPENDENCIES Gui GQt)

ROOT_INSTALL_HEADERS()
