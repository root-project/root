############################################################################
# CMakeLists.txt file for building ROOT graf2d/qt package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(gui/gui)
include(${QT_USE_FILE})

set(headers TGQt.h TQtTimer.h TQtApplication.h TQtBrush.h
            TQMimeTypes.h TQtClientFilter.h TQtClientWidget.h TQtWidget.h
            TQtMarker.h TQtTimer.h TQtRootSlot.h TQtPadFont.h)

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-Wno-deprecated-register -Wno-uninitialized)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  add_definitions(-Wno-deprecated -Wno-uninitialized)
endif()

QT4_WRAP_CPP(mocfiles inc/TQtWidget.h inc/TQtEmitter.h inc/TQtClientFilter.h
                      inc/TQtClientGuard.h inc/TQtClientWidget.h inc/TQtTimer.h
                      inc/TQtRootSlot.h)

ROOT_GENERATE_DICTIONARY(G__GQt ${headers} MODULE GQt LINKDEF LinkDef.h OPTIONS -Wno-deprecated)


ROOT_LINKER_LIBRARY(GQt *.cxx ${mocfiles} G__GQt.cxx LIBRARIES ${QT_LIBRARIES} DEPENDENCIES Gui Gpad Graf Rint)

ROOT_INSTALL_HEADERS()
