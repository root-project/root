############################################################################
# CMakeLists.txt file for building ROOT graf2d/qt package
# @author Pere Mato, CERN
############################################################################

include(${QT_USE_FILE})

set(headers TGQt.h TQtApplication.h TQtBrush.h
            TQMimeTypes.h TQtClientFilter.h TQtClientWidget.h TQtWidget.h
            TQtMarker.h TQtTimer.h TQtRootSlot.h TQtPadFont.h)

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated-register)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-uninitialized)

QT4_WRAP_CPP(mocfiles inc/TQtWidget.h inc/TQtEmitter.h inc/TQtClientFilter.h
                      inc/TQtClientGuard.h inc/TQtClientWidget.h inc/TQtTimer.h
                      inc/TQtRootSlot.h)

ROOT_GENERATE_DICTIONARY(G__GQt ${headers} MODULE GQt LINKDEF LinkDef.h OPTIONS -Wno-deprecated)


ROOT_LINKER_LIBRARY(GQt *.cxx ${mocfiles} G__GQt.cxx LIBRARIES ${QT_LIBRARIES} DEPENDENCIES Gui Gpad Graf Rint)

ROOT_INSTALL_HEADERS()
