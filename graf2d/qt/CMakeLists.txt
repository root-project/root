############################################################################
# CMakeLists.txt file for building ROOT graf2d/qt package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(gui/gui)
include(${QT_USE_FILE})

set(headers TGQt.h TQtTimer.h TQtApplication.h TQtBrush.h
            TQMimeTypes.h TQtClientFilter.h TQtClientWidget.h TQtWidget.h
            TQtMarker.h TQtTimer.h TQtRootSlot.h TQtPadFont.h)


QT4_WRAP_CPP(mocfiles inc/TQtWidget.h inc/TQtEmitter.h inc/TQtClientFilter.h
                      inc/TQtClientGuard.h inc/TQtClientWidget.h inc/TQtTimer.h
                      inc/TQtRootSlot.h)

ROOT_GENERATE_DICTIONARY(G__GQt ${headers} LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(GQt LINKDEF LinkDef.h DEPENDENCIES )

ROOT_LINKER_LIBRARY(GQt *.cxx ${mocfiles} G__GQt.cxx LIBRARIES ${QT_LIBRARIES})

ROOT_INSTALL_HEADERS()
