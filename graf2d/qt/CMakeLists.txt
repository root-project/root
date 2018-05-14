############################################################################
# CMakeLists.txt file for building ROOT graf2d/qt package
# @author Pere Mato, CERN
############################################################################

if(fail-on-missing)
  set(QT_REQUIRED REQUIRED)
endif()

find_package(Qt4 4.8 COMPONENTS QtCore QtGui ${QT_REQUIRED})

if(NOT QT4_FOUND)
  set(qt OFF CACHE BOOL "" FORCE)
  return()
endif()

include(${QT_USE_FILE})

set(headers
  inc/TGQt.h
  inc/TQMimeTypes.h
  inc/TQtApplication.h
  inc/TQtBrush.h
  inc/TQtClientFilter.h
  inc/TQtClientWidget.h
  inc/TQtMarker.h
  inc/TQtPadFont.h
  inc/TQtRootSlot.h
  inc/TQtTimer.h
  inc/TQtWidget.h
  inc/TQtIncludes.h
)

set(sources
  src/GQtGUI.cxx
  src/TGQt.cxx
  src/TGQtDummy.cxx
  src/TQMimeTypes.cxx
  src/TQtApplication.cxx
  src/TQtBrush.cxx
  src/TQtClientFilter.cxx
  src/TQtClientGuard.cxx
  src/TQtClientWidget.cxx
  src/TQtEvent.cxx
  src/TQtEventQueue.cxx
  src/TQtMarker.cxx
  src/TQtPadFont.cxx
  src/TQtPen.cxx
  src/TQtRootApplication.cxx
  src/TQtRootSlot.cxx
  src/TQtSymbolCodec.cxx
  src/TQtTimer.cxx
  src/TQtWidget.cxx
)

QT4_WRAP_CPP(moc_sources
  inc/TQtWidget.h
  inc/TQtEmitter.h
  inc/TQtClientFilter.h
  inc/TQtClientGuard.h
  inc/TQtClientWidget.h
  inc/TQtTimer.h
  inc/TQtRootSlot.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(GQt
                              HEADERS ${headers}
                              SOURCES ${sources} ${moc_sources}
                              DICTIONARY_OPTIONS -Wno-deprecated
                              LIBRARIES Qt4::QtCore Qt4::QtGui
                              DEPENDENCIES Gui Gpad Graf Rint)
