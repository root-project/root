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

# Avoid warnings about deprecated register from Qt4
ROOT_ADD_CXX_FLAG(_GQt_FLAGS -Wno-deprecated-register)
ROOT_ADD_CXX_FLAG(_GQt_FLAGS -Wno-register)
separate_arguments(_GQt_FLAGS)

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
  HEADERS
    TGQt.h
    TQMimeTypes.h
    TQtApplication.h
    TQtBrush.h
    TQtClientFilter.h
    TQtClientWidget.h
    TQtMarker.h
    TQtPadFont.h
    TQtRootSlot.h
    TQtTimer.h
    TQtWidget.h
    TQtIncludes.h
  SOURCES
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
    ${moc_sources}
  DICTIONARY_OPTIONS
    ${_GQt_FLAGS}
  LIBRARIES
    Qt4::QtCore
    Qt4::QtGui
  DEPENDENCIES
    Gui
    Gpad
    Graf
    Rint
)

target_compile_options(GQt PRIVATE ${_GQt_FLAGS})
