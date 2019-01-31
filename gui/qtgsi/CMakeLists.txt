############################################################################
# CMakeLists.txt file for building ROOT gui/qtgsi package
# @author Pere Mato, CERN
############################################################################

if(fail-on-missing)
  set(QT_REQUIRED REQUIRED)
endif()

find_package(Qt4 4.8 COMPONENTS QtCore QtGui ${QT_REQUIRED})

if(NOT QT4_FOUND)
  message(STATUS "Qt4 not found. Switching off qtgsi option")
  set(qtgsi OFF CACHE BOOL "" FORCE)
  return()
endif()

include(${QT_USE_FILE})

set(headers
  inc/TQApplication.h
  inc/TQCanvasImp.h
  inc/TQCanvasMenu.h
  inc/TQRootApplication.h
  inc/TQRootCanvas.h
  inc/TQRootDialog.h
  inc/TQRootGuiFactory.h
  inc/TQtGSIIncludes.h
)

set(sources
  src/TQApplication.cxx
  src/TQCanvasImp.cxx
  src/TQCanvasMenu.cxx
  src/TQRootApplication.cxx
  src/TQRootCanvas.cxx
  src/TQRootDialog.cxx
  src/TQRootGuiFactory.cxx
)

QT4_WRAP_CPP(moc_sources
  inc/TQCanvasMenu.h
  inc/TQRootApplication.h
  inc/TQRootCanvas.h
  inc/TQRootDialog.h
)

# Avoid warnings about deprecated register from Qt4
ROOT_ADD_CXX_FLAG(_QtGSI_FLAGS -Wno-deprecated-register)
ROOT_ADD_CXX_FLAG(_QtGSI_FLAGS -Wno-register)
separate_arguments(_QtGSI_FLAGS)

ROOT_STANDARD_LIBRARY_PACKAGE(QtGSI
                              HEADERS ${headers}
                              SOURCES ${sources} ${moc_sources}
                              LIBRARIES Qt4::QtCore Qt4::QtGui
                              DICTIONARY_OPTIONS ${_QtGSI_FLAGS}
                              DEPENDENCIES Gui Gpad)

target_compile_options(QtGSI PRIVATE ${_QtGSI_FLAGS})
