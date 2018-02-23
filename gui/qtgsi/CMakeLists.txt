############################################################################
# CMakeLists.txt file for building ROOT gui/qtgsi package
# @author Pere Mato, CERN
############################################################################

if(fail-on-missing)
  set(QT_REQUIRED REQUIRED)
endif()

find_package(Qt4 4.8 COMPONENTS QtCore QtGui Qt3Support ${QT_REQUIRED})

if(NOT QT4_FOUND)
  message(STATUS "Qt4 not found. Switching off qtgsi option")
  set(qtgsi OFF CACHE BOOL "" FORCE)
  return()
endif()

include(${QT_USE_FILE})

math(EXPR QT_VERSION_NUM "${QT_VERSION_MAJOR}*10000 + ${QT_VERSION_MINOR}*100 + ${QT_VERSION_PATCH}")

set(headers
  inc/TQApplication.h
  inc/TQCanvasImp.h
  inc/TQCanvasMenu.h
  inc/TQRootApplication.h
  inc/TQRootCanvas.h
  inc/TQRootDialog.h
  inc/TQRootGuiFactory.h
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

ROOT_STANDARD_LIBRARY_PACKAGE(QtGSI
                              HEADERS ${headers}
                              SOURCES ${sources} ${moc_sources}
                              DICTIONARY_OPTIONS -DQTVERS=0x${QT_VERSION_NUM} -Wno-deprecated
                              LIBRARIES Qt4::QtCore Qt4::QtGui Qt4::Qt3Support
                              DEPENDENCIES Gui Gpad)
