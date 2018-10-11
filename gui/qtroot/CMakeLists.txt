############################################################################
# CMakeLists.txt file for building ROOT gui/qtroot package
# @author Pere Mato, CERN
############################################################################

if(fail-on-missing)
  set(QT_REQUIRED REQUIRED)
endif()

find_package(Qt4 4.8 COMPONENTS QtCore QtGui ${QT_REQUIRED})

if(NOT QT4_FOUND)
  message(STATUS "Qt4 not found. Switching off qt option")
  set(qt OFF CACHE BOOL "" FORCE)
  return()
endif()

include(${QT_USE_FILE})

ROOT_STANDARD_LIBRARY_PACKAGE(QtRoot DEPENDENCIES Core Gui GQt)
