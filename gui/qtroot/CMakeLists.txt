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

# Avoid warnings about deprecated register from Qt4
ROOT_ADD_CXX_FLAG(_QtRoot_FLAGS -Wno-deprecated-register)
ROOT_ADD_CXX_FLAG(_QtRoot_FLAGS -Wno-register)
separate_arguments(_QtRoot_FLAGS)
target_compile_options(QtRoot PRIVATE ${_QtRoot_FLAGS})
