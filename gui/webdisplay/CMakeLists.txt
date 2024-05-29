# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/webdisplay package
############################################################################

if(NOT http)
  message(SEND_ERROR "webgui needs the option 'http' enabled.")
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTWebDisplay
    HEADERS
       ROOT/RWebDisplayArgs.hxx
       ROOT/RWebDisplayHandle.hxx
       ROOT/RWebWindow.hxx
       ROOT/RWebWindowsManager.hxx
    SOURCES
       RWebDisplayArgs.cxx
       RWebDisplayHandle.cxx
       RWebWindow.cxx
       RWebWindowsManager.cxx
    DEPENDENCIES
       Core
       RHTTP
       Net
       MathCore
)

ROOT_ADD_TEST_SUBDIRECTORY(test)

if(qt5web)
   target_compile_definitions(ROOTWebDisplay PRIVATE -DWITH_QT5WEB)
endif()

if(qt6web)
   target_compile_definitions(ROOTWebDisplay PRIVATE -DWITH_QT6WEB)
endif()

if(cefweb)
   target_compile_definitions(ROOTWebDisplay PRIVATE -DWITH_CEFWEB)
endif()