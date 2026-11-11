# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for testing components from ROOT gui/webdisplay package
############################################################################


# test only can be run if Firefox or Chrome are detected on the system
# On Windows, ROOT starts the browser with 'wmic process call create' (RWebDisplayHandle.cxx).
# wmic is no longer installed by default on Windows 11 (24H2 and later), so skip the test there.
if(WIN32)
  find_program(WMIC_EXECUTABLE wmic)
endif()
if ((CHROME_EXECUTABLE OR FIREFOX_EXECUTABLE) AND (NOT WIN32 OR WMIC_EXECUTABLE))
  ROOT_ADD_TEST(test-webgui-ping
                RUN_SERIAL
                COPY_TO_BUILDDIR ${CMAKE_SOURCE_DIR}/tutorials/visualisation/webgui/ping/ping.cxx ${CMAKE_SOURCE_DIR}/tutorials/visualisation/webgui/ping/ping.html
                COMMAND root.exe -b -q ping.cxx
                PASSREGEX "PING-PONG TEST COMPLETED"
                TIMEOUT 300)
endif()
