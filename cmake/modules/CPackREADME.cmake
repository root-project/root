# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#---------------------------------------------------------------------------------------------------
#  CPackREADME.cmake
#   - Pre-packaging script to convert README.md to html
#---------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------
# Apple productbuild cannot handle .md files as CPACK_PACKAGE_DESCRIPTION_FILE;
# convert to HTML instead.
#
if (APPLE)
  find_program(CONVERTER textutil)
  if (NOT CONVERTER)
    message(FATAL_ERROR "textutil executable not found")
  endif()
  execute_process(COMMAND ${CONVERTER} -convert html README.md -output "${CMAKE_BINARY_DIR}/README.html")
  set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_BINARY_DIR}/README.html")
  set(CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/README.html")
endif()
