# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ExternalProject_Add(BUILTIN_JSROOT
  GIT_REPOSITORY https://github.com/root-project/jsroot.git
  GIT_TAG        7.12.0
  GIT_SHALLOW    TRUE  # only works with tags/branches, comment out when commit hashes used
  SOURCE_DIR     ${CMAKE_BINARY_DIR}/js/
  UPDATE_COMMAND ""
  PATCH_COMMAND  ${CMAKE_COMMAND} -E rm -rf <SOURCE_DIR>/demo
                                            <SOURCE_DIR>/fonts
                                            <SOURCE_DIR>/docs
                                            <SOURCE_DIR>/libs
                                            <SOURCE_DIR>/server
                                            <SOURCE_DIR>/.git
                                            <SOURCE_DIR>/.github
                                            <SOURCE_DIR>/api.htm
                                            <SOURCE_DIR>/examples.htm
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)
