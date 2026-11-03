# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(ROOT_JSROOT_VERSION 7.12.0)
set(ROOT_JSROOT_HASH "e5d82888750bda70090ac4bff355734000372c983a3eff11d67efa09f30d4b06")
# The GitHub release asset is named without the dots, JsRoot7120.tar.gz
string(REPLACE "." "" ROOT_JSROOT_VERSION_COMPACT ${ROOT_JSROOT_VERSION})

ExternalProject_Add(BUILTIN_JSROOT
  URL            ${lcgpackages}/jsroot-${ROOT_JSROOT_VERSION}.tar.gz
                 https://github.com/root-project/jsroot/releases/download/${ROOT_JSROOT_VERSION}/JsRoot${ROOT_JSROOT_VERSION_COMPACT}.tar.gz
  URL_HASH       SHA256=${ROOT_JSROOT_HASH}

#  GIT_REPOSITORY https://github.com/root-project/jsroot.git
#  GIT_TAG        7.12.0
#  GIT_SHALLOW    TRUE  # only works with tags/branches, comment out when commit hashes used
#  UPDATE_COMMAND ""
#  PATCH_COMMAND  ${CMAKE_COMMAND} -E rm -rf <SOURCE_DIR>/demo
#                                            <SOURCE_DIR>/fonts
#                                            <SOURCE_DIR>/docs
#                                            <SOURCE_DIR>/libs
#                                            <SOURCE_DIR>/server
#                                            <SOURCE_DIR>/.git
#                                            <SOURCE_DIR>/.github
#                                            <SOURCE_DIR>/api.htm
#                                            <SOURCE_DIR>/examples.htm

  SOURCE_DIR        ${CMAKE_BINARY_DIR}/js/
  TIMEOUT           600
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/js/   DESTINATION ${CMAKE_INSTALL_JSROOTDIR} COMPONENT libraries ${DIR_PERMISSIONS})
