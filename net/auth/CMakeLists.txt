# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/auth package
# @author Pere Mato, CERN
############################################################################

add_library(rsa STATIC src/rsaaux.cxx src/rsalib.cxx src/rsafun.cxx)
target_include_directories(rsa PUBLIC res)
set_property(TARGET rsa PROPERTY POSITION_INDEPENDENT_CODE ON)

ROOT_STANDARD_LIBRARY_PACKAGE(RootAuth
  HEADERS
    AuthConst.h
    TAuthenticate.h
    THostAuth.h
    TRootAuth.h
    TRootSecContext.h
  SOURCES
    src/TAuthenticate.cxx
    src/THostAuth.cxx
    src/TRootAuth.cxx
    src/TRootSecContext.cxx
  LINKDEF
    LinkDefRoot.h
  DEPENDENCIES
    Net
    RIO
)

target_link_libraries(RootAuth PRIVATE rsa $<$<PLATFORM_ID:Linux>:crypt>)
