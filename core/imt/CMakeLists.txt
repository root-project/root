# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/imt package
############################################################################

ROOT_LINKER_LIBRARY(Imt
    src/base.cxx
    src/TTaskGroup.cxx
  BUILTINS
    TBB
)

target_link_libraries(Imt PRIVATE Thread INTERFACE Core)

if(imt)
  ROOT_GENERATE_DICTIONARY(G__Imt STAGE1
    ROOT/TFuture.hxx
    ROOT/TPoolManager.hxx
    ROOT/TTaskGroup.hxx
    ROOT/TThreadExecutor.hxx
    NOTARGET
    LINKDEF
      LinkDef.h
    MODULE
      Imt
    DEPENDENCIES
      Core
      Thread
    BUILTINS
      TBB
  )

  # G__Imt.cxx is automatically added by ROOT_GENERATE_DICTIONARY()
  target_sources(Imt PRIVATE
    src/TImplicitMT.cxx
    src/TPoolManager.cxx
    src/TThreadExecutor.cxx
  )

  target_include_directories(Imt PRIVATE ${TBB_INCLUDE_DIRS})
  target_link_libraries(Imt PRIVATE ${TBB_LIBRARIES})

  ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()

ROOT_INSTALL_HEADERS()
