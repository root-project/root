############################################################################
# CMakeLists.txt file for building ROOT core/imt package
############################################################################

if(imt)
  include_directories(BEFORE "inc")

  ROOT_GENERATE_DICTIONARY(G__Imt STAGE1
    ROOT/TFuture.hxx
    ROOT/TPoolManager.hxx
    ROOT/TTaskGroup.hxx
    ROOT/TThreadExecutor.hxx
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
endif()

ROOT_LINKER_LIBRARY(Imt
    src/base.cxx
    src/TTaskGroup.cxx
  BUILTINS
    TBB
)

target_link_libraries(Imt PRIVATE Thread INTERFACE Core)

if(imt)
  target_include_directories(Imt PRIVATE ${TBB_INCLUDE_DIRS})
  target_link_libraries(Imt PRIVATE ${TBB_LIBRARIES})

  # G__Imt.cxx is automatically added by ROOT_LINKER_LIBRARY.
  target_sources(Imt PRIVATE
    src/TImplicitMT.cxx
    src/TPoolManager.cxx
    src/TThreadExecutor.cxx
  )

  ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()

ROOT_INSTALL_HEADERS()
