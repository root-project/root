# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/rootcling_stage1 package
############################################################################

if(WIN32)
  set_source_files_properties(src/rootcling_stage1.cxx PROPERTIES COMPILE_FLAGS "-D_WIN32 -DNOMINMAX")
else()
  if(CXX_HAS_fno_rtti)
    set_source_files_properties(src/rootcling_stage1.cxx PROPERTIES COMPILE_FLAGS "-fno-rtti")
  endif()
endif()

if(NOT builtin_clang)
  list(APPEND CLING_LIBRARIES ${CLING_DEPEND_LIBS})
  link_directories("${LLVM_LIBRARY_DIR}")
endif()

ROOT_EXECUTABLE(rootcling_stage1 src/rootcling_stage1.cxx
                              $<TARGET_OBJECTS:Clib>
                              $<TARGET_OBJECTS:ClingUtils>
                              $<TARGET_OBJECTS:Dictgen>
                              $<TARGET_OBJECTS:Foundation_Stage1>
                              LIBRARIES ${CLING_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT}
                              NOINSTALL)

target_include_directories(rootcling_stage1 PRIVATE
   ${CMAKE_SOURCE_DIR}/core/foundation/inc    # for RConfig.hxx
   ${CMAKE_SOURCE_DIR}/core/dictgen/res # for rootcling_impl.h
   ${CMAKE_BINARY_DIR}/ginclude         # for RConfigure.h
)

set_target_properties(rootcling_stage1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY src)
add_dependencies(rootcling_stage1 ClingUtils)
