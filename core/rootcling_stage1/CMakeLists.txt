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
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4217,4286")
else()
  if(CXX_HAS_fno_rtti)
    set_source_files_properties(src/rootcling_stage1.cxx PROPERTIES COMPILE_FLAGS "-fno-rtti")
  endif()
endif()

if(builtin_clang)
  set(CLING_LIBRARIES "clingMetaProcessor")
else()
  set(clang_libraries)
  if(MSVC)
    ROOT_GET_CLANG_LIBRARIES(clang_libraries)
  endif()
  list(APPEND CLING_LIBRARIES ${CLING_DEPEND_LIBS} "${clang_libraries}")
  link_directories("${LLVM_LIBRARY_DIR}")
endif()

ROOT_EXECUTABLE(rootcling_stage1 src/rootcling_stage1.cxx
                              $<TARGET_OBJECTS:ROOTclib>
                              $<TARGET_OBJECTS:ClingUtils>
                              $<TARGET_OBJECTS:Dictgen>
                              $<TARGET_OBJECTS:Foundation_Stage1>
                              LIBRARIES ${CLING_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${ROOT_ATOMIC_LIBS}
                              NOINSTALL)
target_compile_options(rootcling_stage1 PRIVATE -DCMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}")

target_include_directories(rootcling_stage1 PRIVATE
   ${CMAKE_SOURCE_DIR}/core/foundation/inc    # for RConfig.hxx
   ${CMAKE_SOURCE_DIR}/core/dictgen/res # for rootcling_impl.h
   ${CMAKE_BINARY_DIR}/ginclude         # for RConfigure.h
)
if (CMAKE_SYSTEM_NAME MATCHES FreeBSD)
  target_link_libraries(rootcling_stage1 util procstat)
endif()

if(NOT (WIN32 AND CMAKE_GENERATOR MATCHES Ninja))
  set_target_properties(rootcling_stage1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
endif()
add_dependencies(rootcling_stage1 ClingUtils)

# With Ninja, dictionaries depend on this stamp instead of the executable, see ROOT_GENERATE_DICTIONARY().
# It is written only once (Ninja restats the link edge), so like the target-level dependency it
# replaces, changes to rootcling_stage1 (e.g. dictgen) intentionally don't regenerate the dictionaries.
if(CMAKE_GENERATOR STREQUAL "Ninja")
  file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/empty.stamp)
  add_custom_command(TARGET rootcling_stage1 POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/empty.stamp
                             ${ROOTCLING_STAGE1_STAMP}
                     BYPRODUCTS ${ROOTCLING_STAGE1_STAMP})
endif()
