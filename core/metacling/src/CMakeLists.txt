# TInterpreter implementation for cling. Only in libCling; needs to resolve
# symbols from libCore.

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)

include_directories(${LLVM_INCLUDE_DIRS} ${CLING_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../res
        ${CMAKE_CURRENT_SOURCE_DIR}/../../clingutils/res
        ${CMAKE_CURRENT_SOURCE_DIR}/../../foundation/res)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLING_CXXFLAGS}")

if(MSVC)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingCallbacks.cxx COMPILE_FLAGS -GR-)
else()
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/TClingCallbacks.cxx COMPILE_FLAGS -fno-rtti)
endif()

# This to avoid warning coming from llvm/src/tools/clang/include/clang/Sema/Lookup.h:441
# in clang + llvm version r227800.
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-conditional-uninitialized)
endif()

# This is to avoid warnings coming from GCC 7 in llvm/src/include/llvm/ADT/DenseMap.h:1010
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-maybe-uninitialized)
endif()

ROOT_OBJECT_LIBRARY(MetaCling ${sources})
add_dependencies(MetaCling CLING)


##### libCling #############################################################

ROOT_LINKER_LIBRARY(Cling
        $<TARGET_OBJECTS:ClingUtils>
        $<TARGET_OBJECTS:Dictgen>
        $<TARGET_OBJECTS:MetaCling>
        LIBRARIES ${CLING_LIBRARIES})

if(APPLE)
target_link_libraries(Cling -Wl,-bind_at_load -Wl,-undefined -Wl,dynamic_lookup)
else()
target_link_libraries(Cling -Wl,--unresolved-symbols=ignore-in-object-files)
endif()
