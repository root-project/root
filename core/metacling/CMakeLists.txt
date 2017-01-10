############################################################################
# CMakeLists.txt file for building ROOT core/metacling package
############################################################################

# TInterpreter implementation for cling. Only in libCling; needs to resolve
# symbols from libCore.

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

include_directories(${LLVM_INCLUDE_DIRS} ${CLING_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res
        ${CMAKE_CURRENT_SOURCE_DIR}/../clingutils/res
        ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/res)
add_definitions(${CLING_CXXFLAGS})

if(MSVC)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/TClingCallbacks.cxx COMPILE_FLAGS -GR-)
else()
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/TClingCallbacks.cxx COMPILE_FLAGS -fno-rtti)
endif()

# This to avoid warning coming from  message coming from llvm/src/tools/clang/include/clang/Sema/Lookup.h:441
# in clang + llvm version r227800.
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-conditional-uninitialized)

ROOT_OBJECT_LIBRARY(MetaCling ${sources})
add_dependencies(MetaCling CLING)


##### libCling #############################################################

ROOT_LINKER_LIBRARY(Cling
        $<TARGET_OBJECTS:ClingUtils>
        $<TARGET_OBJECTS:Dictgen>
        $<TARGET_OBJECTS:MetaCling>
        LIBRARIES ${CLING_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT}
        DEPENDENCIES Core RIO)
