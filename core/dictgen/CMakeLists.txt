############################################################################
# CMakeLists.txt file for building ROOT core/dictgen package
############################################################################

# Provides capabilities to parse Linkdef / selection files and generate
# dictionaries.
#
# These files are linked into rootcling_stage1 and libCling. rootcling accesses
# them through rootcling_impl.

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/res
        ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/res
        ${CMAKE_CURRENT_SOURCE_DIR}/../clingutils/res
        ${LLVM_INCLUDE_DIRS}
        ${CLING_INCLUDE_DIRS})
add_definitions(${CLING_CXXFLAGS})
CHECK_CXX_COMPILER_FLAG("-fno-rtti" CXX_HAS_fno_rtti)
if(CXX_HAS_fno_rtti)
    set_source_files_properties(src/rootcling_impl.cxx   PROPERTIES COMPILE_FLAGS "-fno-rtti")
    set_source_files_properties(src/LinkdefReader.cxx    PROPERTIES COMPILE_FLAGS "-fno-rtti")
    set_source_files_properties(src/TModuleGenerator.cxx PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

ROOT_OBJECT_LIBRARY(Dictgen
  ${sources}
)

add_dependencies(Dictgen CLING)
