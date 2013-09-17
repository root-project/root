############################################################################
# CMakeLists.txt file for building ROOT core/cont package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
set(sources_llvm  ${CMAKE_CURRENT_SOURCE_DIR}/src/BaseSelectionRule.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/ClassSelectionRule.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/VariableSelectionRule.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/RClStl.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/Scanner.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/SelectionRules.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/XMLReader.cxx)
list(REMOVE_ITEM sources ${sources_llvm} ${CMAKE_CURRENT_SOURCE_DIR}/src/TMetaUtils.cxx)

ROOT_GLOB_HEADERS(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/TMetaUtils.h)

set(MetaUtils_dict_headers ${headers} PARENT_SCOPE)

include_directories(${LLVM_INCLUDE_DIR} ${CLING_INCLUDE_DIR})
add_definitions(${CLING_CXXFLAGS})

ROOT_OBJECT_LIBRARY(MetaUtils ${sources})
ROOT_OBJECT_LIBRARY(MetaUtilsLLVM ${sources_llvm})

add_dependencies(MetaUtils CLING)
add_dependencies(MetaUtilsLLVM CLING)

ROOT_INSTALL_HEADERS()

####CMakeTODO
#### STL dictionary (replacement for cintdlls)


