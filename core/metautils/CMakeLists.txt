############################################################################
# CMakeLists.txt file for building ROOT core/cont package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
set(sources_llvm  ${CMAKE_CURRENT_SOURCE_DIR}/src/BaseSelectionRule.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/ClassSelectionRule.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/VariableSelectionRule.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/RStl.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/Scanner.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/SelectionRules.cxx
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/XMLReader.cxx)
list(REMOVE_ITEM sources ${sources_llvm} ${CMAKE_CURRENT_SOURCE_DIR}/src/TMetaUtils.cxx)

ROOT_GLOB_HEADERS(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/TMetaUtils.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/libcpp_string_view.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/RWrap_libcpp_string_view.h)

set(MetaUtils_dict_headers ${headers} PARENT_SCOPE)

include_directories(${LLVM_INCLUDE_DIRS} ${CLING_INCLUDE_DIRS})
add_definitions(${CLING_CXXFLAGS})

ROOT_OBJECT_LIBRARY(MetaUtils ${sources})
ROOT_OBJECT_LIBRARY(MetaUtilsLLVM ${sources_llvm})

add_dependencies(MetaUtils CLING)
add_dependencies(MetaUtilsLLVM CLING)

ROOT_INSTALL_HEADERS()

#### STL dictionary (replacement for cintdlls)##############################

set(stldicts
    vector
    list
    forward_list
    deque
    map map2 unordered_map
    multimap multimap2 unordered_multimap
    set unordered_set
    multiset unordered_multiset
    complex)
if(NOT WIN32)
  list(APPEND stldicts valarray)
endif()
include_directories(${CMAKE_BINARY_DIR}/etc/cling/cint)
foreach(dict ${stldicts})
  string(REPLACE "2" "" header ${dict})
  string(REPLACE "complex" "root_std_complex.h" header ${header})
  string(REPLACE "multi" "" header ${header})
  ROOT_GENERATE_DICTIONARY(G__std_${dict} STAGE1 MODULE ${dict}Dict OPTIONS ${header} LINKDEF src/${dict}Linkdef.h)
  ROOT_LINKER_LIBRARY(${dict}Dict G__std_${dict}.cxx DEPENDENCIES  Core)
endforeach()
