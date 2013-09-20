############################################################################
# CMakeLists.txt file for building ROOT core/meta package
############################################################################

ROOT_GLOB_FILES(sources FILTER "(^|/)TCling" ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
ROOT_GLOB_HEADERS(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)

set(Meta_dict_headers ${headers} PARENT_SCOPE)
ROOT_OBJECT_LIBRARY(Meta ${sources})

##### libCling #############################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/TCling*.cxx)
add_definitions(${CLING_CXXFLAGS})
include_directories(${LLVM_INCLUDE_DIR} ${CLING_INCLUDE_DIR})
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/TClingCallbacks.cxx COMPILE_FLAGS -fno-rtti)

ROOT_GENERATE_DICTIONARY(G__Cling ${CMAKE_CURRENT_SOURCE_DIR}/src/TCling.h 
                         MODULE Cling DEPENDENCIES Core 
                         LINKDEF LinkDefCling.h)
ROOT_GENERATE_ROOTMAP(Cling LINKDEF LinkDefCling.h)
ROOT_LINKER_LIBRARY(Cling ${sources} G__Cling.cxx 
                    ${CMAKE_SOURCE_DIR}/core/metautils/src/TMetaUtils.cxx 
                    $<TARGET_OBJECTS:MetaUtilsLLVM>
                    LIBRARIES ${CLING_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} 
                    DEPENDENCIES Core)

ROOT_INSTALL_HEADERS()

