############################################################################
# CMakeLists.txt file for building ROOT core/meta package
############################################################################

ROOT_GLOB_FILES(sources FILTER "(^|/)TCling" ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
ROOT_GLOB_FILES(headers FILTER "(LinkDef|RootMetaSelection.h)" ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)

set(Meta_dict_headers ${headers} PARENT_SCOPE)
ROOT_OBJECT_LIBRARY(Meta ${sources})

##### libCling #############################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/TCling*.cxx)
add_definitions(${CLING_CXXFLAGS})
include_directories(${LLVM_INCLUDE_DIR} ${CLING_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/io/io/inc)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/TClingCallbacks.cxx COMPILE_FLAGS -fno-rtti)

ROOT_OBJECT_LIBRARY(MetaLLVM ${sources})
add_dependencies(MetaLLVM CLING)

#ROOT_GENERATE_DICTIONARY(G__Cling ${CMAKE_CURRENT_SOURCE_DIR}/src/TCling.h
#                         MODULE Cling DEPENDENCIES Core
#                         LINKDEF LinkDefCling.h)

ROOT_LINKER_LIBRARY(Cling
                    ${CMAKE_SOURCE_DIR}/core/metautils/src/TMetaUtils.cxx
                    $<TARGET_OBJECTS:MetaLLVM>
                    $<TARGET_OBJECTS:MetaUtilsLLVM>
                    LIBRARIES ${CLING_LIBRARIES} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_TINFO_LIBS}
                    DEPENDENCIES Core RIO)

ROOT_INSTALL_HEADERS()

