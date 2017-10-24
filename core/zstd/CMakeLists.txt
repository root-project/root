############################################################################
# CMakeLists.txt file for building ROOT core/zstd package
############################################################################


#---The ZSTD library is built (or discovered) using the CMake ExternalProject standard module
#   in cmake/modules/SearchInstalledSoftare.cmake

#---Declare ZipZSTD sources as part of libCore-------------------------------
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/ZSTDEngine.hxx)
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZSTDEngine.cxx)


foreach(dir ${ZSTD_INCLUDE_DIR})
	include_directories(${dir})
endforeach()

ROOT_OBJECT_LIBRARY(Zstd ${sources})
target_compile_definitions(Zstd PRIVATE ${ZSTD_DEFINITIONS})

if(builtin_zstd)
  add_dependencies(Zstd ZSTD)
endif()

ROOT_INSTALL_HEADERS()
install(FILES ${ZSTD_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
