############################################################################
# CMakeLists.txt file for building ROOT core/lz4 package
############################################################################


#---The builtin LZ4 library is built using the CMake ExternalProject standard module
#   in cmake/modules/SearchInstalledSoftare.cmake

#---Declare ZipLZ4 sources as part of libCore-------------------------------
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ZipLZ4.h)
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipLZ4.cxx)


foreach(dir ${LZ4_INCLUDE_DIR})
	include_directories(${dir})
endforeach()

ROOT_OBJECT_LIBRARY(Lz4 ${sources})
target_compile_definitions(Lz4 PRIVATE ${LZ4_DEFINITIONS})

if(builtin_lz4)
  add_dependencies(Lz4 LZ4)
endif()

ROOT_INSTALL_HEADERS()
install(FILES ${LZ4_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
