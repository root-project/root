# Build core/clib
# Define all header and source files which are needed
# to build the Core library
# The Core library is build in the main CMakeLists.txt
# of the core subdirectory.

Set(ZipOldHeaders
  ${CMAKE_CURRENT_SOURCE_DIR}/inc/Compression.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inc/RZip.h
)

Set(ZipOldSource
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ZDeflate.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ZInflate.c
)

# Depending on the system either the old and the new
# or only the old compression is build
set(ZLIB_HEADERS ${ZipOldHeaders})
set(ZLIB_SRCS ${ZipOldSource} ${CMAKE_CURRENT_SOURCE_DIR}/src/Compression.cxx ${CMAKE_CURRENT_SOURCE_DIR}/src/RZip.cxx)

# Define all the header files which should be installed when
# doing a "make install"
install(FILES ${ZLIB_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)
string(REPLACE ${CMAKE_SOURCE_DIR} "" tgt ${CMAKE_CURRENT_SOURCE_DIR})
string(MAKE_C_IDENTIFIER move_header${tgt} tgt)
add_custom_target(${tgt})
set_property(GLOBAL APPEND PROPERTY ROOT_HEADER_TARGETS ${tgt})
foreach(include_file ${ZLIB_HEADERS})
  add_custom_command(
    TARGET ${tgt}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${include_file} ${CMAKE_BINARY_DIR}/include
    DEPENDS ${include_file})
endforeach()
set_property(GLOBAL APPEND PROPERTY ROOT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)

ROOT_OBJECT_LIBRARY(Zip ${ZLIB_SRCS})
