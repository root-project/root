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
set_property(GLOBAL APPEND PROPERTY ROOT_HEADER_TARGETS ${tgt})
set(dst_list)
foreach(include_file ${ZLIB_HEADERS})
  get_filename_component(incname ${include_file} NAME)
  set(dst ${CMAKE_BINARY_DIR}/include/${incname})
  add_custom_command(
    OUTPUT ${dst}
    COMMAND ${CMAKE_COMMAND} -E copy ${include_file} ${dst}
    DEPENDS ${include_file})
    list(APPEND dst_list ${dst})  
endforeach()
add_custom_target(${tgt} DEPENDS ${dst_list})
set_property(GLOBAL APPEND PROPERTY ROOT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)

ROOT_OBJECT_LIBRARY(Zip ${ZLIB_SRCS})
