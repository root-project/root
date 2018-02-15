find_package(ZLIB REQUIRED)

set(ZIP_HEADERS
  inc/Compression.h
  inc/RZip.h
)

set(ZIP_SOURCES
  src/Bits.c
  src/ZDeflate.c
  src/ZTrees.c
  src/ZInflate.c
  src/Compression.cxx
  src/RZip.cxx
)

include_directories(${ZLIB_INCLUDE_DIR})
ROOT_OBJECT_LIBRARY(Zip ${ZIP_SOURCES})

add_custom_target(Zip_headers DEPENDS ${ZIP_HEADERS})

foreach(header ${ZIP_HEADERS})
  string(REPLACE "inc/" "" header ${header})
  set(src_header ${CMAKE_CURRENT_SOURCE_DIR}/inc/${header})
  set(dst_header ${CMAKE_BINARY_DIR}/include/${header})
  add_custom_command(TARGET Zip_headers BYPRODUCTS ${dst_header}
    COMMAND ${CMAKE_COMMAND} -E copy ${src_header} ${dst_header})
endforeach()

add_dependencies(Zip Zip_headers)

install(FILES ${ZIP_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)
