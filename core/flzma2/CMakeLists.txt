############################################################################
# CMakeLists.txt file for building ROOT core/lzmafl2 package
############################################################################

find_package(FLZMA2 REQUIRED)

#---Declare ZipLZMAFL2 sources as part of libCore-------------------------------
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ZipFLZMA2.h)
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipFLZMA2.cxx)

ROOT_OBJECT_LIBRARY(FLzma2 ${sources} BUILTINS FLZMA2)
target_include_directories(FLzma2 PRIVATE
   ${FLMA2_INCLUDE_DIRS}
   ${CMAKE_SOURCE_DIR}/core/foundation/inc
   ${CMAKE_BINARY_DIR}/ginclude
)

ROOT_INSTALL_HEADERS()
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
