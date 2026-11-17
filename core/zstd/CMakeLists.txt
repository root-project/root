############################################################################
# CMakeLists.txt file for building ROOT core/zstd package
############################################################################

target_sources(Core
PRIVATE
  src/ZipZSTD.cxx
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    inc/ZipZSTD.h
)

target_link_libraries(Core PRIVATE ZSTD::ZSTD)

target_include_directories(Core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)

ROOT_INSTALL_HEADERS()
