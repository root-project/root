############################################################################
# CMakeLists.txt file for building ROOT core/compression/zstd package
############################################################################

target_sources(Core PRIVATE src/ZipZSTD.cxx)

target_link_libraries(Core PRIVATE ZSTD::ZSTD)

target_include_directories(Core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)

ROOT_INSTALL_HEADERS()
