############################################################################
# CMakeLists.txt file for building ROOT core/zstd package
############################################################################
find_package(ZSTD REQUIRED)

#---Declare ZipZSTD sources as part of libCore-------------------------------
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ZipZSTD.hxx)
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipZSTD.cxx)

ROOT_OBJECT_LIBRARY(Zstd ${sources} BUILTINS ZSTD)
target_compile_definitions(Zstd PRIVATE ${ZSTD_DEFINITIONS})
target_include_directories(Zstd PRIVATE ${ZSTD_INCLUDE_DIR})

ROOT_INSTALL_HEADERS()
install(FILES ${ZSTD_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
