############################################################################
# CMakeLists.txt file for building ROOT core/zstd package
############################################################################
find_package(ZSTD REQUIRED)

#---Declare ZipZSTD sources as part of libCore-------------------------------
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ZipZSTD.hxx)
set(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/ZipZSTD.cxx)

foreach(dir ${ZSTD_INCLUDE_DIR})
	include_directories(${dir})
endforeach()

ROOT_OBJECT_LIBRARY(Zstd ${sources})
target_compile_definitions(Zstd PRIVATE ${ZSTD_DEFINITIONS})

ROOT_INSTALL_HEADERS()
install(FILES ${ZSTD_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})