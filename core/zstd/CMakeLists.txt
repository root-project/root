############################################################################
# CMakeLists.txt file for building ROOT core/zstd package
############################################################################

find_package(ZSTD REQUIRED)

target_sources(Core PRIVATE src/ZipZSTD.cxx)
target_link_libraries(Core PRIVATE ${ZSTD_LIBRARIES})
target_compile_definitions(Core PRIVATE ${ZSTD_DEFINITIONS})
target_include_directories(Core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<BUILD_INTERFACE:${ZSTD_INCLUDE_DIR}>
)

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      Core
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/
      FILES
          inc/ZipZSTD.h
  )
endif()
ROOT_INSTALL_HEADERS()
install(FILES ${ZSTD_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
