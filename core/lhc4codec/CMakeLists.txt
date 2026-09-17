############################################################################
# CMakeLists.txt file for building ROOT core/lhc4codec package
############################################################################

add_library(ROOTLHC4Zip OBJECT src/ZipLHC4.cxx)
set_target_properties(ROOTLHC4Zip PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ROOTLHC4Zip PRIVATE LHC4CODEC::LHC4CODEC)
target_include_directories(ROOTLHC4Zip
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_sources(Core PRIVATE $<TARGET_OBJECTS:ROOTLHC4Zip>)

target_link_libraries(Core PRIVATE LHC4CODEC::LHC4CODEC)

# Bundled bzip3 is a separate static lib in the lhc4codec build; propagate it when
# built via add_subdirectory (ExternalProject uses INTERFACE_LINK_LIBRARIES).
if(TARGET lhc4_bzip3_bundled)
  target_link_libraries(Core PRIVATE lhc4_bzip3_bundled)
endif()

target_include_directories(Core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)

ROOT_INSTALL_HEADERS()
