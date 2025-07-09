include(GNUInstallDirs)
include(headers.cmake)

# Turn relative paths to header files into absolute ones.
list(TRANSFORM histv7_headers PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/inc/)

add_library(ROOTHist INTERFACE)
target_include_directories(ROOTHist INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(ROOTHist INTERFACE cxx_std_17)

# Install header files manually: PUBLIC_HEADER has the disadvantage that CMake
# flattens the directory structure on install, and FILE_SETs are only available
# with CMake v3.23.
install(FILES ${histv7_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ROOT)
