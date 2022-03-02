# Support library for diagnostic handling in ROOT unit tests.
# This library is linked into all googletest executables. It installs
# a ROOT message handler that triggers test failures when diagnostics
# higher than kInfo are issued by tests.
# Stephan Hageboeck, CERN, 2022

set(libname TestSupport)
set(header_dir ROOT/)

add_library(${libname} STATIC src/TestSupport.cxx)
target_include_directories(${libname} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/>
  $<INSTALL_INTERFACE:./>
  )
target_link_libraries(${libname} PUBLIC Core gtest)

# Installation of header and library:
set_target_properties(${libname} PROPERTIES PUBLIC_HEADER inc/${header_dir}/TestSupport.hxx)
install(TARGETS ${libname}
  EXPORT ${CMAKE_PROJECT_NAME}Exports
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${header_dir})
set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS ${libname})

# Make it usable inside and outside of ROOT under a single name if somebody writes their own tests using ROOT_ADD_GTEST
add_library(ROOT::${libname} ALIAS ${libname})

