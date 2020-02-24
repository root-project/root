# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(py_sources
  cppyy_backend/__init__.py
  cppyy_backend/_cling_config.py
  cppyy_backend/_cppyy_generator.py
  cppyy_backend/_genreflex.py
  cppyy_backend/_rootcling.py
  cppyy_backend/bindings_utils.py
  cppyy_backend/loader.py
  cppyy_backend/_get_cppflags.py
)

foreach(py_source ${py_sources})
  file(COPY cling/python/${py_source} DESTINATION ${localruntimedir}/cppyy_backend)
  install(FILES cling/python/${py_source} DESTINATION ${CMAKE_INSTALL_LIBDIR}/cppyy_backend)
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m py_compile ${d}/${py_source})")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -O -m py_compile ${d}/${py_source})")
endforeach()

add_library(cppyy_backend SHARED clingwrapper/src/clingwrapper.cxx)
target_include_directories(cppyy_backend PRIVATE clingwrapper/src)
target_link_libraries(cppyy_backend Core ${CMAKE_DL_LIBS})

# cppyy uses ROOT headers from binary directory
add_dependencies(cppyy_backend move_headers)
target_include_directories(cppyy_backend PRIVATE ${CMAKE_BINARY_DIR}/include)

set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS cppyy_backend)
install(TARGETS cppyy_backend EXPORT ROOTExports DESTINATION ${CMAKE_INSTALL_LIBDIR})
