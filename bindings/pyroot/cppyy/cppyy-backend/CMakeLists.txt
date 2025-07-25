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

set(cppyy_backendPySrcDir cling/python/cppyy_backend)
file(COPY ${cppyy_backendPySrcDir}
     DESTINATION ${localruntimedir}
     PATTERN "cmake" EXCLUDE
     PATTERN "pkg_templates" EXCLUDE)

add_library(cppyy_backend STATIC clingwrapper/src/clingwrapper.cxx)
target_link_libraries(cppyy_backend Core ${CMAKE_DL_LIBS})
target_compile_options(cppyy_backend PRIVATE -fPIC)

set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS cppyy_backend)

# Compile .py files
foreach(py_source ${py_sources})
  install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m py_compile ${localruntimedir}/${py_source})")
  install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -O -m py_compile ${localruntimedir}/${py_source})")
endforeach()


# Install Python sources and bytecode
install(DIRECTORY ${localruntimedir}/cppyy_backend
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
        COMPONENT libraries)
