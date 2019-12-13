# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(py_sources
  cppyy/_stdcpp_fix.py
  cppyy/__init__.py
  cppyy/_cpython_cppyy.py
  cppyy/_pypy_cppyy.py
  cppyy/_pythonization.py
  cppyy/_typemap.py
  cppyy/_version.py
  cppyy/interactive.py
)

foreach(py_source ${py_sources})
  file(COPY python/${py_source} DESTINATION ${py_localruntimedir}/cppyy)
  install(FILES python/${py_source} DESTINATION ${CMAKE_INSTALL_LIBDIR}/cppyy)
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m py_compile ${d}/${py_source})")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -O -m py_compile ${d}/${py_source})")
endforeach()

# create egg with cppyy info
execute_process(COMMAND bash -c "${PYTHON_EXECUTABLE} -c 'import setuptools; import distutils'; echo $?" OUTPUT_VARIABLE packages_installed)
execute_process(COMMAND bash -c "${PYTHON_EXECUTABLE} -c 'import setuptools as s; l = s.__version__.split(\".\"); l = l[0:2]; print(\".\".join(l))'" OUTPUT_VARIABLE setuptools_version)
if(packages_installed EQUAL 0 AND setuptools_version GREATER_EQUAL 39.0)
  add_custom_target(cppyy_egg ALL COMMAND ${PYTHON_EXECUTABLE} setup.py install_egg_info -d ${py_localruntimedir}/cppyy WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_custom_command(TARGET cppyy_egg POST_BUILD COMMAND bash -c "rm -rf python/*.egg-info" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
else()
  # create cppyy egg in a dummy way
  set(cppyy_egg "${CMAKE_CURRENT_BINARY_DIR}/cppyy.egg")
  execute_process(COMMAND bash -c "touch ${cppyy_egg}; cat ${CMAKE_CURRENT_SOURCE_DIR}/python/cppyy/_version.py > ${cppyy_egg}")
endif()
