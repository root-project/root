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

set(d $ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX}/${runtimedir})
foreach(py_source ${py_sources})
  file(COPY python/${py_source} DESTINATION ${localruntimedir}/cppyy)
  install(FILES python/${py_source} DESTINATION ${runtimedir}/cppyy)
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m py_compile ${d}/${py_source})")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -O -m py_compile ${d}/${py_source})")
endforeach()
