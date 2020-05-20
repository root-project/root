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
  cppyy/ll.py
)

set(cppyyPySrcDir python/cppyy)
file(COPY ${cppyyPySrcDir} DESTINATION ${localruntimedir})

# Install .py source files
foreach(py_source ${py_sources})
  install(FILES python/${py_source} DESTINATION ${install_dir_lib}/cppyy)
endforeach()

foreach(val RANGE ${how_many_pythons})
  list(GET python_executables ${val} python_executable)

  # Install compiled .py files
  foreach(py_source ${py_sources})
    install(CODE "execute_process(COMMAND ${python_executable} -m py_compile ${install_dir_lib}/${py_source})")
    install(CODE "execute_process(COMMAND ${python_executable} -O -m py_compile ${install_dir_lib}/${py_source})")
  endforeach()
endforeach()
