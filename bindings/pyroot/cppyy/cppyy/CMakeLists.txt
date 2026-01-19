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
  cppyy/numba_ext.py
  cppyy/reflex.py
  cppyy/types.py
)

# Ensure output directory exists
file(MAKE_DIRECTORY ${localruntimedir}/cppyy)

set(cppyy_copy_commands)
set(cppyy_py_sources_in_source_dir)
set(cppyy_py_sources_in_localruntimedir)

foreach(py_source ${py_sources})
  list(APPEND cppyy_copy_commands
       COMMAND ${CMAKE_COMMAND} -E copy_if_different
               ${CMAKE_CURRENT_SOURCE_DIR}/python/${py_source}
               ${localruntimedir}/${py_source})

  list(APPEND cppyy_py_sources_in_source_dir
       ${CMAKE_CURRENT_SOURCE_DIR}/python/${py_source})

  list(APPEND cppyy_py_sources_in_localruntimedir
       ${localruntimedir}/${py_source})
endforeach()

add_custom_command(
  OUTPUT ${cppyy_py_sources_in_localruntimedir}
  ${cppyy_copy_commands}
  DEPENDS ${cppyy_py_sources_in_source_dir}
  COMMENT "Copying cppyy Python sources"
)

add_custom_target(cppyyPySources ALL DEPENDS ${cppyy_py_sources_in_localruntimedir})

# Compile .py files

# Stamp file so CMake knows it doesn't need to re-compile. We can't set the
# actual bytecode files as the OUTPUT of the custom command, because their
# names are CPython implementation dependent and therefore not reliable.
set(cppyy_bytecode_stamp ${localruntimedir}/cppyy/bytecode.stamp)

# It's 10x faster to compile all in one go than in single invocations
add_custom_command(
  OUTPUT ${cppyy_bytecode_stamp}
  COMMAND ${Python3_EXECUTABLE} -m py_compile ${cppyy_py_sources_in_localruntimedir}
  COMMAND ${Python3_EXECUTABLE} -O -m py_compile ${cppyy_py_sources_in_localruntimedir}
  COMMAND ${CMAKE_COMMAND} -E touch ${cppyy_bytecode_stamp}
  DEPENDS ${cppyy_py_sources_in_localruntimedir}
  COMMENT "Compiling cppyy Python sources"
)

add_custom_target(cppyyPyBytecode ALL DEPENDS ${cppyy_bytecode_stamp})
add_dependencies(cppyyPyBytecode cppyyPySources)

# Install Python sources and bytecode
install(DIRECTORY ${localruntimedir}/cppyy
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
        COMPONENT libraries
        PATTERN *.stamp EXCLUDE
        PATTERN *.so EXCLUDE)

ROOT_ADD_TEST_SUBDIRECTORY(test)
