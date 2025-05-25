# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

###########################################################
# CMakeLists.txt file for building DistRDF
###########################################################

set(py_sources
  DistRDF/__init__.py
  DistRDF/_graph_cache.py
  DistRDF/ComputationGraphGenerator.py
  DistRDF/DataFrame.py
  DistRDF/HeadNode.py
  DistRDF/LiveVisualize.py
  DistRDF/Node.py
  DistRDF/Operation.py
  DistRDF/Proxy.py
  DistRDF/PythonMergeables.py
  DistRDF/Ranges.py
  DistRDF/Backends/__init__.py
  DistRDF/Backends/Base.py
  DistRDF/Backends/Utils.py
  DistRDF/Backends/Dask/__init__.py
  DistRDF/Backends/Dask/Backend.py
  DistRDF/Backends/Spark/__init__.py
  DistRDF/Backends/Spark/Backend.py
)

# Add custom rules to copy the Python sources into the build directory
foreach(py_source ${py_sources})
  add_custom_command(
      OUTPUT ${localruntimedir}/${py_source}
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/${py_source}
                                       ${localruntimedir}/${py_source}
      DEPENDS python/${py_source}
      COMMENT "Copying ${CMAKE_CURRENT_SOURCE_DIR}/python/${py_source}")
  list(APPEND py_sources_in_localruntimedir ${localruntimedir}/${py_source})
endforeach()

# A custom target that depends on the Python sources being present in the build
# directory. This will be used as a dependency of the pythonization libraries,
# such that the Python sources get re-copied to the build directory when
# changed.
add_custom_target(DistRDF ALL DEPENDS ${py_sources_in_localruntimedir})

# Compile .py files
# We include DistRDF in the build only if Python 3.8+ is used,
# so we can directly use the main Python executable to compile the sources
foreach(py_source ${py_sources})
  install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m py_compile ${localruntimedir}/${py_source})")
  install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -O -m py_compile ${localruntimedir}/${py_source})")
endforeach()

# Install Python sources and bytecode
install(DIRECTORY ${localruntimedir}/DistRDF
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
        COMPONENT libraries)

ROOT_ADD_TEST_SUBDIRECTORY(test)
ROOT_ADD_TEST_SUBDIRECTORY(test/backend)
