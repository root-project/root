# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

###########################################################
# CMakeLists.txt file for building DistRDF
###########################################################

set(DistRDFDir python/DistRDF)
file(COPY ${DistRDFDir} DESTINATION ${localruntimedir})

set(py_sources
  DistRDF/__init__.py
  DistRDF/ComputationGraphGenerator.py
  DistRDF/DataFrame.py
  DistRDF/Node.py
  DistRDF/Operation.py
  DistRDF/Proxy.py
  DistRDF/Backends/__init__.py
  DistRDF/Backends/Base.py
  DistRDF/Backends/Utils.py
  DistRDF/Backends/Spark/__init__.py
  DistRDF/Backends/Spark/Backend.py
)

# Compile .py files
# We include DistRDF in the build only if Python 3.7+ is used,
# so we can directly use the main Python executable to compile the sources
foreach(py_source ${py_sources})
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE_Development_Main} -m py_compile ${localruntimedir}/${py_source})")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE_Development_Main} -O -m py_compile ${localruntimedir}/${py_source})")
endforeach()

# Install Python sources and bytecode
install(DIRECTORY ${localruntimedir}/DistRDF
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
        COMPONENT libraries)

ROOT_ADD_TEST_SUBDIRECTORY(test)
ROOT_ADD_TEST_SUBDIRECTORY(test/backend)
