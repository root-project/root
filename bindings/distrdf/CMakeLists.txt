# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

###########################################################
# CMakeLists.txt file for building DistRDF
###########################################################

ROOT_PYTHON_PACKAGE(DistRDF
  SOURCES
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

ROOT_ADD_TEST_SUBDIRECTORY(test)
ROOT_ADD_TEST_SUBDIRECTORY(test/backend)
