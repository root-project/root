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
    __init__.py
    _graph_cache.py
    ComputationGraphGenerator.py
    DataFrame.py
    HeadNode.py
    LiveVisualize.py
    Node.py
    Operation.py
    Proxy.py
    PythonMergeables.py
    Ranges.py
    Backends/__init__.py
    Backends/Base.py
    Backends/Utils.py
    Backends/Dask/__init__.py
    Backends/Dask/Backend.py
    Backends/Spark/__init__.py
    Backends/Spark/Backend.py
  SUBDIR
    ROOT/_distrdf
)
