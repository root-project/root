# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

###########################################################
# CMakeLists.txt file for building PyRDF
###########################################################

set(PyRDFDir PyRDF)
file(COPY ${PyRDFDir} DESTINATION ${localruntimedir})

# Install Python sources and bytecode
install(DIRECTORY ${PyRDFDir}
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}
        COMPONENT libraries)

ROOT_ADD_TEST_SUBDIRECTORY(test/integration/local)
ROOT_ADD_TEST_SUBDIRECTORY(test/integration/spark)
ROOT_ADD_TEST_SUBDIRECTORY(test/unit)
ROOT_ADD_TEST_SUBDIRECTORY(test/unit/backend)
