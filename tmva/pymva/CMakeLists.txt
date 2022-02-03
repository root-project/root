# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for PyMVA  package
############################################################################
#Author: Omar Zapata, Lorenzo Moneta, Sergei Glayzer 22/08/2015
#website:http://oproject.org

include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS_Development_Main} ${NUMPY_INCLUDE_DIRS})

ROOT_STANDARD_LIBRARY_PACKAGE(PyMVA
  HEADERS
    TMVA/MethodPyAdaBoost.h
    TMVA/MethodPyGTB.h
    TMVA/MethodPyKeras.h
    TMVA/MethodPyRandomForest.h
    TMVA/MethodPyTorch.h
    TMVA/RModelParser_Keras.h
    TMVA/RModelParser_PyTorch.h
    TMVA/PyMethodBase.h
  SOURCES
    src/MethodPyAdaBoost.cxx
    src/MethodPyGTB.cxx
    src/MethodPyKeras.cxx
    src/MethodPyRandomForest.cxx
    src/MethodPyTorch.cxx
    src/RModelParser_Keras.cxx
    src/RModelParser_PyTorch.cxx
    src/PyMethodBase.cxx
  LIBRARIES
  ${PYTHON_LIBRARIES_Development_Main}
  DEPENDENCIES
    Core
    Matrix
    Thread
    RIO
    TMVA
    ROOTTMVASofie
)

if(APPLE)
    target_link_options(PyMVA PRIVATE ${PYTHON_LINK_OPTIONS_Development_Main})
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
