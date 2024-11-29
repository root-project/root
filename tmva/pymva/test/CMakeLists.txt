# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building PyMVA tests
# @author Stefan Wunsch
############################################################################

project(pymva-tests)

set(Libraries Core MathCore TMVA PyMVA ROOTTMVASofie)

# Look for needed python modules
find_python_module(torch QUIET)
find_python_module(keras QUIET)
find_python_module(theano QUIET)
find_python_module(tensorflow QUIET)
find_python_module(sklearn QUIET)

if(PY_SKLEARN_FOUND)
   # Test PyRandomForest: Classification
   ROOT_EXECUTABLE(testPyRandomForestClassification testPyRandomForestClassification.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-RandomForest-Classification COMMAND testPyRandomForestClassification)

   # Test PyRandomForest: Multi-class classification
   ROOT_EXECUTABLE(testPyRandomForestMulticlass testPyRandomForestMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-RandomForest-Multiclass COMMAND testPyRandomForestMulticlass)

   # Test PyGTB: Classification
   ROOT_EXECUTABLE(testPyGTBClassification testPyGTBClassification.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-GTB-Classification COMMAND testPyGTBClassification DEPENDS PyMVA-RandomForest-Classification)

   # Test PyGTB: Multi-class classification
   ROOT_EXECUTABLE(testPyGTBMulticlass testPyGTBMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-GTB-Multiclass COMMAND testPyGTBMulticlass DEPENDS PyMVA-RandomForest-Multiclass)

   # Test PyAdaBoost: Classification
   ROOT_EXECUTABLE(testPyAdaBoostClassification testPyAdaBoostClassification.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-AdaBoost-Classification COMMAND testPyAdaBoostClassification DEPENDS PyMVA-GTB-Classification)

   # Test PyAdaBoost: Multi-class classification
   ROOT_EXECUTABLE(testPyAdaBoostMulticlass testPyAdaBoostMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-AdaBoost-Multiclass COMMAND testPyAdaBoostMulticlass DEPENDS PyMVA-GTB-Multiclass)

endif(PY_SKLEARN_FOUND)


# Enable tests based on available python modules
if(PY_TORCH_FOUND)
   configure_file(generatePyTorchModelClassification.py generatePyTorchModelClassification.py COPYONLY)
   configure_file(generatePyTorchModelMulticlass.py generatePyTorchModelMulticlass.py COPYONLY)
   configure_file(generatePyTorchModelRegression.py generatePyTorchModelRegression.py COPYONLY)
   configure_file(generatePyTorchModels.py generatePyTorchModels.py COPYONLY)
   # Test PyTorch: Binary classification

   if (PY_SKLEARN_FOUND)
      set(PyMVA-Torch-Classification-depends PyMVA-AdaBoost-Classification)
      set(PyMVA-Torch-Multiclass-depends PyMVA-AdaBoost-Multiclass)
   endif()

   ROOT_EXECUTABLE(testPyTorchClassification testPyTorchClassification.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Torch-Classification COMMAND testPyTorchClassification DEPENDS ${PyMVA-Torch-Classification-depends})

   # Test PyTorch: Regression
   ROOT_EXECUTABLE(testPyTorchRegression testPyTorchRegression.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Torch-Regression COMMAND testPyTorchRegression)

   # Test PyTorch: Multi-class classification
   ROOT_EXECUTABLE(testPyTorchMulticlass testPyTorchMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Torch-Multiclass COMMAND testPyTorchMulticlass DEPENDS ${PyMVA-Torch-Multiclass-depends})

   # Test RModelParser_PyTorch

   ROOT_ADD_GTEST(TestRModelParserPyTorch TestRModelParserPyTorch.C
         LIBRARIES
         ROOTTMVASofie
         TMVA
         Python3::NumPy
         Python3::Python
         INCLUDE_DIRS
         SYSTEM
         ${CMAKE_CURRENT_BINARY_DIR}
      )
   target_link_libraries(TestRModelParserPyTorch ${BLAS_LINKER_FLAGS} ${BLAS_LIBRARIES})

endif(PY_TORCH_FOUND)

if((PY_KERAS_FOUND AND PY_THEANO_FOUND) OR (PY_KERAS_FOUND AND PY_TENSORFLOW_FOUND))
   configure_file(generateKerasModels.py generateKerasModels.py COPYONLY)
   configure_file(scale_by_2_op.hxx scale_by_2_op.hxx COPYONLY)

   if (PY_TORCH_FOUND)
      set(PyMVA-Keras-Classification-depends PyMVA-Torch-Classification)
      set(PyMVA-Keras-Regression-depends PyMVA-Torch-Regression)
      set(PyMVA-Keras-Multiclass-depends PyMVA-Torch-Multiclass)
   endif()


   # Test PyKeras: Binary classification
   ROOT_EXECUTABLE(testPyKerasClassification testPyKerasClassification.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Keras-Classification COMMAND testPyKerasClassification DEPENDS ${PyMVA-Keras-Classification-depends})

   # Test PyKeras: Regression
   #if (NOT ROOT_ARCHITECTURE MATCHES macosx)
   #veto also keras tutorial on macos due to issue in disabling eager execution on macos
      ROOT_EXECUTABLE(testPyKerasRegression testPyKerasRegression.C
         LIBRARIES ${Libraries})
      ROOT_ADD_TEST(PyMVA-Keras-Regression COMMAND testPyKerasRegression DEPENDS ${PyMVA-Keras-Regression-depends})
   #endif()


   # Test PyKeras: Multi-class classification
   ROOT_EXECUTABLE(testPyKerasMulticlass testPyKerasMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Keras-Multiclass COMMAND testPyKerasMulticlass DEPENDS ${PyMVA-Keras-Multiclass-depends})

   ROOT_ADD_GTEST(TestRModelParserKeras TestRModelParserKeras.C
    LIBRARIES
    ROOTTMVASofie
    PyMVA
    Python3::NumPy
    Python3::Python
    INCLUDE_DIRS
    SYSTEM
    ${CMAKE_CURRENT_BINARY_DIR}
   )
   target_link_libraries(TestRModelParserKeras ${BLAS_LINKER_FLAGS} ${BLAS_LIBRARIES})

endif((PY_KERAS_FOUND AND PY_THEANO_FOUND) OR (PY_KERAS_FOUND AND PY_TENSORFLOW_FOUND))
