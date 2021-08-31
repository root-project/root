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

set(Libraries Core MathCore TMVA PyMVA)
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

# Look for needed python modules
find_python_module(torch QUIET)
find_python_module(keras QUIET)
find_python_module(theano QUIET)
find_python_module(tensorflow QUIET)
find_python_module(sklearn QUIET)

# Enable tests based on available python modules
if(PY_TORCH_FOUND)
   configure_file(generatePyTorchModelClassification.py generatePyTorchModelClassification.py COPYONLY)
   configure_file(generatePyTorchModelMulticlass.py generatePyTorchModelMulticlass.py COPYONLY)
   configure_file(generatePyTorchModelRegression.py generatePyTorchModelRegression.py COPYONLY)
   configure_file(generatePyTorchModelModule.py generatePyTorchModelModule.py COPYONLY)
   configure_file(generatePyTorchModelSequential.py generatePyTorchModelSequential.py COPYONLY)
   # Test PyTorch: Binary classification${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS}
   ROOT_EXECUTABLE(testPyTorchClassification testPyTorchClassification.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Torch-Classification COMMAND testPyTorchClassification)

   # Test PyTorch: Regression
   ROOT_EXECUTABLE(testPyTorchRegression testPyTorchRegression.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Torch-Regression COMMAND testPyTorchRegression)

   # Test PyTorch: Multi-class classification
   ROOT_EXECUTABLE(testPyTorchMulticlass testPyTorchMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Torch-Multiclass COMMAND testPyTorchMulticlass)

   # Test RModelParser_PyTorch
   add_executable(emitFromPyTorch
                  EmitFromPyTorch.cxx
                  ${CMAKE_SOURCE_DIR}/tmva/pymva/src/RModelParser_PyTorch.cxx
                  ${CMAKE_SOURCE_DIR}/tmva/sofie/src/RModel.cxx
                  ${CMAKE_SOURCE_DIR}/tmva/sofie/src/SOFIE_common.cxx
                 )
   target_link_libraries(emitFromPyTorch ${PYTHON_LIBRARIES} ${Libraries})
   target_include_directories(emitFromPyTorch PRIVATE
                              ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
                              ${CMAKE_SOURCE_DIR}/tmva/inc
                              ${CMAKE_SOURCE_DIR}/core/foundation/inc
                              ${CMAKE_BINARY_DIR}/ginclude   # this is for RConfigure.h
               )
   set_target_properties(emitFromPyTorch PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
   add_custom_target(SofieCompileModels_PyTorch)
   add_dependencies(SofieCompileModels_PyTorch emitFromPyTorch)
   add_custom_command(TARGET SofieCompileModels_PyTorch POST_BUILD
		COMMAND ./emitFromPyTorch
		USES_TERMINAL
	)
   ROOT_ADD_GTEST(TestRModelParserPyTorch TestRModelParserPyTorch.C
    LIBRARIES
    ROOTTMVASofie
    blas
    ${PYTHON_LIBRARIES}
    INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}
   )
   add_dependencies(TestRModelParserPyTorch SofieCompileModels_PyTorch)
endif(PY_TORCH_FOUND)

if((PY_KERAS_FOUND AND PY_THEANO_FOUND) OR (PY_KERAS_FOUND AND PY_TENSORFLOW_FOUND))
   configure_file(generateKerasModelFunctional.py generateKerasModelFunctional.py COPYONLY)
   configure_file(generateKerasModelSequential.py generateKerasModelSequential.py COPYONLY)

   # Test PyKeras: Binary classification
   ROOT_EXECUTABLE(testPyKerasClassification testPyKerasClassification.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Keras-Classification COMMAND testPyKerasClassification)

   # Test PyKeras: Regression
   ROOT_EXECUTABLE(testPyKerasRegression testPyKerasRegression.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Keras-Regression COMMAND testPyKerasRegression)

   # Test PyKeras: Multi-class classification
   ROOT_EXECUTABLE(testPyKerasMulticlass testPyKerasMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Keras-Multiclass COMMAND testPyKerasMulticlass)

   # Test RModelParser_Keras
   add_executable(emitFromKeras
                 EmitFromKeras.cxx
                 ${CMAKE_SOURCE_DIR}/tmva/pymva/src/RModelParser_Keras.cxx
                 ${CMAKE_SOURCE_DIR}/tmva/sofie/src/RModel.cxx
                 ${CMAKE_SOURCE_DIR}/tmva/sofie/src/SOFIE_common.cxx
                 )
   target_link_libraries(emitFromKeras ${PYTHON_LIBRARIES} ${Libraries})
   target_include_directories(emitFromKeras PRIVATE
                 ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
                 ${CMAKE_SOURCE_DIR}/tmva/inc
                 ${CMAKE_SOURCE_DIR}/core/foundation/inc
                 ${CMAKE_BINARY_DIR}/ginclude   # this is for RConfigure.h
               )
   set_target_properties(emitFromKeras PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
   add_custom_target(SofieCompileModels_Keras)
   add_dependencies(SofieCompileModels_Keras emitFromKeras)
   add_custom_command(TARGET SofieCompileModels_Keras POST_BUILD
		COMMAND ./emitFromKeras
		USES_TERMINAL
	)
   ROOT_ADD_GTEST(TestRModelParserKeras TestRModelParserKeras.C
    LIBRARIES
    ROOTTMVASofie
    blas
    ${PYTHON_LIBRARIES}
    INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}
   )
   add_dependencies(TestRModelParserKeras SofieCompileModels_Keras)
endif((PY_KERAS_FOUND AND PY_THEANO_FOUND) OR (PY_KERAS_FOUND AND PY_TENSORFLOW_FOUND))

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
   ROOT_ADD_TEST(PyMVA-GTB-Classification COMMAND testPyGTBClassification)

   # Test PyGTB: Multi-class classification
   ROOT_EXECUTABLE(testPyGTBMulticlass testPyGTBMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-GTB-Multiclass COMMAND testPyGTBMulticlass)

   # Test PyAdaBoost: Classification
   ROOT_EXECUTABLE(testPyAdaBoostClassification testPyAdaBoostClassification.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-AdaBoost-Classification COMMAND testPyAdaBoostClassification)

   # Test PyAdaBoost: Multi-class classification
   ROOT_EXECUTABLE(testPyAdaBoostMulticlass testPyAdaBoostMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-AdaBoost-Multiclass COMMAND testPyAdaBoostMulticlass)
endif(PY_SKLEARN_FOUND)
