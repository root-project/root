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
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

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
   # Test PyTorch: Binary classification${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS}

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
   add_executable(emitFromPyTorch
                  EmitFromPyTorch.cxx
                 )
   target_link_libraries(emitFromPyTorch ${PYTHON_LIBRARIES} ${Libraries})
if(APPLE)
   target_link_options(emitFromPyTorch PRIVATE ${PYTHON_LINK_OPTIONS_Development_Main})
endif()

   target_include_directories(emitFromPyTorch PRIVATE
                              ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
                              ${CMAKE_SOURCE_DIR}/tmva/inc
               )
   set_target_properties(emitFromPyTorch PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
   add_custom_target(SofieCompileModels_PyTorch)
   add_dependencies(SofieCompileModels_PyTorch emitFromPyTorch)
   if(MSVC)
      add_custom_command(TARGET SofieCompileModels_PyTorch POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E env "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}" $<CONFIG>/emitFromPyTorch.exe
         USES_TERMINAL
      )
   else()
      add_custom_command(TARGET SofieCompileModels_PyTorch POST_BUILD
         COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}" ./emitFromPyTorch
         USES_TERMINAL
      )
      ROOT_ADD_GTEST(TestRModelParserPyTorch TestRModelParserPyTorch.C
         LIBRARIES
         ROOTTMVASofie
         blas
         ${PYTHON_LIBRARIES}
         INCLUDE_DIRS
         SYSTEM
         ${PYTHON_INCLUDE_DIRS_Development_Main}
         ${NUMPY_INCLUDE_DIRS}
         ${CMAKE_CURRENT_BINARY_DIR}
      )
      if(APPLE)
      target_link_options(TestRModelParserPyTorch PRIVATE ${PYTHON_LINK_OPTIONS_Development_Main})
      endif()
      add_dependencies(TestRModelParserPyTorch SofieCompileModels_PyTorch)
   endif()
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
   ROOT_EXECUTABLE(testPyKerasRegression testPyKerasRegression.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Keras-Regression COMMAND testPyKerasRegression DEPENDS ${PyMVA-Keras-Regression-depends})


   # Test PyKeras: Multi-class classification
   ROOT_EXECUTABLE(testPyKerasMulticlass testPyKerasMulticlass.C
      LIBRARIES ${Libraries})
   ROOT_ADD_TEST(PyMVA-Keras-Multiclass COMMAND testPyKerasMulticlass DEPENDS ${PyMVA-Keras-Multiclass-depends})

   # Test RModelParser_Keras
   add_executable(emitFromKeras
                 EmitFromKeras.cxx
                 )
   target_link_libraries(emitFromKeras ${PYTHON_LIBRARIES} ${Libraries})
if(APPLE)
   target_link_options(emitFromKeras PRIVATE ${PYTHON_LINK_OPTIONS_Development_Main})
endif()

   target_include_directories(emitFromKeras PRIVATE
                 ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
                 ${CMAKE_SOURCE_DIR}/tmva/inc
               )
   set_target_properties(emitFromKeras PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
   add_custom_target(SofieCompileModels_Keras)
   add_dependencies(SofieCompileModels_Keras emitFromKeras)
   add_custom_command(TARGET SofieCompileModels_Keras POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}" ./emitFromKeras
		USES_TERMINAL
	)

   # Testing SOFIE Custom Operator functionality
   add_executable(emitCustomModel
                  EmitCustomModel.cxx
                  ${CMAKE_SOURCE_DIR}/tmva/pymva/src/RModelParser_Keras.cxx
                  ${CMAKE_SOURCE_DIR}/tmva/sofie/src/RModel.cxx
                  ${CMAKE_SOURCE_DIR}/tmva/sofie/src/SOFIE_common.cxx
   )
   target_link_libraries(emitCustomModel ${PYTHON_LIBRARIES} ${Libraries})
   if(APPLE)
      target_link_options(emitCustomModel PRIVATE ${PYTHON_LINK_OPTIONS_Development_Main})
   endif()
   target_include_directories(emitCustomModel PRIVATE
                              ${CMAKE_SOURCE_DIR}/tmva/sofie/inc
                              ${CMAKE_SOURCE_DIR}/tmva/inc
                              ${CMAKE_SOURCE_DIR}/core/foundation/inc
                              ${CMAKE_BINARY_DIR}/ginclude   # this is for RConfigure.h
   )
   set_target_properties(emitCustomModel PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
   add_custom_target(SofieCustomModel)
   add_dependencies(SofieCustomModel emitCustomModel)
   add_custom_command(TARGET SofieCustomModel POST_BUILD
      COMMAND ./emitCustomModel
      USES_TERMINAL
   )

   ROOT_ADD_GTEST(TestRModelParserKeras TestRModelParserKeras.C
    LIBRARIES
    ROOTTMVASofie
    blas
    ${PYTHON_LIBRARIES}
    INCLUDE_DIRS
    SYSTEM
    ${PYTHON_INCLUDE_DIRS_Development_Main}
    ${NUMPY_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
   )
   if(APPLE)
   target_link_options(TestRModelParserKeras PRIVATE ${PYTHON_LINK_OPTIONS_Development_Main})
   endif()
   add_dependencies(TestRModelParserKeras SofieCompileModels_Keras SofieCustomModel)

endif((PY_KERAS_FOUND AND PY_THEANO_FOUND) OR (PY_KERAS_FOUND AND PY_TENSORFLOW_FOUND))
