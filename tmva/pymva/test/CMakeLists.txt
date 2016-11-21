############################################################################
# CMakeLists.txt file for building PyMVA tests
# @author Stefan Wunsch
############################################################################

project(pymva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore TMVA PyMVA)
include_directories(${ROOT_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

# Look for needed python modules
find_python_module(keras QUIET)
find_python_module(theano QUIET)
find_python_module(tensorflow QUIET)
find_python_module(sklearn QUIET)

# Enable tests based on available python modules
if((PY_KERAS_FOUND AND PY_THEANO_FOUND) OR (PY_KERAS_FOUND AND PY_TENSORFLOW_FOUND))
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
endif(PY_SKLEARN_FOUND)
