############################################################################
# CMakeLists.txt file for building PyMVA tests
# @author Stefan Wunsch
############################################################################

project(pymva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore TMVA PyMVA)
include_directories(${ROOT_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

# Test PyKeras: Classification
ROOT_EXECUTABLE(testPyKerasClassification testPyKerasClassification.C
   LIBRARIES ${Libraries})
ROOT_ADD_TEST(PyMVA-Keras-Classification COMMAND testPyKerasClassification)

# Test PyKeras: Regression
ROOT_EXECUTABLE(testPyKerasRegression testPyKerasRegression.C
   LIBRARIES ${Libraries})
ROOT_ADD_TEST(PyMVA-Keras-Regression COMMAND testPyKerasRegression)
