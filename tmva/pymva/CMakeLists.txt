############################################################################
# CMakeLists.txt file for PyMVA  package
############################################################################
#Autor: Omar  Zapata, Lorenzo Moneta, Sergei Glayzer 22/08/2015
#website:http://oproject.org


set(libname PyMVA)


include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})


set(PY_HEADERS TMVA/PyMethodBase.h
               TMVA/MethodPyRandomForest.h
               TMVA/MethodPyAdaBoost.h
               TMVA/MethodPyGTB.h
               TMVA/MethodPyKeras.h)

ROOT_STANDARD_LIBRARY_PACKAGE(PyMVA
                              HEADERS ${PY_HEADERS}
                              LIBRARIES ${PYTHON_LIBRARIES}
                              DEPENDENCIES Core Matrix Thread RIO TMVA)

ROOT_ADD_TEST_SUBDIRECTORY(test)
