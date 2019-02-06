############################################################################
# CMakeLists.txt file for PyMVA  package
############################################################################
#Autor: Omar  Zapata, Lorenzo Moneta, Sergei Glayzer 22/08/2015
#website:http://oproject.org


set(libname PyMVA)


include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})


set(PY_HEADERS ${CMAKE_SOURCE_DIR}/tmva/pymva/inc/TMVA/PyMethodBase.h
               ${CMAKE_SOURCE_DIR}/tmva/pymva/inc/TMVA/MethodPyRandomForest.h
               ${CMAKE_SOURCE_DIR}/tmva/pymva/inc/TMVA/MethodPyAdaBoost.h
               ${CMAKE_SOURCE_DIR}/tmva/pymva/inc/TMVA/MethodPyGTB.h
               ${CMAKE_SOURCE_DIR}/tmva/pymva/inc/TMVA/MethodPyKeras.h)

if(ROOT_PYTHON_NO_LINK)
    ROOT_STANDARD_LIBRARY_PACKAGE(PyMVA
                                  HEADERS ${PY_HEADERS}
                                  DEPENDENCIES Core Matrix Thread RIO TMVA)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set_property(TARGET PyMVA APPEND PROPERTY LINK_FLAGS "-undefined dynamic_lookup")
    endif()
else()
    ROOT_STANDARD_LIBRARY_PACKAGE(PyMVA
                                  HEADERS ${PY_HEADERS}
                                  LIBRARIES ${PYTHON_LIBRARIES}
                                  DEPENDENCIES Core Matrix Thread RIO TMVA)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
