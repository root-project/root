# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT bindings/pyroot_experimental package
############################################################################

if(dataframe)
    list(APPEND PYROOT_EXTRA_PYSOURCE
        ROOT/pythonization/_rdataframe.py
        ROOT/pythonization/_rtensor.py)
    list(APPEND PYROOT_EXTRA_SOURCE
        src/RDataFramePyz.cxx
        src/RTensorPyz.cxx)
endif()

set(py_sources
  ROOT/__init__.py
  ROOT/_application.py
  ROOT/_facade.py
  ROOT/pythonization/__init__.py
  ROOT/pythonization/_cppinstance.py
  ROOT/pythonization/_drawables.py
  ROOT/pythonization/_generic.py
  ROOT/pythonization/_rooabscollection.py
  ROOT/pythonization/_roodatahist.py
  ROOT/pythonization/_rvec.py
  ROOT/pythonization/_stl_vector.py
  ROOT/pythonization/_tarray.py
  ROOT/pythonization/_tclonesarray.py
  ROOT/pythonization/_tcollection.py
  ROOT/pythonization/_tcomplex.py
  ROOT/pythonization/_tdirectory.py
  ROOT/pythonization/_tdirectoryfile.py
  ROOT/pythonization/_tfile.py
  ROOT/pythonization/_tgraph.py
  ROOT/pythonization/_th1.py
  ROOT/pythonization/_titer.py
  ROOT/pythonization/_tobject.py
  ROOT/pythonisation/_tobjstring.py
  ROOT/pythonization/_tseqcollection.py
  ROOT/pythonisation/_tstring.py
  ROOT/pythonization/_ttree.py
  ROOT/pythonization/_tvector3.py
  ROOT/pythonization/_tvectort.py
  ${PYROOT_EXTRA_PYSOURCE}
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTPython
  HEADERS
    TPyArg.h
    TPyReturn.h
    TPython.h
  SOURCES
    src/CppCallablePyz.cxx
    src/PyROOTModule.cxx
    src/PyROOTStrings.cxx
    src/PyROOTWrapper.cxx
    src/RPyROOTApplication.cxx
    src/GenericPyz.cxx
    src/RVecPyz.cxx
    src/TClassPyz.cxx
    src/TClonesArrayPyz.cxx
    src/TDirectoryFilePyz.cxx
    src/TDirectoryPyz.cxx
    src/TFilePyz.cxx
    src/TMemoryRegulator.cxx
    src/TPyArg.cxx
    src/TPyClassGenerator.cxx
    src/TPyReturn.cxx
    src/TPython.cxx
    src/TObjectPyz.cxx
    src/TTreePyz.cxx
    src/PyzCppHelpers.cxx
    src/PyzPythonHelpers.cxx
    src/CPPInstancePyz.cxx
    ${PYROOT_EXTRA_SOURCE}
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    Tree
    cppyy
)

set(JupyROOTDirName python/JupyROOT)
add_library(JupyROOT SHARED ${JupyROOTDirName}/src/IOHandler.cxx)
target_include_directories(JupyROOT PRIVATE ${JupyROOTDirName}/src)
target_link_libraries(JupyROOT Core ${CMAKE_DL_LIBS})
set_property(GLOBAL APPEND PROPERTY CMAKENOEXPORT JupyROOT)

#---Install python modules----------------------------------------------------------------------

file(COPY python/ROOT DESTINATION ${localruntimedir})
install(DIRECTORY python/ROOT DESTINATION ${runtimedir})

set(d $ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX}/${runtimedir})
foreach(py_source ${py_sources})
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m py_compile ${d}/${py_source})")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -O -m py_compile ${d}/${py_source})")
endforeach()

file(COPY ${JupyROOTDirName} DESTINATION ${localruntimedir})
install(DIRECTORY ${JupyROOTDirName} DESTINATION ${runtimedir})

# Disables warnings caused by Py_RETURN_TRUE/Py_RETURN_FALSE
if(NOT MSVC)
    target_compile_options(ROOTPython PRIVATE -Wno-strict-aliasing)
endif()

# Disables warnings originating from deprecated register keyword in Python
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_STANDARD GREATER_EQUAL 11)
    target_compile_options(ROOTPython PRIVATE -Wno-register)
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" AND CMAKE_CXX_STANDARD GREATER_EQUAL 11)
    target_compile_options(ROOTPython PRIVATE -Wno-deprecated-register)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
