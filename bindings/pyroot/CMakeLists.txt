############################################################################
# CMakeLists.txt file for building ROOT bindings/pyroot package
############################################################################

set(PYROOT_HEADERS inc/TPyFitFunction.h    inc/TPython.h
                   inc/TPyArg.h            inc/TPyROOTApplication.h
                   inc/TTreeAsFlatMatrix.h inc/TPyDispatcher.h
                   inc/TPyReturn.h         inc/TPyException.h
                   inc/TPySelector.h)

set(PYROOT_PKG_DEPENDENCIES Core MathCore Net Rint Tree ROOTVecOps ROOTDataFrame)

if( CMAKE_SIZEOF_VOID_P EQUAL 4 ) # With this we exclude ROOTDataFrame on 32 bits builds
  list(REMOVE_ITEM PYROOT_HEADERS inc/TTreeAsFlatMatrix.h)
  list(REMOVE_ITEM PYROOT_PKG_DEPENDENCIES ROOTDataFrame)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(PyROOT
                              NO_INSTALL_HEADERS
                              HEADERS ${PYROOT_HEADERS}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES Core MathCore Net Rint Tree
                              DEPENDENCIES ${PYROOT_PKG_DEPENDENCIES})
ROOT_LINKER_LIBRARY(JupyROOT ../JupyROOT/src/*.cxx DEPENDENCIES Core CMAKENOEXPORT)

ROOT_ADD_CXX_FLAG(_PyROOT_FLAGS -fno-strict-aliasing)
ROOT_ADD_CXX_FLAG(_PyROOT_FLAGS -Wno-parentheses-equality)
ROOT_ADD_CXX_FLAG(_PyROOT_FLAGS -Wno-cast-function-type)
ROOT_ADD_CXX_FLAG(_PyROOT_FLAGS -Wno-deprecated-register)

separate_arguments(_PyROOT_FLAGS)

target_compile_options(PyROOT PRIVATE ${_PyROOT_FLAGS})
target_link_libraries(PyROOT PRIVATE ${PYTHON_LIBRARIES})
target_include_directories(PyROOT PRIVATE ${PYTHON_INCLUDE_DIRS})

if(MSVC)
  if(${PYTHON_VERSION_MAJOR} GREATER_EQUAL 3)
    set(init_symbol PyInit_libPyROOT)
  else()
    set(init_symbol initlibPyROOT)
  endif()
  add_custom_command(TARGET PyROOT POST_BUILD
                     COMMAND link -dll -nologo -IGNORE:4001 -machine:ix86 -export:${init_symbol}
                     $<TARGET_LINKER_FILE:PyROOT> msvcrt.lib
                     -out:$<TARGET_FILE_DIR:PyROOT>/libPyROOT.pyd)
  install(FILES  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libPyROOT.pyd
                 DESTINATION ${CMAKE_INSTALL_BINDIR}
                 COMPONENT libraries)
endif()

#---Install python modules--------------------------------------------------
file(GLOB pyfiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)
foreach(pyfile ${pyfiles})
  install(FILES ${pyfile} DESTINATION ${runtimedir})
  if(IS_ABSOLUTE ${runtimedir})
    set(absruntimedir ${runtimedir})
  else()
    set(absruntimedir \${CMAKE_INSTALL_PREFIX}/${runtimedir})
  endif()
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m py_compile \$ENV{DESTDIR}${absruntimedir}/${pyfile})")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -O -m py_compile \$ENV{DESTDIR}${absruntimedir}/${pyfile})")
  file(COPY ${pyfile} DESTINATION ${localruntimedir})
endforeach()

set(JupyROOTDirName "JupyROOT")
install (DIRECTORY ${JupyROOTDirName} DESTINATION ${runtimedir})
file(COPY ${JupyROOTDirName} DESTINATION ${localruntimedir})

set( JsMVADirName "JsMVA")
install (DIRECTORY ${JsMVADirName} DESTINATION ${runtimedir})
file(COPY ${JsMVADirName} DESTINATION ${localruntimedir})

ROOT_INSTALL_HEADERS()

ROOT_ADD_TEST_SUBDIRECTORY(test)
