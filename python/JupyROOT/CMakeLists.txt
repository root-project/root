if (ROOT_pyroot_FOUND AND NOT ROOT_CLASSIC_BUILD) # Do not run with classic build

set(MODULES_LOCATION ${ROOTSYS}/lib/JupyROOT/helpers)
set(NBDIFFUTIL ${CMAKE_CURRENT_SOURCE_DIR}/nbdiff.py )

# List of notebook files
# TODO: To be extended with the list of downloaded notebooks used in the
# documentation and as tutorials.
set(NOTEBOOKS ${CMAKE_CURRENT_SOURCE_DIR}/importROOT.ipynb
              ${CMAKE_CURRENT_SOURCE_DIR}/simpleCppMagic.ipynb)

# Include cpp notebooks if metakernel is present
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" "import metakernel" RESULT_VARIABLE FoundMetakernel ERROR_QUIET)
if(FoundMetakernel EQUAL 0)
  set(NOTEBOOKS ${NOTEBOOKS}
                ${CMAKE_CURRENT_SOURCE_DIR}/thread_local.ipynb
                ${CMAKE_CURRENT_SOURCE_DIR}/ROOT_kernel.ipynb)
endif()

find_python_module(IPython QUIET)

if(PY_IPYTHON_FOUND)
# Test all modules with doctest. All new tests will be automatically picked up
  file(GLOB pyfiles ${MODULES_LOCATION}/*.py)
  foreach(pyfile ${pyfiles})
    get_filename_component(SHORTPYFILE ${pyfile} NAME_WE)
    if (NOT ${SHORTPYFILE} STREQUAL "__init__")
      ROOTTEST_ADD_TEST(${SHORTPYFILE}_doctest
                        COMMAND ${PYTHON_EXECUTABLE} -m doctest ${pyfile})
    endif()
  endforeach()

# Test all notebooks available
  foreach(NOTEBOOK ${NOTEBOOKS})
    get_filename_component(NOTEBOOKBASE ${NOTEBOOK} NAME_WE)
    ROOTTEST_ADD_TEST(${NOTEBOOKBASE}_notebook
                      COMMAND ${PYTHON_EXECUTABLE} ${NBDIFFUTIL} ${NOTEBOOK})
  endforeach()

  if(FoundMetakernel EQUAL 0 AND ROOT_imt_FOUND)
    # No need to compare output here, just check it runs with no error
    ROOTTEST_ADD_TEST(Cpp_IMT_Canvas_notebook
                      COMMAND ${PYTHON_EXECUTABLE} ${NBDIFFUTIL} ${CMAKE_CURRENT_SOURCE_DIR}/Cpp_IMT_Canvas.ipynb "OFF")
  endif()
endif()



endif()
