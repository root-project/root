#disable until JupyROOT works on Windows
if(NOT MSVC OR win_broken_tests)

# Do not run with classic build or if tests are vetoed
if (ROOT_pyroot_FOUND AND NOT ROOT_CLASSIC_BUILD)

set(MODULES_LOCATION ${ROOTSYS}/lib/JupyROOT/helpers)
set(NBDIFFUTIL ${CMAKE_CURRENT_SOURCE_DIR}/nbdiff.py )
set(DOCTEST_LAUNCHER ${CMAKE_CURRENT_SOURCE_DIR}/doctest_launcher.py)

# List of notebook files
# TODO: To be extended with the list of downloaded notebooks used in the
# documentation and as tutorials.
set(NOTEBOOKS importROOT.ipynb
              simpleCppMagic.ipynb
              thread_local.ipynb
              ROOT_kernel.ipynb
              tpython.ipynb)

# Test all modules with doctest. All new tests will be automatically picked up
file(GLOB pyfiles ${MODULES_LOCATION}/*.py)
foreach(pyfile ${pyfiles})
  get_filename_component(SHORTPYFILE ${pyfile} NAME_WE)
  if (NOT ${SHORTPYFILE} STREQUAL "__init__")
    ROOTTEST_ADD_TEST(${SHORTPYFILE}_doctest
                      COMMAND ${Python3_EXECUTABLE} ${DOCTEST_LAUNCHER} ${pyfile}
                      PYTHON_DEPS IPython)
  endif()
endforeach()

# Test all notebooks available
foreach(NOTEBOOK ${NOTEBOOKS})
  get_filename_component(NOTEBOOKBASE ${NOTEBOOK} NAME_WE)
  ROOTTEST_ADD_TEST(${NOTEBOOKBASE}_notebook
                    COPY_TO_BUILDDIR ${NOTEBOOK}
                    COMMAND ${Python3_EXECUTABLE} ${NBDIFFUTIL} ${NOTEBOOK}
                    RUN_SERIAL
                    PYTHON_DEPS jupyter)
endforeach()

if(ROOT_imt_FOUND)
  # No need to compare output here, just check it runs with no error
  set(IMT_NB Cpp_IMT_Canvas.ipynb)
  get_filename_component(NOTEBOOKBASE ${IMT_NB} NAME_WE)
  ROOTTEST_ADD_TEST(${NOTEBOOKBASE}_notebook
                    COPY_TO_BUILDDIR ${IMT_NB}
                    COMMAND ${Python3_EXECUTABLE} ${NBDIFFUTIL} ${IMT_NB} "OFF"
                    RUN_SERIAL
                    PYTHON_DEPS jupyter)
endif()

endif()

endif()
