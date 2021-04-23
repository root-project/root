#disable until JupyROOT works on Windows
if(NOT MSVC OR win_broken_tests)

# Do not run with classic build or if tests are vetoed
if (ROOT_pyroot_FOUND AND
    NOT ROOT_CLASSIC_BUILD AND
    NOT (DEFINED ENV{ROOTTEST_IGNORE_JUPYTER_PY2} AND PYTHON_VERSION_MAJOR_Development_Main EQUAL 2) AND
    NOT (DEFINED ENV{ROOTTEST_IGNORE_JUPYTER_PY3} AND PYTHON_VERSION_MAJOR_Development_Main EQUAL 3))

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
                      COMMAND ${PYTHON_EXECUTABLE} ${DOCTEST_LAUNCHER} ${pyfile})
  endif()
endforeach()

# Test all notebooks available
foreach(NOTEBOOK ${NOTEBOOKS})
  get_filename_component(NOTEBOOKBASE ${NOTEBOOK} NAME_WE)
  ROOTTEST_ADD_TEST(${NOTEBOOKBASE}_notebook
                    COPY_TO_BUILDDIR ${NOTEBOOK}
                    COMMAND ${PYTHON_EXECUTABLE} ${NBDIFFUTIL} ${NOTEBOOK}
                    RUN_SERIAL)
endforeach()

if(ROOT_imt_FOUND)
  # No need to compare output here, just check it runs with no error
  set(IMT_NB Cpp_IMT_Canvas.ipynb)
  get_filename_component(NOTEBOOKBASE ${IMT_NB} NAME_WE)
  ROOTTEST_ADD_TEST(${NOTEBOOKBASE}_notebook
                    COPY_TO_BUILDDIR ${IMT_NB}
                    COMMAND ${PYTHON_EXECUTABLE} ${NBDIFFUTIL} ${IMT_NB} "OFF"
                    RUN_SERIAL)
endif()

endif()

endif()
