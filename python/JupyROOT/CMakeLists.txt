if (ROOT_python_FOUND AND NOT ROOT_CLASSIC_BUILD) # Do not run with classic build

set (MODULES_LOCATION ${ROOTSYS}/lib/JupyROOT/)
set(NBDIFFUTIL ${CMAKE_CURRENT_SOURCE_DIR}/nbdiff.py )

# List of notebook files
# TODO: To be extended with the list of downloaded notebooks used in the
# documentation and as tutorials. In order to spot problems with downloads
# one should not use GLOB but rather list explicitely all the files
set(NOTEBOOKS ${CMAKE_CURRENT_SOURCE_DIR}/importROOT.ipynb
              ${CMAKE_CURRENT_SOURCE_DIR}/simpleCppMagic.ipynb)
#              ${CMAKE_CURRENT_SOURCE_DIR}/ROOT_kernel.ipynb)

find_python_module(IPython QUIET)

if(PY_IPYTHON_FOUND)
# Test all modules with doctest. All new tests will be automatically picked up
  file(GLOB pyfiles ${MODULES_LOCATION}/*.py)
  foreach(pyfile ${pyfiles})
    get_filename_component(SHORTPYFILE ${pyfile} NAME_WE)
    if (NOT ${SHORTPYFILE} STREQUAL "__init__")
      ROOTTEST_ADD_TEST(${SHORTPYFILE}_doctest
                        COMMAND ${PYTHON_EXECUTABLE} -m doctest ${pyfile}
                        RUN_SERIAL)
    endif()
  endforeach()

# Test all notebooks available
  foreach(NOTEBOOK ${NOTEBOOKS})
    get_filename_component(NOTEBOOKBASE ${NOTEBOOK} NAME_WE)
    ROOTTEST_ADD_TEST(${NOTEBOOKBASE}
                      COMMAND ${PYTHON_EXECUTABLE} ${NBDIFFUTIL} ${NOTEBOOK}
                      RUN_SERIAL)
  endforeach()
endif()



endif()
