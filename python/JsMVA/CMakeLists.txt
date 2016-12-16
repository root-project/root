if (ROOT_python_FOUND AND NOT ROOT_CLASSIC_BUILD) # Do not run with classic build

set(NBDIFFUTIL ${CMAKE_CURRENT_SOURCE_DIR}/nbdiff.py )

# List of notebook files
set(NOTEBOOKS EnableJsMVA.ipynb
	            UserInterface.ipynb
	            NewMethods.ipynb)

find_python_module(IPython QUIET)

if(PY_IPYTHON_FOUND)
  # Test all notebooks available
  foreach(NOTEBOOK ${NOTEBOOKS})
    get_filename_component(NOTEBOOKBASE ${NOTEBOOK} NAME_WE)
    ROOTTEST_ADD_TEST(${NOTEBOOKBASE}
                      COPY_TO_BUILDDIR ${NOTEBOOK}
                      COMMAND ${PYTHON_EXECUTABLE} ${NBDIFFUTIL} ${NOTEBOOK}
                      RUN_SERIAL)
  endforeach()
endif()

endif()
