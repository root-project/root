#if (ROOT_pyroot_FOUND AND NOT ROOT_CLASSIC_BUILD) # Do not run with classic build

#set(NBDIFFUTIL ${CMAKE_CURRENT_SOURCE_DIR}/nbdiff.py )

# List of notebook files
#set(NOTEBOOKS ${CMAKE_CURRENT_SOURCE_DIR}/EnableJsMVA.ipynb
#              ${CMAKE_CURRENT_SOURCE_DIR}/UserInterface.ipynb
#              ${CMAKE_CURRENT_SOURCE_DIR}/NewMethods.ipynb)

#find_python_module(IPython QUIET)

#if(PY_IPYTHON_FOUND)
#  # Test all notebooks available
#  foreach(NOTEBOOK ${NOTEBOOKS})
#    get_filename_component(NOTEBOOKBASE ${NOTEBOOK} NAME_WE)
#    ROOTTEST_ADD_TEST(${NOTEBOOKBASE}
#                      COMMAND ${Python3_EXECUTABLE} ${NBDIFFUTIL} ${NOTEBOOK}
#                      RUN_SERIAL)
#  endforeach()
#endif()

#endif()
