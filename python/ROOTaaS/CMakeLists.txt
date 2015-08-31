if (ROOT_python_FOUND AND NOT ROOT_CLASSIC_BUILD) # Do not run with classic build

set (MODULES_LOCATION ${ROOTSYS}/lib/ROOTaaS/iPyROOT/)

# Disable until ipython is part of the externals
#file(GLOB pyfiles ${MODULES_LOCATION}/*.py)
#foreach(pyfile ${pyfiles})
#  get_filename_component(SHORTPYFILE ${pyfile} NAME_WE)
#  if (NOT ${SHORTPYFILE} STREQUAL "__init__")
#    ROOTTEST_ADD_TEST(roottest-python-ROOTaaS-${SHORTPYFILE}_doctest
#                      COMMAND python -m doctest ${pyfile})
#  endif()
#endforeach()

endif()
