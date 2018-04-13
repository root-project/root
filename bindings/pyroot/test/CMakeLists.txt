ROOT_ADD_PYUNITTEST(pyroot_conversions conversions.py)
ROOT_ADD_PYUNITTEST(pyroot_list_initialization list_initialization.py)
if(NUMPY_FOUND)
    ROOT_ADD_PYUNITTEST(pyroot_array_interface array_interface.py)
    ROOT_ADD_PYUNITTEST(pyroot_ttree_asmatrix ttree_asmatrix.py)
endif()

