# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

list(LENGTH python_executables len)
math(EXPR how_many_pythons "${len} - 1")

foreach(val RANGE ${how_many_pythons})
  list(GET python_executables ${val} python_executable)
  list(GET python_version_majors ${val} py_maj)
  list(GET numpys_found ${val} numpy_found)

  # Test ROOT module
  ROOT_ADD_PYUNITTEST(pyroot_root_module_python${py_maj} root_module.py PY_EX ${python_executable})

  # General pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_pretty_printing_python${py_maj} pretty_printing.py)
  ROOT_ADD_PYUNITTEST(pyroot_pyz_array_interface_python${py_maj} array_interface.py PY_EX ${python_executable} DEPENDENCIES_FOUND ${numpy_found})

  # TObject and subclasses pythonisations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tobject_contains_python${py_maj} tobject_contains.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tobject_comparisonops_python${py_maj} tobject_comparisonops.py PY_EX ${python_executable})

  # TClass pythonisations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tclass_dynamiccast_python${py_maj} tclass_dynamiccast.py PY_EX ${python_executable})

  # TDirectory and subclasses pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tdirectory_attrsyntax_python${py_maj} tdirectory_attrsyntax.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tdirectoryfile_attrsyntax_get_python${py_maj} tdirectoryfile_attrsyntax_get.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tfile_attrsyntax_get_writeobject_open_python${py_maj} tfile_attrsyntax_get_writeobject_open.py PY_EX ${python_executable})

  # TTree and subclasses pythonizations
  file(COPY TreeHelper.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch_attr_python${py_maj} ttree_branch_attr.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_iterable_python${py_maj} ttree_iterable.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_setbranchaddress_python${py_maj} ttree_setbranchaddress.py PY_EX ${python_executable} DEPENDENCIES_FOUND ${numpy_found})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch_python${py_maj} ttree_branch.py PY_EX ${python_executable} DEPENDENCIES_FOUND ${numpy_found})
  if (dataframe)
    ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_asmatrix_python${py_maj} ttree_asmatrix.py PY_EX ${python_executable} DEPENDENCIES_FOUND ${numpy_found})
  endif()

  # TH1 and subclasses pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_th1_operators_python${py_maj} th1_operators.py PY_EX ${python_executable})

  # TGraph, TGraph2D and error subclasses pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tgraph_getters_python${py_maj} tgraph_getters.py PY_EX ${python_executable})

  # TCollection and subclasses pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_len_python${py_maj} tcollection_len.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_listmethods_python${py_maj} tcollection_listmethods.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_operators_python${py_maj} tcollection_operators.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_iterable_python${py_maj} tcollection_iterable.py PY_EX ${python_executable})

  # TSeqCollection and subclasses pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tseqcollection_itemaccess_python${py_maj} tseqcollection_itemaccess.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tseqcollection_listmethods_python${py_maj} tseqcollection_listmethods.py PY_EX ${python_executable})

  # TIter pythonisations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_titer_iterator_python${py_maj} titer_iterator.py PY_EX ${python_executable})

  # TClonesArray pythonisations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tclonesarray_itemaccess_python${py_maj} tclonesarray_itemaccess.py PY_EX ${python_executable})

  # TArray and subclasses pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tarray_len_python${py_maj} tarray_len.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tarray_getitem_python${py_maj} tarray_getitem.py PY_EX ${python_executable})

  # TVectorT pythonisations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tvectort_len_python${py_maj} tvectort_len.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tvectort_getitem_python${py_maj} tvectort_getitem.py PY_EX ${python_executable})

  # TVector3 pythonisations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tvector3_len_python${py_maj} tvector3_len.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tvector3_getitem_python${py_maj} tvector3_getitem.py PY_EX ${python_executable})

  # TString pythonisations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring_len_python${py_maj} tstring_len.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring_str_repr_python${py_maj} tstring_str_repr.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring_comparisonops_python${py_maj} tstring_comparisonops.py PY_EX ${python_executable})

  # TString converter
  ROOT_ADD_PYUNITTEST(pyroot_conv_tstring_python${py_maj} tstring_converter.py PY_EX ${python_executable})

  # TObjString pythonisations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring_len_python${py_maj} tobjstring_len.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring_str_repr_python${py_maj} tobjstring_str_repr.py PY_EX ${python_executable})
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring_comparisonops_python${py_maj} tobjstring_comparisonops.py PY_EX ${python_executable})

  # RVec and subclasses pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_rvec_asrvec_python${py_maj} rvec_asrvec.py PY_EX ${python_executable} DEPENDENCIES_FOUND ${numpy_found})

  # RDataFrame and subclasses pythonizations
  if (dataframe)
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_asnumpy_python${py_maj} rdataframe_asnumpy.py PY_EX ${python_executable} DEPENDENCIES_FOUND ${numpy_found})
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_makenumpy_python${py_maj} rdataframe_makenumpy.py PY_EX ${python_executable} DEPENDENCIES_FOUND ${numpy_found})
  endif()

  # RTensor pythonizations
  if (dataframe)
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rtensor_python${py_maj} rtensor.py PY_EX ${python_executable} DEPENDENCIES_FOUND ${numpy_found})
  endif()

  # Passing Python callables to ROOT.TF
  ROOT_ADD_PYUNITTEST(pyroot_pyz_tf_pycallables_python${py_maj} tf_pycallables.py PY_EX ${python_executable})

  if(roofit)
    # RooAbsCollection and subclasses pythonizations
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rooabscollection_len_python${py_maj} rooabscollection_len.py PY_EX ${python_executable})

    # RooDataHist pythonisations
    ROOT_ADD_PYUNITTEST(pyroot_pyz_roodatahist_ploton_python${py_maj} roodatahist_ploton.py PY_EX ${python_executable})
  endif()

  # std::string_view backport in CPyCppyy
  if (dataframe)
    ROOT_ADD_PYUNITTEST(pyroot_string_view_backport_python${py_maj} string_view_backport.py PY_EX ${python_executable})
  endif()

  # Test wrapping Python callables for use in C++
  if (dataframe)
    ROOT_ADD_PYUNITTEST(pyroot_cppcallable_python${py_maj} cppcallable.py PY_EX ${python_executable})
  endif()

  # TComplex pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_tcomplex_python${py_maj} tcomplex_operators.py PY_EX ${python_executable})

endforeach()
