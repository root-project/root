# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Test library loads during importing ROOT
# Testing only the Linux systems is sufficient to detect unwanted links to libraries at import time.
# Mac (and potentially Windows) pull in many system libraries which makes this test very complex.
if (NOT APPLE AND NOT WIN32)
    ROOT_ADD_PYUNITTEST(pyroot_import_load_libs import_load_libs.py)
endif()

# Test ROOT module
ROOT_ADD_PYUNITTEST(pyroot_root_module root_module.py)

# @pythonization decorator
ROOT_ADD_PYUNITTEST(pyroot_pyz_decorator pythonization_decorator.py)

# General pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_pretty_printing pretty_printing.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_array_interface array_interface.py PYTHON_DEPS numpy)

# STL containers pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_stl_vector stl_vector.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_stl_set stl_set.py)

# TObject and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobject_contains tobject_contains.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobject_comparisonops tobject_comparisonops.py)

# TClass pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tclass_dynamiccast tclass_dynamiccast.py)

# TContext pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcontext_contextmanager tcontext_contextmanager.py)

# TDirectory and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tdirectory_attrsyntax tdirectory_attrsyntax.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tdirectoryfile_attrsyntax_get tdirectoryfile_attrsyntax_get.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tfile_attrsyntax_get_writeobject_open tfile_attrsyntax_get_writeobject_open.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tfile_constructor tfile_constructor.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tfile_context_manager tfile_context_manager.py)

# TTree and subclasses pythonizations
file(COPY TreeHelper.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch_attr ttree_branch_attr.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_iterable ttree_iterable.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_setbranchaddress ttree_setbranchaddress.py PYTHON_DEPS numpy)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch ttree_branch.py PYTHON_DEPS numpy)

# TH1 and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_th1_operators th1_operators.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_th1_fillN th1_fillN.py PYTHON_DEPS numpy)
ROOT_ADD_PYUNITTEST(pyroot_pyz_th2 th2.py)

# TGraph, TGraph2D and error subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tgraph_getters tgraph_getters.py)

# TCollection and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_len tcollection_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_listmethods tcollection_listmethods.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_operators tcollection_operators.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_iterable tcollection_iterable.py)

# TSeqCollection and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tseqcollection_itemaccess tseqcollection_itemaccess.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tseqcollection_listmethods tseqcollection_listmethods.py)

# TIter pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_titer_iterator titer_iterator.py)

# TArray and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tarray_len tarray_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tarray_getitem tarray_getitem.py)

# TVectorT pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvectort_len tvectort_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvectort_getitem tvectort_getitem.py)

# TVector3 pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvector3_len tvector3_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvector3_getitem tvector3_getitem.py)

# TString pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring_len tstring_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring_str_repr tstring_str_repr.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring_comparisonops tstring_comparisonops.py)

# TString converter
ROOT_ADD_PYUNITTEST(pyroot_conv_tstring tstring_converter.py)

# TObjString pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring_len tobjstring_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring_str_repr tobjstring_str_repr.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring_comparisonops tobjstring_comparisonops.py)

# RVec and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_rvec rvec.py)
if(NOT MSVC OR win_broken_tests)
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rvec_asrvec rvec_asrvec.py PYTHON_DEPS numpy)
endif()

# RDataFrame and subclasses pythonizations
if (dataframe)
    if(NOT MSVC OR win_broken_tests)
        ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_makenumpy rdataframe_makenumpy.py PYTHON_DEPS numpy)
    endif()
    if(NOT MSVC OR win_broken_tests OR CMAKE_CXX_STANDARD GREATER 14)
        ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_asnumpy rdataframe_asnumpy.py PYTHON_DEPS numpy)
        ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_histo_profile rdataframe_histo_profile.py)
    endif()
endif()

# RDFDescription pythonization
if (dataframe)
    ROOT_ADD_PYUNITTEST(pyroot_rdfdescription rdfdescription.py)
endif()

# SOFIE-GNN pythonizations
if (tmva)
    if(NOT MSVC OR CMAKE_SIZEOF_VOID_P EQUAL 4 OR win_broken_tests)
        ROOT_FIND_PYTHON_MODULE(sonnet)
        ROOT_FIND_PYTHON_MODULE(graph_nets)
        if (ROOT_SONNET_FOUND AND ROOT_GRAPH_NETS_FOUND)
            ROOT_ADD_PYUNITTEST(pyroot_pyz_sofie_gnn sofie_gnn.py PYTHON_DEPS numpy sonnet graph_nets)
        endif()
    endif()
endif()

# RTensor pythonizations
if (tmva AND dataframe)
    if(NOT MSVC OR CMAKE_SIZEOF_VOID_P EQUAL 4 OR win_broken_tests)
        ROOT_ADD_PYUNITTEST(pyroot_pyz_rtensor rtensor.py PYTHON_DEPS numpy)
    endif()
endif()

# Passing Python callables to ROOT.TF
ROOT_ADD_PYUNITTEST(pyroot_pyz_tf_pycallables tf_pycallables.py)

if(roofit)

  ROOT_ADD_PYUNITTEST(pyroot_roofit_rooabspdf_fitto roofit/rooabspdf_fitto.py)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_rooabsreal_ploton roofit/rooabsreal_ploton.py)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_rooarglist roofit/rooarglist.py)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_roocmdarg roofit/roocmdarg.py)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_roodatahist_numpy roofit/roodatahist_numpy.py PYTHON_DEPS numpy)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_roodatahist_ploton roofit/roodatahist_ploton.py)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_roodataset roofit/roodataset.py)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_roodataset_numpy roofit/roodataset_numpy.py PYTHON_DEPS numpy)
  ROOT_ADD_PYUNITTEST(pyroot_roofit_roolinkedlist roofit/roolinkedlist.py)

  if(NOT MSVC OR CMAKE_SIZEOF_VOID_P EQUAL 4 OR win_broken_tests)
    ROOT_ADD_PYUNITTEST(pyroot_roofit_rooabscollection roofit/rooabscollection.py)
  endif()

  if(NOT MSVC OR win_broken_tests)
    # Test pythonizations for the RooFitHS3 package, which is not built on Windows.
    ROOT_ADD_PYUNITTEST(pyroot_roofit_roojsonfactorywstool roofit/roojsonfactorywstool.py)

    # Other pythonizations that fail on Windows for unknown reasons
    ROOT_ADD_PYUNITTEST(pyroot_roofit_rooglobalfunc roofit/rooglobalfunc.py)
    ROOT_ADD_PYUNITTEST(pyroot_roofit_roosimultaneous roofit/roosimultaneous.py)

    # RooWorkspace pythonization that fails on Windows
    ROOT_ADD_PYUNITTEST(pyroot_roofit_rooworkspace roofit/rooworkspace.py)
  endif()

endif()

if (dataframe)
    # std::string_view in CPyCppyy
    ROOT_ADD_PYUNITTEST(pyroot_string_view string_view.py)
    if(NOT MSVC OR win_broken_tests)
        if(NOT DEFINED ENV{ROOTTEST_IGNORE_NUMBA_PY3})
            # Test wrapping Python callables for use in C++ using numba
            ROOT_ADD_PYUNITTEST(pyroot_numbadeclare numbadeclare.py PYTHON_DEPS numba)
            ROOT_ADD_PYUNITTEST(pyroot_rdf_filter_pyz rdf_filter_pyz.py PYTHON_DEPS numba)
            ROOT_ADD_PYUNITTEST(pyroot_rdf_define_pyz rdf_define_pyz.py PYTHON_DEPS numba)
        endif()
    endif()
endif()

# TComplex pythonizations
ROOT_ADD_PYUNITTEST(pyroot_tcomplex tcomplex_operators.py)

# Tests with memory usage
ROOT_ADD_PYUNITTEST(pyroot_memory memory.py)

# rbatchgenerator tests
# TODO: We currently do not support TensorFlow for Python >= 3.12 (see requirements.txt)
# Update here once that is fixed.
if (NOT MSVC AND Python3_VERSION VERSION_LESS 3.12)
  ROOT_ADD_PYUNITTEST(batchgen rbatchgenerator_completeness.py PYTHON_DEPS numpy tensorflow torch)
endif()
