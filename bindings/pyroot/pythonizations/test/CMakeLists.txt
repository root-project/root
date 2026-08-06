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
ROOT_ADD_PYUNITTEST(pyroot_root_no_implicit_cppyy root_no_implicit_cppyy.py)

# General pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_pretty_printing pretty_printing.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_array_interface array_interface.py PYTHON_DEPS numpy)
ROOT_ADD_PYUNITTEST(uhi_plotting uhi_plotting.py GENERIC PYTHON_DEPS uhi numpy)
ROOT_ADD_PYUNITTEST(uhi_indexing uhi_indexing.py GENERIC PYTHON_DEPS uhi numpy)
ROOT_ADD_PYUNITTEST(uhi_serialization_json uhi_serialization_json.py GENERIC PYTHON_DEPS uhi numpy)
if(Python3_VERSION VERSION_GREATER_EQUAL 3.9)
    # Enable uhi testing helper for Python 3.9+
    ROOT_ADD_PYUNITTEST(uhi_test_suite uhi_test_suite.py GENERIC PYTHON_DEPS uhi numpy)
endif()

# STL containers pythonizations (std::vector and std::set)
ROOT_ADD_PYUNITTEST(pyroot_pyz_stl stl.py)

# TObject and subclasses pythonizations (__contains__ and comparison operators)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobject tobject.py)

# TDirectory and TFile pythonizations (attr/item syntax, Get, WriteObject, Open,
# constructor and context manager)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tdirectory_tfile tdirectory_tfile.py)

# TTree and subclasses pythonizations (branch attr access, iteration and
# SetBranchAddress). ttree_branch stays separate: it declares its own MyStruct,
# which would clash with the TreeHelper.h definition used by the others.
file(COPY TreeHelper.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree ttree.py PYTHON_DEPS numpy)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch ttree_branch.py PYTHON_DEPS numpy)

# TColor-related pythonizations (regression_20018 lives in tcolor.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcolor tcolor.py)

# TH1 and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_th1 th1.py)
# The above tests a deadlock. It should complete in about 1s. If we don't reduce the timeout, we need to wait 1500 s.
set_tests_properties(pyunittests-bindings-pyroot-pythonizations-pyroot-pyz-th1 PROPERTIES TIMEOUT 30)
ROOT_ADD_PYUNITTEST(pyroot_pyz_th1_fillN th1_fillN.py PYTHON_DEPS numpy)
ROOT_ADD_PYUNITTEST(pyroot_pyz_th2_fillN th2_fillN.py PYTHON_DEPS numpy)
# th1_addfunction, th2 and th3 merged into thist.py
ROOT_ADD_PYUNITTEST(pyroot_pyz_thist thist.py)

# TCollection and subclasses pythonizations (len, list methods, operators, iteration)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection tcollection.py)

# TSeqCollection and subclasses pythonizations (item access and list methods)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tseqcollection tseqcollection.py)

# TArray and subclasses pythonizations (len and item access)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tarray tarray.py)

# TVectorT pythonizations (len and item access)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvectort tvectort.py)

# TVector3 pythonizations (len and item access)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvector3 tvector3.py)

# TString pythonizations (len, str/repr, comparison operators and converter)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring tstring.py)

# TObjString pythonizations (len, str/repr and comparison operators)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring tobjstring.py)

# @pythonization decorator (kept standalone: it inspects the global set of
# pythonized classes, so it must not share a process with other tests)
ROOT_ADD_PYUNITTEST(pyroot_pyz_decorator pythonization_decorator.py)

# Miscellaneous small pythonizations grouped in a single file to avoid paying the
# Python-startup + import-ROOT overhead once per tiny test: TClass::DynamicCast,
# TContext context manager, TComplex operators, TIter iteration and TGraph getters.
ROOT_ADD_PYUNITTEST(pyroot_pyz_misc misc_pyz.py)

# RVec and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_rvec rvec.py)
if(NOT MSVC OR win_broken_tests)
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rvec_asrvec rvec_asrvec.py PYTHON_DEPS numpy)
endif()

# RDataFrame and subclasses pythonizations
if (dataframe)
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_makenumpy rdataframe_makenumpy.py PYTHON_DEPS numpy)
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_asnumpy rdataframe_asnumpy.py PYTHON_DEPS numpy)
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_histo_profile rdataframe_histo_profile.py)
endif()

# RDFDescription pythonization
if (dataframe)
    ROOT_ADD_PYUNITTEST(pyroot_rdfdescription rdfdescription.py)

    ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_misc rdataframe_misc.py PYTHON_DEPS numpy)
endif()

# SOFIE-GNN pythonizations
if (tmva)
    ROOT_FIND_PYTHON_MODULE(sonnet)
    ROOT_FIND_PYTHON_MODULE(graph_nets)
    if (ROOT_SONNET_FOUND AND ROOT_GRAPH_NETS_FOUND)
        ROOT_ADD_PYUNITTEST(pyroot_pyz_sofie_gnn sofie_gnn.py PYTHON_DEPS numpy sonnet graph_nets)
    endif()
endif()

# RTensor pythonizations
if (tmva AND dataframe)
    ROOT_ADD_PYUNITTEST(pyroot_pyz_rtensor rtensor.py PYTHON_DEPS numpy)
endif()

# Passing Python callables to ROOT.TF
ROOT_ADD_PYUNITTEST(pyroot_pyz_tf_pycallables tf_pycallables.py PYTHON_DEPS numpy)

if(roofit)
  ROOT_ADD_PYUNITTEST(pyroot_roofit roofit.py)
endif()

if (dataframe)
    if(NOT MSVC OR win_broken_tests)
        # Test wrapping Python callables for use in C++ using numba
        ROOT_ADD_PYUNITTEST(pyroot_numbadeclare numbadeclare.py PYTHON_DEPS numba)
        ROOT_ADD_PYUNITTEST(pyroot_rdf_filter_pyz rdf_filter_pyz.py PYTHON_DEPS numba)
        ROOT_ADD_PYUNITTEST(pyroot_rdf_define_pyz rdf_define_pyz.py PYTHON_DEPS numba)
        # std::string_view in CPyCppyy
        ROOT_ADD_PYUNITTEST(pyroot_string_view string_view.py)
    endif()
endif()

# Tests with memory usage
ROOT_ADD_PYUNITTEST(pyroot_memory memory.py)

# ML dataloader tests
if (dataframe)
  ROOT_ADD_PYUNITTEST(ml_dataloader ml_dataloader.py PYTHON_DEPS numpy tensorflow torch)
endif()

ROOT_ADD_PYUNITTEST(regression_18441 regression_18441.py)

ROOT_ADD_PYUNITTEST(regression_pickle_no_cppyy regression_pickle_no_cppyy.py)

ROOT_ADD_PYUNITTEST(package_metadata package_metadata.py)

if(NOT MSVC OR CMAKE_SIZEOF_VOID_P EQUAL 8 OR win_broken_tests)
    # Test the conversion and low level views of interpreter-defined C++ arrays to NumPy
    # We do not run this test on Windows x86 due to an interpreter issue with arrays:
    # https://github.com/root-project/root/issues/9809
    ROOT_ADD_PYUNITTEST(pyroot_array_numpy_views array_conversions.py PYTHON_DEPS numpy)
endif()
