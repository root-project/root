# General pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_pretty_printing pretty_printing.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_array_interface array_interface.py)

# TObject and subclasses pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobject_contains tobject_contains.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobject_comparisonops tobject_comparisonops.py)

# TClass pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tclass_dynamiccast tclass_dynamiccast.py)

# TDirectory and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tdirectory_attrsyntax tdirectory_attrsyntax.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tdirectoryfile_attrsyntax_get tdirectoryfile_attrsyntax_get.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tfile_attrsyntax_get_writeobject_open tfile_attrsyntax_get_writeobject_open.py)

# TTree and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch_attr ttree_branch_attr.py
                    COPY_TO_BUILDDIR TreeHelper.h)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_iterable ttree_iterable.py
                    COPY_TO_BUILDDIR TreeHelper.h)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_setbranchaddress ttree_setbranchaddress.py
                    COPY_TO_BUILDDIR TreeHelper.h)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch ttree_branch.py
                    COPY_TO_BUILDDIR TreeHelper.h)

# TH1 and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_th1_operators th1_operators.py)

# TCollection and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_len tcollection_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_listmethods tcollection_listmethods.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_operators tcollection_operators.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tcollection_iterable tcollection_iterable.py)

# TSeqCollection and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tseqcollection_itemaccess tseqcollection_itemaccess.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tseqcollection_listmethods tseqcollection_listmethods.py)

# TIter pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_titer_iterator titer_iterator.py)

# TClonesArray pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tclonesarray_itemaccess tclonesarray_itemaccess.py)

# TArray and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tarray_len tarray_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tarray_getitem tarray_getitem.py)

# TVectorT pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvectort_len tvectort_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvectort_getitem tvectort_getitem.py)

# TVector3 pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvector3_len tvector3_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tvector3_getitem tvector3_getitem.py)

# TString pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring_len tstring_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring_str_repr tstring_str_repr.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tstring_comparisonops tstring_comparisonops.py)

# TObjString pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring_len tobjstring_len.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring_str_repr tobjstring_str_repr.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tobjstring_comparisonops tobjstring_comparisonops.py)

# RVec and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_rvec_asrvec rvec_asrvec.py)

# RDataFrame and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_rdataframe_asnumpy rdataframe_asnumpy.py)

# Passing Python callables to ROOT.TF
ROOT_ADD_PYUNITTEST(pyroot_pyz_tf_pycallables tf_pycallables.py)

if(roofit)
  # RooAbsCollection and subclasses pythonizations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_rooabscollection_len rooabscollection_len.py)

  # RooDataHist pythonisations
  ROOT_ADD_PYUNITTEST(pyroot_pyz_roodatahist_ploton roodatahist_ploton.py)
endif()

# std::string_view backport in CPyCppyy
ROOT_ADD_PYUNITTEST(pyroot_string_view_backport string_view_backport.py)
