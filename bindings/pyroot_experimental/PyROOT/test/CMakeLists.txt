# General pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_pretty_printing pretty_printing.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_array_interface array_interface.py)

# TTree/TChain pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch_attr ttree_branch_attr.py
                    COPY_TO_BUILDDIR TreeHelper.h mystruct.h)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_iterable ttree_iterable.py
                    COPY_TO_BUILDDIR TreeHelper.h mystruct.h)

ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree-tchain_setbranchaddress ttree-tchain_setbranchaddress.py
                    COPY_TO_BUILDDIR TreeHelper.h mystruct.h)

# TFile pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tfile_open_attrsyntax tfile_open_attrsyntax.py)

# TDirectory::WriteObject pythonisations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tdirectory_write_getobject tdirectory_write_getobject.py
                    COPY_TO_BUILDDIR mystruct.h)

