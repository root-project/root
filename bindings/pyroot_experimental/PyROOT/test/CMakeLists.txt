# General pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_pretty_printing pretty_printing.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_array_interface array_interface.py)

# TTree/TChain pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch_attr ttree_branch_attr.py
                    COPY_TO_BUILDDIR TreeHelper.h)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_iterable ttree_iterable.py
                    COPY_TO_BUILDDIR TreeHelper.h)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree-tchain_setbranchaddress ttree-tchain_setbranchaddress.py
                    COPY_TO_BUILDDIR TreeHelper.h)
