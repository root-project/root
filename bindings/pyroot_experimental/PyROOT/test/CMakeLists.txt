# General pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pretty_printing pretty_printing.py)

# TTree pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch_attr ttree_branch_attr.py
                    COPY_TO_BUILDDIR TreeHelper.h)

