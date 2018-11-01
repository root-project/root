# General pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_pretty_printing pretty_printing.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_array_interface array_interface.py)

# TFile, TDirectory, TDirectoryFile pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_tdirectoryfile_read_write tdirectoryfile_read_write.py)
ROOT_ADD_PYUNITTEST(pyroot_pyz_tfile_open_read_write tfile_open_read_write.py)

# TTree and subclasses pythonizations
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch_attr ttree_branch_attr.py
                    COPY_TO_BUILDDIR TreeHelper.h)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_iterable ttree_iterable.py
                    COPY_TO_BUILDDIR TreeHelper.h)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_setbranchaddress ttree_setbranchaddress.py
                    COPY_TO_BUILDDIR TreeHelper.h)
ROOT_ADD_PYUNITTEST(pyroot_pyz_ttree_branch ttree_branch.py
                    COPY_TO_BUILDDIR TreeHelper.h)
