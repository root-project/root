# -*- coding:utf-8 -*-
# -----------------------------------------------------------------------------
#  Copyright (c) 2026, ROOT Team.
#  Authors: Sergey Linev <S.Linev@gsi.de> GSI
# -----------------------------------------------------------------------------

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from ROOT._jupyroot.helpers.utils import browseRootFile


@magics_class
class RootBrowseMagics(Magics):
    def __init__(self, shell):
        super(RootBrowseMagics, self).__init__(shell)

    @line_magic
    @magic_arguments()
    @argument(
        "arg",
        nargs="?",
        default="",
        help="Open and browse ROOT file",
    )
    @argument(
       "--force",
       "-f",
       action="store_true",
       help="Force opening of large files"
    )
    def rootbrowse(self, line):
        """start root browser."""
        args = parse_argstring(self.rootbrowse, line)
        if not args:
            print("Provide ROOT file name")
        else:
            browseRootFile(args.arg, args.force)



def load_ipython_extension(ipython):
    ipython.register_magics(RootBrowseMagics)
