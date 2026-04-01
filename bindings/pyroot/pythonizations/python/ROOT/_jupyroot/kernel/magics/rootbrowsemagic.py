# -*- coding:utf-8 -*-
# -----------------------------------------------------------------------------
#  Copyright (c) 2016, ROOT Team.
#  Authors: Danilo Piparo <Danilo.Piparo@cern.ch> CERN
# -----------------------------------------------------------------------------

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from metakernel import Magic, option

from ROOT._jupyroot.helpers.utils import browseRootFile


class RootBrowseMagics(Magic):
    def __init__(self, kernel):
        super(RootBrowseMagics, self).__init__(kernel)

    @option("arg", default="", help="Show JSROOT browser with file content")
    @option(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force opening of large files"
    )
    def line_rootbrowse(self, arg, force):
        """Open file and start browser."""
        if not browseRootFile(arg, force):
            self.kernel.Error(f"Not able to open file {arg}")
        else:
            self.kernel.do_display()


def register_magics(kernel):
    kernel.register_magics(RootBrowseMagics)
