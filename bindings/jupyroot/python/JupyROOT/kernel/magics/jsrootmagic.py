# -*- coding:utf-8 -*-
#-----------------------------------------------------------------------------
#  Copyright (c) 2016, ROOT Team.
#  Authors: Danilo Piparo <Danilo.Piparo@cern.ch> CERN
#-----------------------------------------------------------------------------

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from JupyROOT.helpers.utils import enableJSVis, disableJSVis, enableJSVisDebug, TBufferJSONErrorMessage, TBufferJSONAvailable

from IPython.core.magic import Magics, line_magic, magics_class


@magics_class
class JSRootMagics(Magics):
    @line_magic
    def line_jsroot(self, args):
        """Enable or disable JavaScript visualisation. Possible values: on (default), off."""
        if args == 'on' or args == '':
           self.printErrorIfNeeded()
           enableJSVis()
        elif args == 'off':
           disableJSVis()
        elif args == 'debug':
           self.printErrorIfNeeded()
           enableJSVisDebug()

    def printErrorIfNeeded(self):
        if not TBufferJSONAvailable():
            self.kernel.Error(TBufferJSONErrorMessage)

def register_magics(kernel):
    kernel.register_magics(JSRootMagics)

