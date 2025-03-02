# -*- coding:utf-8 -*-
#-----------------------------------------------------------------------------
#  Authors: Omar Zapata <Omar.Zapata@cern.ch> http://oproject.org
#           Danilo Piparo <Danilo.Piparo@cern.ch> CERN
#           Enric Tejedor enric.tejedor.saavedra@cern.ch> CERN
#-----------------------------------------------------------------------------

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from IPython.core.magic import Magics, cell_magic, magics_class

import sys

#NOTE:actually JupyROOT is not capturing the error on %%cpp -d if the function is wrong
@magics_class
class CppMagics(Magics):
    @cell_magic
    def cell_cpp(self, args):
        """Executes the content of the cell as C++ code.

        Options (mutually exclusive):
            -a : Compile code with ACLiC.
            -d : Declare functions and/or classes.:
        """
        if self.code.strip():
             execFunc = None
             if args == '-a':
                 execFunc = self.kernel.ACLiC
             elif args == '-d':
                 execFunc = self.kernel.Declarer.Run
             else: # normal flow
                self.kernel.do_execute_direct(str(self.code))
                self.evaluate = False
                return

             self.kernel.ioHandler.Clear()
             self.kernel.ioHandler.InitCapture()
             execFunc(self.code)
             self.kernel.ioHandler.EndCapture()
             self.kernel.print_output(self.kernel.ioHandler)

        self.evaluate = False

def register_magics(kernel):
    kernel.register_magics(CppMagics)

