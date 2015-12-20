# -*- coding:utf-8 -*-
#-----------------------------------------------------------------------------
#  Copyright (c) 2015, ROOT Team.
#  
#  Authors: Danilo Piparo <Danilo.Piparo@cern.ch> CERN
#           Enric Tejedor enric.tejedor.saavedra@cern.ch> CERN
#  Modified by: Omar Zapata <Omar.Zapata@cern.ch> http://oproject.org
#  website: http://oproject.org/ROOT+Jupyter+Kernel (information only for ROOT kernel)
#  Distributed under the terms of the Modified LGPLv3 License.
#
#  The full license is in the file COPYING.rst, distributed with this software.
#-----------------------------------------------------------------------------

from metakernel.display import display,HTML

import ROOT

try:
    from JupyROOT.utils import CanvasDrawer as CanvasDrawerCore
    from JupyROOT.utils import enhanceROOTModule
except ImportError:
    raise Exception("Error: JupyROOT not found")

class CanvasDrawer(CanvasDrawerCore):
    '''
    Capture the canvas which is drawn and decide if it should be displayed using
    jsROOT with metakernel.
    '''
    def __init__(self, canvas):
        CanvasDrawerCore.__init__(self,canvas)

    def _jsDisplay(self):
        '''Overloaded method to display with metakernel.display'''
        display(HTML(self.getJsCode()))


    def _pngDisplay(self):
        display(self.getPngImage())

#required to display using metakernel.display instead IPython.display
def _PyDraw(thePad):
   """
   Invoke the draw function and intercept the graphics
   """
   drawer = CanvasDrawer(thePad)
   drawer.Draw()
   
def LoadDrawer():
   enhanceROOTModule()
   ROOT.TCanvas.Draw = _PyDraw
