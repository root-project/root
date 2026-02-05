# Author: Sergey Linev GSI  01/2025

################################################################################
# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
\pythondoc TCanvas

Functionality of TCanvas::Update() method was extended to support interactive
graphics in the python scripts. If extra block parameter is True, script execution
will be suspended until <space> key pressed by user. Simple example:

\code{.py}
import ROOT

c = ROOT.TCanvas()
h = ROOT.TH1I("h1", "h1", 100, -5, 5)
h.FillRandom("gaus", 10000)
h.Draw("")

# block here until space is pressed
c.Update(True)

# continues after <space> is pressed
c.SaveAs("canvas.root")
\endcode
\endpythondoc
'''

from . import _run_root_event_loop, pythonization


def _TCanvas_Update(self, block = False):
   """
   Updates the canvas.
   Also blocks script execution and runs the ROOT graphics event loop until the <space> keyword is pressed,
   but only if the following conditions are met:
   * The `block` optional argument is set to `True`.
   * ROOT graphics are enabled, i.e. `ROOT.gROOT.IsBatch() == False`.
   * The script is running not in ipython notebooks.
   """

   self._Update()

   # run loop if block flag is set
   if block:
      _run_root_event_loop()


def _TCanvas_Draw(self, option: str = "", block: bool = False):
   """
   Draw the canvas.
   Also blocks script execution and runs the ROOT graphics event loop until the <space> keyword is pressed,
   but only if the following conditions are met:
   * The `block` optional argument is set to `True`.
   * ROOT graphics are enabled, i.e. `ROOT.gROOT.IsBatch() == False`.
   * The script is running not in ipython notebooks.
   """

   self._Draw(option)

   # run loop if block flag is set
   if block:
      self._Update()
      _run_root_event_loop()


@pythonization('TCanvas')
def pythonize_tcanvas(klass):
    # Parameters:
    # klass: class to be pythonized

    klass._Update = klass.Update
    klass._Draw = klass.Draw
    klass.Update = _TCanvas_Update
    klass.Draw = _TCanvas_Draw

