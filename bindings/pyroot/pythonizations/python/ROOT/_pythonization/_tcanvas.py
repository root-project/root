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
\endcode
import ROOT

c = ROOT.TCanvas()
h = ROOT.TH1I("h1", "h1", 100, -5, 5)
h.FillRandom("gaus", 10000)
h.Draw("")

# block here until space is pressed
c.Update(True)

# continues after <space> is pressed
c.SaveAs("canvas.root")
\endpythondoc
'''

from . import pythonization

def wait_press_windows():
   from ROOT import gSystem
   import msvcrt
   import time

   while not gSystem.ProcessEvents():
      if msvcrt.kbhit():
         k = msvcrt.getch()
         if k[0] == 32:
            break
      else:
         time.sleep(0.01)


def wait_press_posix():
   from ROOT import gSystem
   import sys
   import select
   import tty
   import termios
   import time

   old_settings = termios.tcgetattr(sys.stdin)

   tty.setcbreak(sys.stdin.fileno())

   try:

      while not gSystem.ProcessEvents():
         c = ''
         if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            c = sys.stdin.read(1)
         if (c == '\x20'):
            break
         time.sleep(0.01)

   finally:
      termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def _TCanvas_Update(self, block = False):
   """
   Updates the canvas.
   Also blocks script execution and runs the ROOT graphics event loop until the <space> keyword is pressed,
   but only if the following conditions are met:
   * The `block` optional argument is set to `True`.
   * ROOT graphics are enabled, i.e. `ROOT.gROOT.IsBatch() == False`.
   * The script is running not in ipython notebooks.
   """

   from ROOT import gROOT
   import os
   import sys

   self._Update()

   # blocking flag is not set
   if not block:
      return

   # no special handling in batch mode
   if gROOT.IsBatch():
      return

   # no special handling in case of notebooks
   if 'IPython' in sys.modules and sys.modules['IPython'].version_info[0] >= 5:
      return

   print("Press <space> key to continue")

   if os.name == 'nt':
      wait_press_windows()
   else:
      wait_press_posix()


@pythonization('TCanvas')
def pythonize_tcanvas(klass):
    # Parameters:
    # klass: class to be pythonized

    klass._Update = klass.Update
    klass.Update = _TCanvas_Update

