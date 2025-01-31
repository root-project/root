# Author: Sergey Linev GSI  01/2025

################################################################################
# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization

def wait_press_windows():
   import ROOT
   import msvcrt
   import time

   done = False
   while not done:
      k = ''
      ROOT.gSystem.ProcessEvents()
      if msvcrt.kbhit():
         k = msvcrt.getch()
         done = k[0] == 32
      else:
         time.sleep(0.01)


def wait_press_posix():
   import ROOT
   import sys
   import select
   import tty
   import termios
   import time

   old_settings = termios.tcgetattr(sys.stdin)

   tty.setcbreak(sys.stdin.fileno())

   try:

      while True:
         ROOT.gSystem.ProcessEvents()
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

