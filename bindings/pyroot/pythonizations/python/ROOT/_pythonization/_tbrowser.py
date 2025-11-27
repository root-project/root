# Author: Sergey Linev GSI  11/2025

################################################################################
# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
\pythondoc TBrowser

Functionality of TBrowser::Draw() method was extended to support interactive
work in the python scripts. If extra block parameter is True, script execution
will be suspended until <space> key pressed by user. Simple example:

\code{.py}
\endcode
import ROOT

br = ROOT.TBrowser()

# block here until space is pressed
br.Draw(block=True)

# continues after <space> is pressed
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

def run_root_event_loop():
   from ROOT import gROOT
   import os
   import sys

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



def _TBrowser_Draw(self, option: str = "", block: bool = False):
   """
   Draw the browser.
   Also blocks script execution and runs the ROOT graphics event loop until the <space> keyword is pressed,
   but only if the following conditions are met:
   * The `block` optional argument is set to `True`.
   * ROOT graphics are enabled, i.e. `ROOT.gROOT.IsBatch() == False`.
   * The script is running not in ipython notebooks.
   """

   self._Draw(option)

   # run loop if block flag is set
   if block:
      run_root_event_loop()


@pythonization('TBrowser')
def pythonize_tbrowser(klass):
    # Parameters:
    # klass: class to be pythonized

    klass._Draw = klass.Draw
    klass.Draw = _TBrowser_Draw

