#  Prints a summary of all ROOT benchmarks (must be run before)
#  The ROOTMARK number printed is by reference to a Pentium IV 2.4 Ghz
#  (with 512 MBytes memory and 120 GBytes IDE disk)
#  taken by definition as 600 ROOTMARKS in batch mode in executing
#     python benchmarks.py
#

import os
import ROOT

try:
 # convenience, allowing to run this file from a different directory
   macrodir = os.path.dirname( os.path.join( os.getcwd(), __file__ ) )
except NameError:
   macrodir = ''         # in case of p2.2

# use ROOT macro to make sure that bench numbers get updated in one place
ROOT.gROOT.Macro( ROOT.gSystem.UnixPathName( os.path.join( macrodir, os.pardir, 'rootmarks.C' ) ) )
