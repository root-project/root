## \file
## \ingroup tutorial_pyroot
## \notebook -nodraw
## Geometry
##
## \macro_code
##
## \author Wim Lavrijsen

import os
import ROOT

try:
 # convenience, allowing to run this file from a different directory
   macrodir = os.path.expandvars("$ROOTSYS/tutorials/pyroot/")
except NameError:
   macrodir = ''         # in case of p2.2


# the na49.C file was generated, so no python conversion is provided
ROOT.gROOT.Macro( ROOT.gSystem.UnixPathName( os.path.join( macrodir, os.pardir, 'geom', 'na49.C' ) ) )
execfile( os.path.join( macrodir, 'na49visible.py' ) )
execfile( os.path.join( macrodir, 'na49geomfile.py' ) )
