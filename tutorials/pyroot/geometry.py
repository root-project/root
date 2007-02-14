import os
import ROOT

macrodir = os.path.dirname( os.path.join( os.getcwd(), __file__ ) )

ROOT.gROOT.Reset()

# the na49.C file was generated, so no python conversion is provided
ROOT.gROOT.Macro( os.path.join( macrodir, os.pardir, 'geom', 'na49.C' ) )
execfile( os.path.join( macrodir, 'na49visible.py' ) )
execfile( os.path.join( macrodir, 'na49geomfile.py' ) )
