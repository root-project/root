import ROOT


ROOT.gROOT.Reset()

# the na49.C file was generated, so no python conversion is provided
ROOT.gROOT.Macro( 'na49.C' )
execfile( 'na49visible.py' )
execfile( 'na49geomfile.py' )
