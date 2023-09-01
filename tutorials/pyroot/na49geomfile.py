## \file
## \ingroup tutorial_pyroot
## Before executing this macro, the file makegeometry.C must have been executed
##
## \macro_code
##
## \author Wim Lavrijsen

import ROOT

ROOT.gBenchmark.Start( 'geometry' )
na = ROOT.TFile( 'py-na49.root', 'RECREATE' )
n49 = ROOT.gROOT.FindObject( 'na49' )
n49.Write()
na.Write()
na.Close()
ROOT.gBenchmark.Show( 'geometry' )

