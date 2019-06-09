## \file
## \ingroup tutorial_pyroot_legacy
## Run benchmarks macros.
##
## \macro_code
##
## \author Wim Lavrijsen

from __future__ import print_function
import os, sys
import ROOT

ROOT.SetSignalPolicy( ROOT.kSignalFast )

## macro files
macros = [
   'framework.py', 'hsimple.py', 'hsum.py', 'formula1.py',
   'fillrandom.py','fit1.py', 'h1draw.py', 'graph.py',
   'gerrors.py', 'tornado.py', 'surfaces.py', 'zdemo.py',
   'geometry.py', 'na49view.py', 'file.py',
   'ntuple1.py', 'rootmarks.py' ]

## note: this function is defined in tutorials/rootlogon.C
def bexec( dir, macro, bench ):
   if ROOT.gROOT.IsBatch():
      print('Processing benchmark: %s\n' % macro)

   summary = bench.GetPrimitive( 'TPave' )
   tmacro = summary.GetLineWith( macro )
   if tmacro:
      tmacro.SetTextColor( 4 )
   bench.Modified()
   bench.Update()

   exec( open(os.path.join( macrodir, macro )).read(), sys.modules[ __name__ ].__dict__ )

   summary2 = bench.GetPrimitive( 'TPave' )
   tmacro2 = summary2.GetLineWith( macro )
   if tmacro2:
      tmacro2.SetTextColor( 2 )
   bench.Modified()
   bench.Update()


## --------------------------------------------------------------------------
if __name__ == '__main__':

   macrodir = os.path.join(str(ROOT.gROOT.GetTutorialDir()), 'pyroot')

 # window for keeping track of bench marks that are run
   bench = ROOT.TCanvas( 'bench','Benchmarks Summary', -1000, 50, 200, 500 )
   summary = ROOT.TPaveText( 0, 0, 1, 1 )
   summary.SetTextAlign( 12 )
   summary.SetTextSize( 0.1 )
   summary.Draw()

   for m in macros:
      summary.AddText( ' ** %s' % m )

 # run benchmarks, the last one (rootmarks.py) results in a report
   for m in macros:
      bexec( macrodir, m, bench )
