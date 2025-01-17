## \file
## \ingroup Tutorials
## To run, do "python <path-to>/demos.py"
##
## \macro_code
##
## \author Wim Lavrijsen, Enric Tejedor

import os, sys
import ROOT

# To run, do "python <path-to>/demos.py"

# enable running from another directory than the one where demos.py resides
workdir = os.path.dirname( sys.argv[0] )
if workdir:
   os.chdir( workdir )

# This macro generates a Controlbar menu.
# To execute an item, click with the left mouse button.
# To see the HELP of a button, click on the right mouse button.

ROOT.gStyle.SetScreenFactor(1)   # if you have a large screen, select 1.2 or 1.4

bar = ROOT.TControlBar( 'vertical', 'Demos', 10, 10 )

# The callbacks to python work by having CLING call the python interpreter through
# the "TPython" class. Note the use of "raw strings."
to_run = 'exec(open(\'{}\').read())'


bar.AddButton( 'Help on Demos', r'TPython::Exec( "' + to_run.format('demoshelp.py') + '" );', 'Click Here For Help on Running the Demos' )
bar.AddButton( 'browser',       r'TPython::Exec( "b = ROOT.TBrowser()" );',          'Start the ROOT browser' )
bar.AddButton( 'hsimple',       r'TPython::Exec( "' + to_run.format('hsimple.py') + '" );',   'Creating histograms/Ntuples on file', "button" )
bar.AddButton( 'hsum',          r'TPython::Exec( "' + to_run.format('hist/hsum.py') + '" );',      'Filling Histograms and Some Graphics Options' )
bar.AddButton( 'formula1',      r'TPython::Exec( "' + to_run.format('visualisation/graphics/formula1.py') + '" );',  'Simple Formula and Functions' )
bar.AddButton( 'surfaces',      r'TPython::Exec( "' + to_run.format('visualisation/graphics/surfaces.py') + '" );',  'Surface Drawing Options' )
bar.AddButton( 'fillrandom',    r'TPython::Exec( "' + to_run.format('hist/fillrandom.py') + '" );','Histograms with Random Numbers from a Function' )
bar.AddButton( 'fit1',          r'TPython::Exec( "' + to_run.format('math/fit/fit1.py') + '" );',      'A Simple Fitting Example' )
bar.AddButton( 'multifit',      r'TPython::Exec( "' + to_run.format('math/fit/multifit.py') + '" );',  'Fitting in Subranges of Histograms' )
bar.AddButton( 'h1draw',        r'TPython::Exec( "' + to_run.format('hist/h1ReadAndDraw.py') + '" );',    'Drawing Options for 1D Histograms' )
bar.AddButton( 'graph',         r'TPython::Exec( "' + to_run.format('visualisation/graphs/graph.py') + '" );',     'Example of a Simple Graph' )
bar.AddButton( 'gerrors',       r'TPython::Exec( "' + to_run.format('visualisation/graphs/gerrors.py') + '" );',   'Example of a Graph with Error Bars' )
bar.AddButton( 'tornado',       r'TPython::Exec( "' + to_run.format('visualisation/graphics/tornado.py') + '" );',   'Examples of 3-D PolyMarkers' )
bar.AddButton( 'shapes',        r'TPython::Exec( "' + to_run.format('visualisation/geom/shapes.py') + '" );',    'The Geometry Shapes' )
bar.AddButton( 'geometry',      r'TPython::Exec( "' + to_run.format('visualisation/geom/geometry.py') + '" );',  'Creation of the NA49 Geometry File' )
bar.AddButton( 'na49view',      r'TPython::Exec( "' + to_run.format('visualisation/geom/na49view.py') + '" );',  'Two Views of the NA49 Detector Geometry' )
bar.AddButton( 'ntuple1',       r'TPython::Exec( "' + to_run.format('io/tree/ntuple1.py') + '" );',   'Ntuples and Selections' )
bar.AddSeparator()       # not implemented
bar.AddButton( 'make ntuple',   r'TPython::Exec( "' + to_run.format('io/tree/csv2tntuple.py') + '" );',       'Convert a text file to an ntuple' )

bar.Show()

ROOT.gROOT.SaveContext()


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
   rep = ''
   while not rep in [ 'q', 'Q' ]:
      rep = input( 'enter "q" to quit: ' )
      if 1 < len(rep):
         rep = rep[0]
