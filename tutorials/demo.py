#--Setup the proper environment if in SEAL project and available SCRAM_ARCH-------------------
import os, sys
if 'SEAL' in os.environ : 
  seal = os.environ['SEAL']
  if 'SCRAM_ARCH' in os.environ : arch = os.environ['SCRAM_ARCH']
  else                          : arch = 'rh73_gcc32'
  sys.path.append(seal+'/'+ arch+'/lib')
  sys.path.append(seal+'/src/Scripting/PyROOT/src')
#--end Setup environment-----------------------------------------------------------------------

import ROOT

# To run, do an "execfile( '<path-to>/demos.py' )" or "python <path-to>/demos.py"

## allow running from another directory than the one where demo.py resides
workdir = os.path.dirname( sys.argv[0] )
if workdir:
   os.chdir( workdir )
   
# This macro generates a Controlbar menu: To see the output, click begin_html <a href="gif/demos.gif" >here</a> end_html
# To execute an item, click with the left mouse button.
# To see the HELP of a button, click on the right mouse button.

ROOT.gROOT.Reset()
ROOT.gStyle.SetScreenFactor(1)   # if you have a large screen, select 1.2 or 1.4

bar = ROOT.TControlBar( 'vertical', 'Demos' )

bar.AddButton( 'Help on Demos', r'Python::exec( "execfile( \'demoshelp.py\' )" );', 'Click Here For Help on Running the Demos' )
bar.AddButton( 'browser',       r'Python::exec( "b = ROOT.TBrowser()" );',          'Start the ROOT browser' )
bar.AddButton( 'framework',     r'Python::exec( "execfile( \'framework.py\' )" );', 'An Example of Object Oriented User Interface' )
bar.AddButton( 'first',         r'Python::exec( "execfile( \'first.py\' )" );',     'An Example of Slide with Root' )
bar.AddButton( 'hsimple',       r'Python::exec( "execfile( \'hsimple.py\' )" );',   'Creating histograms/Ntuples on file', "button" )
bar.AddButton( 'hsum',          r'Python::exec( "execfile( \'hsum.py\' )" );',      'Filling Histograms and Some Graphics Options' )
bar.AddButton( 'formula1',      r'Python::exec( "execfile( \'formula1.py\' )" );',  'Simple Formula and Functions' )
bar.AddButton( 'surfaces',      r'Python::exec( "execfile( \'surfaces.py\' )" );',  'Surface Drawing Options' )
bar.AddButton( 'fillrandom',    r'Python::exec( "execfile( \'fillrandom.py\' )" );','Histograms with Random Numbers from a Function' )
bar.AddButton( 'fit1',          r'Python::exec( "execfile( \'fit1.py\' )" );',      'A Simple Fitting Example' )
bar.AddButton( 'multifit',      r'Python::exec( "execfile( \'multifit.py\' )" );',  'Fitting in Subranges of Histograms' )
bar.AddButton( 'h1draw',        r'Python::exec( "execfile( \'h1draw.py\' )" );',    'Drawing Options for 1D Histograms' )
bar.AddButton( 'graph',         r'Python::exec( "execfile( \'graph.py\' )" );',     'Example of a Simple Graph' )
bar.AddButton( 'gerrors',       r'Python::exec( "execfile( \'gerrors.py\' )" );',   'Example of a Graph with Error Bars' )
bar.AddButton( 'tornado',       r'Python::exec( "execfile( \'tornado.py\' )" );',   'Examples of 3-D PolyMarkers' )
bar.AddButton( 'shapes',        r'Python::exec( "execfile( \'shapes.py\' )" );',    'The Geometry Shapes' )
#bar.AddButton( 'geometry',      r'Python::exec( "execfile( \'geometry.py\' )" );',  'Creation of the NA49 Geometry File' )
#bar.AddButton( 'na49view',      r'Python::exec( "execfile( \'na49view.py\' )" );',  'Two Views of the NA49 Detector Geometry' )
bar.AddButton( 'file',          r'Python::exec( "execfile( \'file.py\' )" );',      'The ROOT File Format' )
bar.AddButton( 'fildir',        r'Python::exec( "execfile( \'fildir.py\' )" );',    'The ROOT File, Directories and Keys' )
bar.AddButton( 'tree',          r'Python::exec( "execfile( \'tree.py\' )" );',      'The Tree Data Structure' )
bar.AddButton( 'ntuple1',       r'Python::exec( "execfile( \'ntuple1.py\' )" );',   'Ntuples and Selections' )
#bar.AddButton( 'rootmarks',     r'Python::exec( "execfile( \'rootmarks.py\' )" );', 'Prints an Estimated ROOTMARKS for Your Machine' )
bar.AddSeparator()       # not implemented
bar.AddButton( 'make ntuple',   r'Python::exec( "execfile( \'mrt.py\' )" );',       'Convert a text file to an ntuple' )

bar.Show()

ROOT.gROOT.SaveContext()


## wait for input to keep the GUI (which lives on a ROOT event dispatcher) alive
if __name__ == '__main__':
   rep = ''
   while not rep in [ 'q', 'Q' ]:
      rep = raw_input( 'enter "q" to quit: ' )
      if 1 < len(rep):
         rep = rep[0]
