#to run this demo, do
#  on Unix:    ruby -r../lib/libRuby demo.rb
#  on Windows: ruby -r../bin/libRuby demo.rb

# This macro generates a Controlbar menu: To see the 
# output, click begin_html <a href="gif/demos.gif" >here</a> end_html
# To execute an item, click with the left mouse button.
# To see the HELP of a button, click on the right mouse button.

gROOT.Reset
gStyle.SetScreenFactor(1)   # if you have a large screen, select 1.2 or 1.4

bar = TControlBar.new("vertical", "Demos")

bar.AddButton("Help on Demos",  'TRuby::Exec("load \'demoshelp.rb\'");',
    'Click Here For Help on Running the Demos' )

bar.AddButton('browser', 'TRuby::Exec("b = TBrowser.new");',          
    'Start the ROOT browser' )

bar.AddButton('framework', 'TRuby::Exec("load \'framework.rb\'");', 
    'An Example of Object Oriented User Interface' )

bar.AddButton('canvas', 'TRuby::Exec("load \'canvas.rb\'");', 
    'A simple plot example')

bar.AddButton('surfaces', 'TRuby::Exec("load \'surfaces.rb\'");', 
    'Plotting 2D functions')

bar.AddButton('fillrandom', 'TRuby::Exec("load \'fillrandom.rb\'");', 
    'Filling a histogram with random numbers')

bar.AddButton('hsimple', 'TRuby::Exec("load \'hsimple.rb\'");', 
    'Creating histograms/Ntuples on file')

bar.AddButton('hksimple', 'TRuby::Exec("load \'hksimple.rb\'");',   
    'Utilizing the TH1K functionality')

bar.AddButton('multigraph', 'TRuby::Exec("load \'multigraph.rb\'");',   
    'Using multiple graphs')

bar.AddButton('hsum', 'TRuby::Exec("load \'hsum.rb\'");',    
    'Filling Histograms and Some Graphics Options')

bar.AddButton('hstack', 'TRuby::Exec("load \'hstack.rb\'");',
    'Using stacked histograms')

bar.AddButton('ntuple1', 'TRuby::Exec("load \'ntuple1.rb\'");',
    'Using ntuples')

bar.AddButton('tornado', 'TRuby::Exec("load \'tornado.rb\'");',
    'Examples of 3-D PolyMarkers')

bar.AddButton('latex', 'TRuby::Exec("load \'latex.rb\'");',
    'Creating LaTeX documents')

bar.AddButton('Quit', 'exit(1);',    
    'Filling Histograms and Some Graphics Options')

bar.Show

gROOT.SaveContext

gApplication.Run
