{
//
// This macro generates a Controlbar menu: To see the output, click begin_html <a href="gif/demos.gif" >here</a> end_html
// To execute an item, click with the left mouse button.
// To see the HELP of a button, click on the right mouse button.

   gROOT->Reset();
   gStyle->SetScreenFactor(1); //if you have a large screen, select 1,2 or 1.4

   bar = new TControlBar("vertical", "Demos",10,10);

   bar->AddButton("Help on Demos",".x demoshelp.C", "Click Here For Help on Running the Demos");
   bar->AddButton("browser",     "new TBrowser;",  "Start the ROOT Browser");
   bar->AddButton("framework",   ".x framework.C", "An Example of Object Oriented User Interface");
   bar->AddButton("first",       ".x first.C",     "An Example of Slide with Root");
   bar->AddButton("hsimple",     ".x hsimple.C",   "An Example Creating Histograms/Ntuples on File");
   bar->AddButton("hsum",        ".x hsum.C",      "Filling Histograms and Some Graphics Options");
   bar->AddButton("formula1",    ".x formula1.C",  "Simple Formula and Functions");
   bar->AddButton("surfaces",    ".x surfaces.C",  "Surface Drawing Options");
   bar->AddButton("fillrandom",  ".x fillrandom.C","Histograms with Random Numbers from a Function");
   bar->AddButton("fit1",        ".x fit1.C",      "A Simple Fitting Example");
   bar->AddButton("multifit",    ".x multifit.C",  "Fitting in Subranges of Histograms");
   bar->AddButton("h1draw",      ".x h1draw.C",    "Drawing Options for 1D Histograms");
   bar->AddButton("graph",       ".x graph.C",     "Example of a Simple Graph");
   bar->AddButton("gerrors",     ".x gerrors.C",   "Example of a Graph with Error Bars");
   bar->AddButton("tornado",     ".x tornado.C",   "Examples of 3-D PolyMarkers");
   bar->AddButton("shapes",      ".x shapes.C",    "The Geometry Shapes");
   bar->AddButton("geometry",    ".x geometry.C",  "Creation of the NA49 Geometry File");
   bar->AddButton("na49view",    ".x na49view.C",  "Two Views of the NA49 Detector Geometry");
   bar->AddButton("file",        ".x file.C",      "The ROOT File Format");
   bar->AddButton("fildir",      ".x fildir.C",    "The ROOT File, Directories and Keys");
   bar->AddButton("tree",        ".x tree.C",      "The Tree Data Structure");
   bar->AddButton("ntuple1",     ".x ntuple1.C",   "Ntuples and Selections");
   bar->AddButton("rootmarks",   ".x rootmarks.C", "Prints an Estimated ROOTMARKS for Your Machine");
   bar->Show();
   gROOT->SaveContext();
}

