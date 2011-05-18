{
   // This macro shows a control bar o run some of the ROOT tutorials.
   // To execute an item, click with the left mouse button.

   gROOT->Reset();

   //Add the tutorials directory to the macro path
   //This is necessary in case this macro is executed from another user directory
   TString dir = gSystem->UnixPathName(gInterpreter->GetCurrentMacroName());
   dir.ReplaceAll("demos.C","");
   dir.ReplaceAll("/./","");
   const char *current = gROOT->GetMacroPath();
   gROOT->SetMacroPath(Form("%s:%s",current,dir.Data()));
   
   TControlBar *bar = new TControlBar("vertical", "Demos",10,10);
   bar->AddButton("Help Demos",".x demoshelp.C",        "Click Here For Help on Running the Demos");
   bar->AddButton("browser",   "new TBrowser;",         "Start the ROOT Browser");
   bar->AddButton("framework", ".x graphics/framework.C","An Example of Object Oriented User Interface");
   bar->AddButton("first",     ".x graphics/first.C",   "An Example of Slide with Root");
   bar->AddButton("hsimple",   ".x hsimple.C",          "An Example Creating Histograms/Ntuples on File");
   bar->AddButton("hsum",      ".x hist/hsum.C",        "Filling Histograms and Some Graphics Options");
   bar->AddButton("formula1",  ".x graphics/formula1.C","Simple Formula and Functions");
   bar->AddButton("surfaces",  ".x graphs/surfaces.C",  "Surface Drawing Options");
   bar->AddButton("fillrandom",".x hist/fillrandom.C",  "Histograms with Random Numbers from a Function");
   bar->AddButton("fit1",      ".x fit/fit1.C",         "A Simple Fitting Example");
   bar->AddButton("multifit",  ".x fit/multifit.C",     "Fitting in Subranges of Histograms");
   bar->AddButton("h1draw",    ".x hist/h1draw.C",      "Drawing Options for 1D Histograms");
   bar->AddButton("graph",     ".x graphs/graph.C",     "Example of a Simple Graph");
   bar->AddButton("gerrors",   ".x graphs/gerrors.C",   "Example of a Graph with Error Bars");
   bar->AddButton("tornado",   ".x graphics/tornado.C", "Examples of 3-D PolyMarkers");
   bar->AddButton("shapes",    ".x geom/shapes.C",      "The Geometry Shapes");
   bar->AddButton("geometry",  ".x geom/geometry.C",    "Creation of the NA49 Geometry File");
   bar->AddButton("na49view",  ".x geom/na49view.C",    "Two Views of the NA49 Detector Geometry");
   bar->AddButton("file",      ".x io/file.C",          "The ROOT File Format");
   bar->AddButton("fildir",    ".x io/fildir.C",        "The ROOT File, Directories and Keys");
   bar->AddButton("tree",      ".x tree/tree.C",        "The Tree Data Structure");
   bar->AddButton("ntuple1",   ".x tree/ntuple1.C",     "Ntuples and Selections");
   bar->AddButton("rootmarks", ".x rootmarks.C",        "Prints an Estimated ROOTMARKS for Your Machine");
   bar->SetButtonWidth(90);
   bar->Show();
   gROOT->SaveContext();
}
