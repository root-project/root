void Run() {

  //TString library = ".L namespace.";
  //gROOT->ProcessLine(library+dllsuf);
  //library = ".L template.";
  //gROOT->ProcessLine(library+dllsuf);
  //library = ".L nstemplate.";
  //// gROOT->ProcessLine(library+dllsuf);

  //library = ".L InheritMulti.";
  //gROOT->ProcessLine(library+dllsuf);
   gSystem->Load("./namespace");
   gSystem->Load("./template");
   gSystem->Load("./InheritMulti");

  namespace_driver();
  template_driver();
  //nstemplate_driver();
  if (! InheritMulti_driver() ) exit(1);

  gROOT->ProcessLine(".L array.cxx+");
  array_driver();
}
