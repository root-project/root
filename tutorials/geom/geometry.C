//Example of the old geometry package (now obsolete)
//Author: Rene Brun
   
void geometry() {
   TString dir = gSystem->UnixPathName(gInterpreter->GetCurrentMacroName());
   dir.ReplaceAll("geometry.C","");
   dir.ReplaceAll("/./","/");
   gROOT->Macro(Form("%s/na49.C",dir.Data()));
   gROOT->Macro(Form("%s/na49geomfile.C",dir.Data()));
}
