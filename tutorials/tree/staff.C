void staff() {
  //to create cernstaff.root, execute tutorial $ROOTSYS/tree/cernbuild.C
   TString dir = gSystem->DirName(gInterpreter->GetCurrentMacroName());
   if (gSystem->AccessPathName(dir+"/cernstaff.root")) {
     gROOT->SetMacroPath(dir);
     gROOT->ProcessLine(".x cernbuild.C");
   }
   TFile *f = TFile::Open(dir+"/cernstaff.root");
   TTree *T = (TTree*)f->Get("T");
   T->Draw("Grade:Age:Cost:Division:Nation","","gl5d");
}
   
   
   
