void staff() {
   //to create cernstaff.root, execute tutorial $ROOTSYS/tree/cernbuild.C
   TString tutdir = gROOT->GetTutorialsDir();
   TFile *f = TFile::Open(tutdir + "/tree/cernstaff.root");
   TTree *T = (TTree*)f->Get("T");
   T->Draw("Grade:Age:Cost:Division:Nation","","gl5d");
}



