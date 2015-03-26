void staff() {
   //to create cernstaff.root, execute tutorial $ROOTSYS/tree/cernbuild.C
   TFile *f = TFile::Open("cernstaff.root");
   TTree *T = (TTree*)f->Get("T");
   T->Draw("Grade:Age:Cost:Division:Nation","","gl5d");
}



