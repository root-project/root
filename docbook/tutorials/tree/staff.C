void staff() {
   //to create cernstaff.root, execute tutorial $ROOTSYS/tree/cernbuild.C
   TFile *f = TFile::Open("$ROOTSYS/tutorials/tree/cernstaff.root");
   TTree *T = (TTree*)f->Get("T");
   T->Draw("Grade:Age:Cost:Division:Nation","","gl5d");
}
   
   
   
