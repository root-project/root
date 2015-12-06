/// \file
/// \ingroup tutorial_tree
/// Create a plot of the data in cernstaff.root
/// 
/// To create cernstaff.root, execute tutorial $ROOTSYS/tutorials/tree/cernbuild.C
/// \macro_image
/// \macro_code
///
/// \author Rene Brun
void staff() {
   // Some weight lifting to get the input file
   TString dir = gSystem->UnixPathName(__FILE__);
   dir.ReplaceAll("staff.C","");
   dir.ReplaceAll("/./","/");

   auto f = TFile::Open(Form("%scernstaff.root",dir.Data()));
   if (!f) return;
   TTree *T = nullptr;
   f->GetObject("T",T);
   T->Draw("Grade:Age:Cost:Division:Nation","","gl5d");
   if (gPad) gPad->Print("staff.C.png");
}