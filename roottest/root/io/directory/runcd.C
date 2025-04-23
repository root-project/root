#include "TFile.h"
#include "Riostream.h"

void printLoc() {
   if (gFile) cout << "gFile is: " << gFile->GetName() << endl;
   else cout << "gFile is: null" << endl;
   cout << "gDirectory is: " << gDirectory->GetName() << endl;
}


void runcd() {
// Fill out the code of the actual test
   TFile *file1 = new TFile("runcd1.root","RECREATE");
   file1->mkdir("onetop");
   TFile *file = new TFile("runcd.root","RECREATE");
   file->mkdir("toplevel");
   file->cd("toplevel");
   gDirectory->mkdir("subdir");
   gDirectory->cd("subdir");
   gDirectory->mkdir("lowerdir");

   gROOT->cd(); 
   printLoc();
   TDirectory::Cd("runcd.root:/toplevel");
   printLoc();
   gDirectory->cd("runcd.root:/toplevel/subdir");
   printLoc();
   TDirectory::Cd("..");
   printLoc();
   gDirectory->cd("runcd.root:/toplevel/..");
   printLoc();
   file1->cd("onetop");
   printLoc();
   TDirectory *onetop = gDirectory;
   file->cd("toplevel/subdir/lowerdir3");
   printLoc();
   file->cd("toplevel/subdir/lowerdir");
   printLoc();
   onetop->cd("..");
   printLoc();
   file->cd("/toplevel/subdir");
   printLoc();
   TDirectory::Cd("lowerdir");
   printLoc();



}
