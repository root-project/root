#define mc01_cxx
#include "mc01.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

void mc01::Loop(int arg)
{
//   In a ROOT session, you can do:
//      Root > .L mc01.C
//      Root > mc01 t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(i);  // read all branches
//by  b_branchname->GetEntry(i); //read only this branch

   // This trick seems to have been disabled in CINT :(
   #include "mcode.C"
}
#undef mc01_cxx
