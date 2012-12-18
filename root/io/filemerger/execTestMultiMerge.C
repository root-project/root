#include "TFile.h"

int execTestMultiMerge()
{
   TFile *file = TFile::Open("mfile1-4.root");
   file->ls();
   file->cd("hist");
   gDirectory->ls();
   gDirectory->Get("Gaus")->Print();
   file->cd("named");
   gDirectory->ls();
   file->Get("MyList")->Print();
}
