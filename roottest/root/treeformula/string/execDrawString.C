int execDrawString()
{
   TFile::Open("cernstaff.root");

#ifdef ClingWorkAroundMissingDynamicScope
   TTree *T; gFile->GetObject("T",T);
#endif
   T->Draw("Division");
#ifdef ClingWorkAroundMissingDynamicScope
   TH1F *htemp = (TH1F*)gROOT->FindObject("htemp");
#endif
   htemp->Print("all");

   return 0;
}

