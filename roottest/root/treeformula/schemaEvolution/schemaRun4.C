int schemaRun4()
{
   auto f = TFile::Open("Event.root");
   gSystem->Load("libTreeFormulaScemaEvolution2");

#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("Event * e = nullptr;");
#else
   Event * e = nullptr;
#endif

   TTree *T = nullptr;
   f->GetObject("T",T);

#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("T->SetBranchAddress(\"event\",&e);");
#else
   T->SetBranchAddress("event",&e);
#endif

   T->Show(5); //ok
   Long64_t n = T->Scan("fTemperature"); //ok

   return !n;
}
