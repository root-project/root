int schemaRun2()
{
   TFile f("Event.root");
   gSystem->Load("libTreeFormulaScemaEvolution");

   TTree *T = nullptr;
   f.GetObject("T",T);

   T->Show(5); //ok
   Long64_t n = T->Scan("fTemperature"); //ok

   return !n;
}
