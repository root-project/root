int schemaRun1()
{
   TFile f("Event.root");

   TTree *T = nullptr;
   f.GetObject("T",T);

   T->Show(5); //ok
   Long64_t n = T->Scan("fTemperature"); //ok

   return !n;
}
