int schemaRun3()
{
   TFile f("Event.root");
   gSystem->Load("libTreeFormulaScemaEvolution2");

   TTree *T = nullptr;
   f.GetObject("T",T);

   T->Show(5); //ok
   T->Scan("fTemperature"); //ok
   auto tf = new TTreeFormula("tf","fTemperature",T);
   return 0;
}
