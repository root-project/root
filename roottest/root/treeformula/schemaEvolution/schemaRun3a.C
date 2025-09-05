int schemaRun3a()
{
   gSystem->Load("libTreeFormulaScemaEvolution2");
   TFile f("Event.root");

   TTree *T = nullptr;
   f.GetObject("T",T);

   T->Show(5); //ok
   T->Scan("fTemperature"); //ok
   auto tf = new TTreeFormula("tf","fTemperature",T);
   return 0;
}
