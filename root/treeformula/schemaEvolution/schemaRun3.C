{
   TFile f("Event.root");
   gSystem->Load("./libEvent_2"); 
   T.Show(5); //ok
   T.Scan("fTemperature"); //ok
   // gSystem->Load("libTreePlayer");
   tf = new TTreeFormula("tf","fTemperature",T);
}
