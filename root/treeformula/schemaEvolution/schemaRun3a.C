{
  gSystem->Load("./libEvent_2"); 
  TFile f("Event.root");
  T.Show(5); //ok
  T.Scan("fTemperature"); //ok
  // gSystem->Load("libTreePlayer");
  tf = new TTreeFormula("tf","fTemperature",T);
}
