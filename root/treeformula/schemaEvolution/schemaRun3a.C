{
  gROOT->ProcessLine(".L ./libEvent_2.so");
  TFile f("Event.root");
  T.Show(5); //ok
T.Scan("fTemperature"); //ok
// gSystem->Load("libTreePlayer");
tf = new TTreeFormula("tf","fTemperature",T);
}
