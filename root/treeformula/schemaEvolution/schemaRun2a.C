{
  gROOT->ProcessLine(".L ./libEvent.so");
  TFile f("Event.root");

  T.Show(5); //ok
  T.Scan("fTemperature"); //ok
}
