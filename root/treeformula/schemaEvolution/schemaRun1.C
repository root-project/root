{
  TFile f("Event.root");
  T.Show(5); //ok
  Long64_t n = T.Scan("fTemperature"); //ok
  return (n!=0);
}
