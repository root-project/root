{
  TFile f("Event.root");
  gSystem->Load("./libEvent_2"); 
  Event * e =0;
  T->SetBranchAddress("event",&e);
  T.Show(5); //ok
  Long64_t n = T.Scan("fTemperature"); //ok
  return (n!=0);  
}
