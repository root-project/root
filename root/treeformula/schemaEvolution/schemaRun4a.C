{
  gSystem->Load("./libEvent_2"); 
  TFile f("Event.root");
  Event * e =0;
  T->SetBranchAddress("event",&e);
  T.Show(5); //ok
  T.Scan("fTemperature"); //ok
}
