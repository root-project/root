
{
  gROOT->ProcessLine(".L ./libEvent_2.so");
  TFile f("Event.root");
  Event* e=0;
  T->SetBranchAddress("event",&e);
  T.Show(5); //ok
  T.Scan("fTemperature"); //ok
}
