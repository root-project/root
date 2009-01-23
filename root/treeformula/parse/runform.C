{
  TTree t;
  double x = 1;
  t.Branch("x", &x, "x/D");
  t.Fill();
   t.Draw("x", TString::Format(""));
   t.Draw("x", TString::Format(""));
  return 0;
}
