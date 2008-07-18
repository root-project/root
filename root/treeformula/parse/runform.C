{
  TTree t;
  double x = 1;
  t.Branch("x", &x, "x/D");
  t.Fill();
  t.Draw("x", Form(""));
  t.Draw("x", Form(""));
  return 0;
}
