void transp()
{
   TH1F * hist = new TH1F("a", "b", 10, -2., 3.);
   TH1F * hist2 = new TH1F("c", "d", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   new TColor(1001, 1., 0., 0., "red", 0.85);
   hist->SetFillColor(1001);
   
   new TColor(1002, 0., 1., 0., "green", 0.5);
   hist2->SetFillColor(1002);
   
   hist2->Draw();
   hist->Draw("SAME");
}