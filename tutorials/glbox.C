void glbox()
{
   // Display a 3D histogram using GL (box option).

   TCanvas *c = new TCanvas("glc","TH3 Drawing",0,0,700,900);
   c->Divide(1,2);

   TH3F *h31 = new TH3F("h31", "h31", 10, -1, 1, 10, -1, 1, 10, -1, 1);
   h31->FillRandom("gaus");
   h31->SetFillColor(2);
   c->cd(1);
   h31->Draw("glbox");

   TH3F *h32 = new TH3F("h32", "h32", 10, -2, 2, 10, -1, 1, 10, -3, 3);
   h32->FillRandom("gaus");
   h32->SetFillColor(4);
   c->cd(2);
   h32->Draw("glbox1");
}
