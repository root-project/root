void create_pad_frame_gradient()
{
   const Double_t locations[] = {0., 0.2, 0.8, 1.};
   new TColor(1002, 0.25, 0.25, 0.25, "special pad color1", 0.55);
   new TColor(1003, 1., 1., 1., "special pad color2", 0.05);
   Color_t colorIndices[4] = {1002, 1003, 1003, 1002};
   new TColorGradient(1004, TColorGradient::kGDHorizontal, 4, locations, colorIndices);
}

void create_pad_gradient()
{
   const Double_t locations[] = {0., 1.};
   Color_t colorIndices[4] = {38, 30};
   new TColorGradient(1005, TColorGradient::kGDVertical, 2, locations, colorIndices);
}

void grad()
{
   const Color_t colorIndices[3] = {kRed, kOrange, kYellow};
   const Double_t lengths[3] = {0., 0.5, 1.};
   new TColorGradient(1006, TColorGradient::kGDVertical, 3, lengths, colorIndices);

   create_pad_frame_gradient();
   create_pad_gradient();

   TCanvas *cnv = new TCanvas("cnv", "gradient_test", 100, 100, 600, 600);
   cnv->SetFillColor(1005);
   cnv->SetFrameFillColor(1004);
   
   TH1F * hist = new TH1F("a", "b", 20, -3., 3.);
   hist->SetFillColor(1006);
   hist->FillRandom("gaus", 100000);
   hist->Draw();

   cnv->Update();
}
