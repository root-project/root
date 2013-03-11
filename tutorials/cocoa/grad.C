//This macro shows how to create and use linear gradients to fill
//a histogram or pad.
//It works ONLY on MacOS X with cocoa graphical back-end.

//______________________________________________________________________
void create_pad_frame_gradient()
{
   //We create a gradient with 4 steps - from dark (and semi-transparent)
   //gray to almost transparent (95%) white and to white and to dark gray again.
   const Double_t locations[] = {0., 0.2, 0.8, 1.};
   new TColor(1002, 0.25, 0.25, 0.25, "special pad color1", 0.55);
   new TColor(1003, 1., 1., 1., "special pad color2", 0.05);
   Color_t colorIndices[4] = {1002, 1003, 1003, 1002};
   new TColorGradient(1004, TColorGradient::kGDHorizontal, 4, locations, colorIndices);
}

//______________________________________________________________________
void create_pad_gradient()
{
   //We create two-steps gradient from ROOT's standard colors (38 and 30).
   const Double_t locations[] = {0., 1.};
   Color_t colorIndices[4] = {38, 30};
   new TColorGradient(1005, TColorGradient::kGDVertical, 2, locations, colorIndices);
}

//______________________________________________________________________
void grad()
{
   TCanvas *cnv = new TCanvas("cnv", "gradient test", 100, 100, 600, 600);
   //After canvas was created, gVirtualX should be non-null.
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      std::cout<<"This macro works only on MacOS X with --enable-cocoa\n";
      delete cnv;
      return;
   }

   create_pad_frame_gradient();
   create_pad_gradient();
   
   cnv->SetFillColor(1005);
   cnv->SetFrameFillColor(1004);

   //Gradient to fill a histogramm
   const Color_t colorIndices[3] = {kRed, kOrange, kYellow};
   const Double_t lengths[3] = {0., 0.5, 1.};
   new TColorGradient(1006, TColorGradient::kGDVertical, 3, lengths, colorIndices);
   
   TH1F * hist = new TH1F("a", "b", 20, -3., 3.);
   hist->SetFillColor(1006);
   hist->FillRandom("gaus", 100000);
   hist->Draw();

   cnv->Update();
}
