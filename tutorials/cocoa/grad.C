//This macro shows how to create and use linear gradients to fill
//a histogram or pad.
//It works ONLY on MacOS X with cocoa graphical back-end.

typedef TColorGradient::Point point_type;

//______________________________________________________________________
void create_pad_frame_gradient()
{
   //We create a gradient with 4 steps - from dark (and semi-transparent)
   //gray to almost transparent (95%) white and to white and to dark gray again.
   const Double_t locations[] = {0., 0.2, 0.8, 1.};
   new TColor(1022, 0.25, 0.25, 0.25, "special pad color1", 0.55);
   new TColor(1023, 1., 1., 1., "special pad color2", 0.05);
   Color_t colorIndices[4] = {1022, 1023, 1023, 1022};
   TLinearGradient *grad = new TLinearGradient(1024, 4, locations, colorIndices);
   const point_type start(0., 0.);
   const point_type end(1., 0.);
   grad->SetStartEnd(start, end);
   
}

//______________________________________________________________________
void create_pad_gradient()
{
   //We create two-steps gradient from ROOT's standard colors (38 and 30).
   const Double_t locations[] = {0., 1.};
   Color_t colorIndices[4] = {30, 38};
   TLinearGradient *grad = new TLinearGradient(1025, 2, locations, colorIndices);
   const point_type start(0., 0.);
   const point_type end(0., 1.);
   grad->SetStartEnd(start, end);
   
}

//______________________________________________________________________
void grad()
{
   TCanvas *cnv = new TCanvas("gradient test", "gradient test", 100, 100, 600, 600);
   //After canvas was created, gVirtualX should be non-null.
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      std::cout<<"This macro works only on MacOS X with --enable-cocoa\n";
      delete cnv;
      return;
   }

   create_pad_frame_gradient();
   create_pad_gradient();
   
   cnv->SetFillColor(1025);
   cnv->SetFrameFillColor(1024);

   //Gradient to fill a histogramm
   const Color_t colorIndices[3] = {kYellow, kOrange, kRed};
   const Double_t lengths[3] = {0., 0.5, 1.};
   TLinearGradient *grad = new TLinearGradient(1026, 3, lengths, colorIndices);
   const point_type start(0., 0.);
   const point_type end(0., 1.);
   grad->SetStartEnd(start, end);
   
   TH1F * hist = new TH1F("h11", "h11", 20, -3., 3.);
   hist->SetFillColor(1026);
   hist->FillRandom("gaus", 100000);
   hist->Draw();

   cnv->Update();
}
