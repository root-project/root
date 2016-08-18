/// \file
/// \ingroup tutorial_graphs
/// \notebook -js
/// Macro to test interpolation function Approx
///
/// \macro_image
/// \macro_code
///
/// \author Christian Stratowa, Vienna, Austria.

TCanvas *vC1;
TGraph *grxy, *grin, *grout;

void DrawSmooth(Int_t pad, const char *title, const char *xt, const char *yt)
{
  vC1->cd(pad);
  TH1F *vFrame = gPad->DrawFrame(0,0,15,150);
  vFrame->SetTitle(title);
  vFrame->SetTitleSize(0.2);
  vFrame->SetXTitle(xt);
  vFrame->SetYTitle(yt);
  grxy->SetMarkerColor(kBlue);
  grxy->SetMarkerStyle(21);
  grxy->SetMarkerSize(0.5);
  grxy->Draw("P");
  grin->SetMarkerColor(kRed);
  grin->SetMarkerStyle(5);
  grin->SetMarkerSize(0.7);
  grin->Draw("P");
  grout->DrawClone("LP");
}

void approx()
{
// Test data (square)
   Int_t n = 11;
   Double_t x[] = {1,2,3,4,5,6,6,6,8,9,10};
   Double_t y[] = {1,4,9,16,25,25,36,49,64,81,100};
   grxy = new TGraph(n,x,y);

// X values, for which y values should be interpolated
   Int_t nout = 14;
   Double_t xout[] =
      {1.2,1.7,2.5,3.2,4.4,5.2,5.7,6.5,7.6,8.3,9.7,10.4,11.3,13};

// Create Canvas
   vC1 = new TCanvas("vC1","square",200,10,700,700);
   vC1->Divide(2,2);

// Initialize graph with data
   grin = new TGraph(n,x,y);
// Interpolate at equidistant points (use mean for tied x-values)
   TGraphSmooth *gs = new TGraphSmooth("normal");
   grout = gs->Approx(grin,"linear");
   DrawSmooth(1,"Approx: ties = mean","X-axis","Y-axis");

// Re-initialize graph with data
// (since graph points were set to unique vales)
   grin = new TGraph(n,x,y);
// Interpolate at given points xout
   grout = gs->Approx(grin,"linear", 14, xout, 0, 130);
   DrawSmooth(2,"Approx: ties = mean","","");

// Print output variables for given values xout
   Int_t vNout = grout->GetN();
   Double_t vXout, vYout;
   for (Int_t k=0;k<vNout;k++) {
      grout->GetPoint(k, vXout, vYout);
      cout << "k= " << k << "  vXout[k]= " << vXout
           << "  vYout[k]= " << vYout << endl;
   }

// Re-initialize graph with data
   grin = new TGraph(n,x,y);
// Interpolate at equidistant points (use min for tied x-values)
//   _grout = gs->Approx(grin,"linear", 50, 0, 0, 0, 1, 0, "min");_ 
   grout = gs->Approx(grin,"constant", 50, 0, 0, 0, 1, 0.5, "min");
   DrawSmooth(3,"Approx: ties = min","","");

// Re-initialize graph with data
   grin = new TGraph(n,x,y);
// Interpolate at equidistant points (use max for tied x-values)
   grout = gs->Approx(grin,"linear", 14, xout, 0, 0, 2, 0, "max");
   DrawSmooth(4,"Approx: ties = max","","");

// Cleanup
   delete gs;
}

