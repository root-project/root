TCanvas *vC1;
TGraph *grxy, *grin, *grout;

void approx()
{
//**********************************************
// Macro to test interpolation function Approx
// Author: Christian Stratowa, Vienna, Austria.
// Created: 26 Aug 2001                           
//**********************************************


// test data (square)
   Int_t n = 11;
   Double_t x[] = {1,2,3,4,5,6,6,6,8,9,10};
   Double_t y[] = {1,4,9,16,25,25,36,49,64,81,100};
   grxy = new TGraph(n,x,y);

// x values, for which y values should be interpolated
   Int_t nout = 14;
   Double_t xout[] = 
      {1.2,1.7,2.5,3.2,4.4,5.2,5.7,6.5,7.6,8.3,9.7,10.4,11.3,13};

// create Canvas
   vC1 = new TCanvas("vC1","square",200,10,700,700);
   vC1->Divide(2,2);

// initialize graph with data
   grin = new TGraph(n,x,y);
// interpolate at equidistant points (use mean for tied x-values)
   TGraphSmooth *gs = new TGraphSmooth("normal");
   grout = gs->Approx(grin,"linear");
   DrawSmooth(1,"Approx: ties = mean","X-axis","Y-axis");

// re-initialize graph with data 
// (since graph points were set to unique vales)
   grin = new TGraph(n,x,y);
// interpolate at given points xout
   grout = gs->Approx(grin,"linear", 14, xout, 0, 130);
   DrawSmooth(2,"Approx: ties = mean","","");

// print output variables for given values xout
   Int_t vNout = grout->GetN();
   Double_t vXout, vYout;
   for (Int_t k=0;k<vNout;k++) {
      grout->GetPoint(k, vXout, vYout);
      cout << "k= " << k << "  vXout[k]= " << vXout
           << "  vYout[k]= " << vYout << endl;
   }

// re-initialize graph with data
   grin = new TGraph(n,x,y);
// interpolate at equidistant points (use min for tied x-values)
//   grout = gs->Approx(grin,"linear", 50, 0, 0, 0, 1, 0, "min");
   grout = gs->Approx(grin,"constant", 50, 0, 0, 0, 1, 0.5, "min");
   DrawSmooth(3,"Approx: ties = min","","");

// re-initialize graph with data
   grin = new TGraph(n,x,y);
// interpolate at equidistant points (use max for tied x-values)
   grout = gs->Approx(grin,"linear", 14, xout, 0, 0, 2, 0, "max");
   DrawSmooth(4,"Approx: ties = max","","");

// cleanup
   delete gs;
}

void DrawSmooth(Int_t pad, const char *title, const char *xt, 
   const char *yt)
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

