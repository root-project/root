/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Create and draw a polar graph. See the [TGraphPolar documentation](https://root.cern/doc/master/classTGraphPolar.html)
///
/// Since TGraphPolar is a TGraphErrors, it is painted with
/// [TGraphPainter](https://root.cern/doc/master/classTGraphPainter.html) options.
///
/// With GetPolargram we retrieve the polar axis to format it; see the
/// [TGraphPolargram documentation](https://root.cern/doc/master/classTGraphPolargram.html)
///
/// \macro_image
/// \macro_code
/// \author Olivier Couet

void gr012_polar()
{
   // Illustrates how to use TGraphPolar

   TCanvas * CPol = new TCanvas("CPol","TGraphPolar Examples",1200,600);
   CPol->Divide(2,1);

   // Left-side pad. Two graphs without errors
   CPol->cd(1);
   Double_t xmin=0;
   Double_t xmax=TMath::Pi()*2;

   Double_t x[1000];
   Double_t y[1000];
   Double_t xval1[20];
   Double_t yval1[20];

   // Graph 1 to be drawn with line and fill
   TF1 * fplot = new TF1("fplot","cos(2*x)*cos(20*x)",xmin,xmax);
   for (Int_t ipt = 0; ipt < 1000; ipt++){
      x[ipt] = ipt*(xmax-xmin)/1000 + xmin;
      y[ipt] = fplot->Eval(x[ipt]);
   }
   TGraphPolar * grP = new TGraphPolar(1000,x,y);
   grP->SetLineColor(2);
   grP->SetLineWidth(2);
   grP->SetFillStyle(3012);
   grP->SetFillColor(2);
   grP->Draw("AFL");

   // Graph 2 to be drawn superposed over graph 1, with curve and polymarker
   for (Int_t ipt = 0; ipt < 20; ipt++){
      xval1[ipt] = x[1000/20*ipt];
      yval1[ipt] = y[1000/20*ipt];
   }
   TGraphPolar * grP1 = new TGraphPolar(20,xval1,yval1);
   grP1->SetMarkerStyle(29);
   grP1->SetMarkerSize(2);
   grP1->SetMarkerColor(4);
   grP1->SetLineColor(4);
   grP1->Draw("CP");

   // To format the polar axis, we retrieve the TGraphPolargram.
   // First update the canvas, otherwise GetPolargram returns 0
   CPol->Update();
   if (grP1->GetPolargram()) {
      grP1->GetPolargram()->SetTextColor(8);
      grP1->GetPolargram()->SetRangePolar(-TMath::Pi(),TMath::Pi());
      grP1->GetPolargram()->SetNdivPolar(703);
      grP1->GetPolargram()->SetToRadian(); // tell ROOT that the x and xval1 are in radians
   }

   // Right-side pad. One graph with errors
   CPol->cd(2);
   Double_t x2[30];
   Double_t y2[30];
   Double_t ex[30];
   Double_t ey[30];
   for (Int_t ipt = 0; ipt < 30; ipt++){
      x2[ipt] = x[1000/30*ipt];
      y2[ipt] = 1.2 + 0.4*sin(TMath::Pi()*2*ipt/30);
      ex[ipt] = 0.2+0.1*cos(2*TMath::Pi()/30*ipt);
      ey[ipt] = 0.2;
   }

   // Grah to be drawn with polymarker and errors
   TGraphPolar * grPE = new TGraphPolar(30,x2,y2,ex,ey);
   grPE->SetMarkerStyle(22);
   grPE->SetMarkerSize(1.5);
   grPE->SetMarkerColor(5);
   grPE->SetLineColor(6);
   grPE->SetLineWidth(2);
   grPE->Draw("EP");

   // To format the polar axis, we retrieve the TGraphPolargram.
   // First update the canvas, otherwise GetPolargram returns 0
   CPol->Update();
   if (grPE->GetPolargram()) {
      grPE->GetPolargram()->SetTextSize(0.03);
      grPE->GetPolargram()->SetTwoPi();
      grPE->GetPolargram()->SetToRadian(); // tell ROOT that the x2 values are in radians
   }
}
