void exclusiongraph() {
   // Draw three graphs with an exclusion zone. This drawing mode is activated
   // when the absolute value of the graph line width (set thanks to
   // SetLineWidth) is greater than 99. In that case the line width number is
   // interpreted as 100*ff+ll = ffll . The two digits number "ll" represent the
   // normal line width whereas "ff" is the filled area width. The sign of
   // "ffll" allows to flip the filled area from one side of the line to the
   // other. The current fill area attributes are used to draw the hatched zone.
   //Author: Olivier Couet
   
   TCanvas *c1 = new TCanvas("c1","Exclusion graphs examples",200,10,700,500);
   c1->SetGrid();

   TMultiGraph *mg = new TMultiGraph();
   mg->SetTitle("Exclusion graphs");

   const Int_t n = 35;
   Double_t x1[n], x2[n], x3[n], y1[n], y2[n], y3[n];
   for (Int_t i=0;i<n;i++) {
     x1[i]  = i*0.1;
     x2[i]  = x1[i];
     x3[i]  = x1[i]+.5;
     y1[i] = 10*sin(x1[i]);
     y2[i] = 10*cos(x1[i]);
     y3[i] = 10*sin(x1[i])-2;
   }

   TGraph *gr1 = new TGraph(n,x1,y1);
   gr1->SetLineColor(2);
   gr1->SetLineWidth(1504);
   gr1->SetFillStyle(3005);

   TGraph *gr2 = new TGraph(n,x2,y2);
   gr2->SetLineColor(4);
   gr2->SetLineWidth(-2002);
   gr2->SetFillStyle(3004);
   gr2->SetFillColor(9);

   TGraph *gr3 = new TGraph(n,x3,y3);
   gr3->SetLineColor(5);
   gr3->SetLineWidth(-802);
   gr3->SetFillStyle(3002);
   gr3->SetFillColor(2);

   mg->Add(gr1);
   mg->Add(gr2);
   mg->Add(gr3);
   mg->Draw("AC");
}
