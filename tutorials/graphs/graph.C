/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Draw a simple graph.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void graph() {
   auto c = new TCanvas("c","A Simple Graph Example",200,10,700,500);

   c->SetGrid();

   const Int_t n = 20;
   Double_t x[n], y[n];
   for (Int_t i=0;i<n;i++) {
     x[i] = i*0.1;
     y[i] = 10*sin(x[i]+0.2);
     printf(" i %i %f %f \n",i,x[i],y[i]);
   }
   auto gr = new TGraph(n,x,y);
   gr->SetLineColor(2);
   gr->SetLineWidth(4);
   gr->SetMarkerColor(4);
   gr->SetMarkerStyle(21);
   gr->SetTitle("a simple graph");
   gr->GetXaxis()->SetTitle("X title");
   gr->GetYaxis()->SetTitle("Y title");
   gr->Draw("ACP");

   // TCanvas::Update() draws the frame, after which one can change it
   c->Update();
   c->GetFrame()->SetBorderSize(12);
   c->Modified();
}
