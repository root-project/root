// Draw a graph with text attached to each point.
// The text is drawn in a TExec function, therefore if the text is
// moved interactively, it will be automatically updated.
// Author: Olivier Couet
void graphtext() {
   TCanvas *c = new TCanvas("c","A Simple Graph Example with Text",700,500);
   c->SetGrid();

   const Int_t n = 10;
   TGraph *gr = new TGraph(n);
   gr->SetTitle("A Simple Graph Example with Text");
   gr->SetMarkerStyle(20);
   TExec *ex = new TExec("ex","drawtext();");
   gr->GetListOfFunctions()->Add(ex);

   Double_t x, y;
   for (Int_t i=0;i<n;i++) {
      x = i*0.1;
      y = 10*sin(x+0.2);
      gr->SetPoint(i,x,y);

   }
   gr->Draw("ALP");
}

void drawtext()
{
   Int_t i,n;
   Double_t x,y;
   TLatex *l;

   TGraph *g = (TGraph*)gPad->GetListOfPrimitives()->FindObject("Graph");
   n = g->GetN();
   for (i=1; i<n; i++) {
      g->GetPoint(i,x,y);
      l = new TLatex(x,y+0.2,Form("%4.2f",y));
      l->SetTextSize(0.025);
      l->SetTextFont(42);
      l->SetTextAlign(21);
      l->Paint();
   }
}

