{
   //
   // To see the output of this macro, click begin_html <a href="gif/graph.gif">here</a>. end_html
   //
   gROOT->Reset();
   c1 = new TCanvas("c1","A Simple Graph Example",200,10,700,500);

   c1->SetFillColor(42);
   c1->SetGrid();

   const Int_t n = 20;
   Double_t x[n], y[n];
   for (Int_t i=0;i<n;i++) {
     x[i] = i*0.1;
     y[i] = 10*sin(x[i]+0.2);
     printf(" i %i %f %f \n",i,x[i],y[i]);
   }
   gr = new TGraph(n,x,y);
   gr->SetFillColor(19);
   gr->SetLineColor(2);
   gr->SetLineWidth(4);
   gr->SetMarkerColor(4);
   gr->SetMarkerStyle(21);
   gr->SetTitle("a simple graph");
   gr->Draw("ACP");

   //Add axis titles.
   //A graph is drawn using the services of the TH1F histogram class.
   //The histogram is created by TGraph::Paint.
   //TGraph::Paint is called by TCanvas::Update. This function is called by default
   //when typing <CR> at the keyboard. In a macro, one must force TCanvas::Update.

   c1->Update();
   c1->GetFrame()->SetFillColor(21);
   c1->GetFrame()->SetBorderSize(12);
   gr->GetHistogram()->SetXTitle("X title");
   gr->GetHistogram()->SetYTitle("Y title");
   c1->Modified();
}
