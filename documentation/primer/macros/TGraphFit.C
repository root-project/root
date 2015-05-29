void TGraphFit(){
   //
   // Draw a graph with error bars and fit a function to it
   //
   gStyle->SetOptFit(111) ; //superimpose fit results
   // make nice Canvas
   TCanvas *c1 = new TCanvas("c1" ,"Daten" ,200 ,10 ,700 ,500) ;
   c1->SetGrid( ) ;
   //define some data points ...
   const Int_t n = 10;
   Float_t x[n] = {-0.22, 0.1, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 1.1};
   Float_t y[n] = {0.7, 2.9, 5.6, 7.4, 9., 9.6, 8.7, 6.3, 4.5, 1.1};
   Float_t ey[n] = {.8 ,.7 ,.6 ,.5 ,.4 ,.4 ,.5 ,.6 ,.7 ,.8};
   Float_t ex[n] = {.05 ,.1 ,.07 ,.07 ,.04 ,.05 ,.06 ,.07 ,.08 ,.05};
   // and hand over to TGraphErros object
   TGraphErrors *gr = new TGraphErrors(n,x,y,ex,ey);
   gr->SetTitle("TGraphErrors with Fit") ;
   gr->Draw("AP");
   // now perform a fit (with errors in x and y!)
   gr->Fit("gaus");
   c1->Update();
}
