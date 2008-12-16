void fitcont()
{
   // Example illustrating how to draw the n-sigma contour of a Minuit fit.
   // To get the n-sigma contour the ERRDEF parameter in Minuit has to set
   // to n^2. The fcn function has to be set before the routine is called.
   //
   // WARNING!!! This test works only with TMinuit
   //
   // The TGraph object is created via the interpreter. The user must cast it
   // to a TGraph*
   // Author:  Rene Brun

   //be sure default is Minuit since we will use gMinuit 
   TVirtualFitter::SetDefaultFitter("Minuit");  
      
   TCanvas *c1 = new TCanvas("c1");
   TH1F *h = new TH1F("h","My histogram",100,-3,3);
   h->FillRandom("gaus",6000);
   h->Fit("gaus");
   c1->Update();
   
   TCanvas *c2 = new TCanvas("c2","contours",10,10,600,800);
   c2->Divide(1,2);
   c2->cd(1);
   //get first contour for parameter 1 versus parameter 2
   TGraph *gr12 = (TGraph*)gMinuit->Contour(40,1,2);
   gr12->Draw("alp");
   c2->cd(2);
   //Get contour for parameter 0 versus parameter 2  for ERRDEF=2 
   gMinuit->SetErrorDef(4); //note 4 and not 2!
   TGraph *gr2 = (TGraph*)gMinuit->Contour(80,0,2);
   gr2->SetFillColor(42);
   gr2->Draw("alf");
   //Get contour for parameter 0 versus parameter 2 for ERRDEF=1  
   gMinuit->SetErrorDef(1);
   TGraph *gr1 = (TGraph*)gMinuit->Contour(80,0,2);
   gr1->SetFillColor(38);
   gr1->Draw("lf");
}
