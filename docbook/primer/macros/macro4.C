// Create, Draw and fit a TGraph2DErrors
void macro4(){    
   gStyle->SetPalette(1);
   const double e = 0.3;
   const int nd = 500;
    
   TRandom3 my_random_generator;
   TF2 *f2 = new TF2("f2",
                    "1000*(([0]*sin(x)/x)*([1]*sin(y)/y))+200",
                    -6,6,-6,6);
   f2->SetParameters(1,1);
   TGraph2DErrors *dte = new TGraph2DErrors(nd);
   // Fill the 2D graph
   double rnd, x, y, z, ex, ey, ez;
   for (Int_t i=0; i<nd; i++) {
      f2->GetRandom2(x,y);      
      // A random number in [-e,e]
      rnd = my_random_generator.Uniform(-e,e);
      z = f2->Eval(x,y)*(1+rnd);
      dte->SetPoint(i,x,y,z);
      ex = 0.05*my_random_generator.Uniform();
      ey = 0.05*my_random_generator.Uniform();
      ez = TMath::Abs(z*rnd);
      dte->SetPointError(i,ex,ey,ez);
   }
   // Fit function to generated data   
   f2->SetParameters(0.7,1.5);  // set initial values for fit
   f2->SetTitle("Fitted 2D function");
   dte->Fit(f2);
   // Plot the result    
   TCanvas *c1 = new TCanvas();
   f2->Draw("Surf1");
   dte->Draw("P0 Same");
   // Make the x and y projections
   TCanvas* c_p= new TCanvas("ProjCan",
                             "The Projections",1000,400);
   c_p->Divide(2,1);
   c_p->cd(1);
   dte->Project("x")->Draw();
   c_p->cd(2);
   dte->Project("y")->Draw();
}
