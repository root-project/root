// This macro gets in memory an histogram from a root file
// and fits a user defined function.
// Note that a user defined function must always be defined
// as in this example:
//  - first parameter: array of variables (in this example only 1-dimension)
//  - second parameter: array of parameters
// Note also that in case of user defined functions, one must set
// an initial value for each parameter.

Double_t fitf(Double_t *x, Double_t *par)
{
   Double_t arg = 0;
   if (par[2] != 0) arg = (x[0] - par[1])/par[2];

   Double_t fitval = par[0]*TMath::Exp(-0.5*arg*arg);
   return fitval;
}
void myfit()
{
   TFile *f = new TFile("hsimple.root");

   TCanvas *c1 = new TCanvas("c1","the fit canvas",500,400);

   TH1F *hpx = (TH1F*)f->Get("hpx");

// Creates a Root function based on function fitf above
   TF1 *func = new TF1("fitf",fitf,-2,2,3);

// Sets initial values and parameter names
   func->SetParameters(100,0,1);
   func->SetParNames("Constant","Mean_value","Sigma");

// Fit histogram in range defined by function
   hpx->Fit(func,"r");

// Gets integral of function between fit limits
   printf("Integral of function = %g\n",func->Integral(-2,2));
}
