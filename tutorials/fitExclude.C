// example to illustrate how to fit excluding points in a given range
// Author: Rene Brun
#include "TH1.h"
#include "TF1.h"
Double_t fline(Double_t *x, Double_t *par)
{
    if (x[0] > 2.5 && x[0] < 3.5) {
      TF1::RejectPoint();
      return 0;
   }
   return par[0] + par[1]*x[0];
}

void fitExclude() {
   //Create a source function
   TF1 *f1 = new TF1("f1","[0] +[1]*x +gaus(2)",0,5);
   f1->SetParameters(6,-1,5,3,0.2);
   // create an histogram and fill it according to the source function
   TH1F *h = new TH1F("h","background + signal",100,0,5);
   h->FillRandom("f1",2000);
   TF1 *fline = new TF1("fline",fline,0,5,2);
   fline->SetParameters(2,-1);      
   //we want to fit only the linear background excluding the signal area
   h->Fit("fline","l");
}
   
