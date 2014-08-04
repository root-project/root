// Illustrate how to fit excluding points in a given range
// Author: Rene Brun
#include "TH1.h"
#include "TF1.h"
#include "TList.h"

Bool_t reject;
Double_t fline(Double_t *x, Double_t *par)
{
    if (reject && x[0] > 2.5 && x[0] < 3.5) {
      TF1::RejectPoint();
      return 0;
   }
   return par[0] + par[1]*x[0];
}

void fitExclude() {
   //Create a source function
   TF1 *f1 = new TF1("f1","[0] +[1]*x +gaus(2)",0,5);
   f1->SetParameters(6,-1,5,3,0.2);
   // create and fill histogram according to the source function
   TH1F *h = new TH1F("h","background + signal",100,0,5);
   h->FillRandom("f1",2000);
   TF1 *fl = new TF1("fl",fline,0,5,2);
   fl->SetParameters(2,-1);
   //fit only the linear background excluding the signal area
   reject = kTRUE;
   h->Fit(fl,"0");
   reject = kFALSE;
   //store 2 separate functions for visualization
   TF1 *fleft = new TF1("fleft",fline,0,2.5,2);
   fleft->SetParameters(fl->GetParameters());
   h->GetListOfFunctions()->Add(fleft);
   gROOT->GetListOfFunctions()->Remove(fleft);
   TF1 *fright = new TF1("fright",fline,3.5,5,2);
   fright->SetParameters(fl->GetParameters());
   h->GetListOfFunctions()->Add(fright);
   gROOT->GetListOfFunctions()->Remove(fright);
   h->Draw();
}

