/// \file
/// \ingroup tutorial_fit
/// \notebook -js
/// Example illustrating how to fit a 2-d histogram of type y=f(x)
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Rene Brun

#include <TCanvas.h>
#include <TCutG.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TRandom.h>

void fit2d()
{
   // generate a 2-d histogram using a TCutG
   const int n = 6;
   float x[n] = {0.092,0.83,0.94,0.81,0.12,0.1};
   float y[n] = {0.71,9.4,9,8,0.3,0.71};
   TCutG *cut = new TCutG("cut",n,x,y);
   TH2F *h2 = new TH2F("h2","h2",40,0,1,40,0,10);
   float u,v;
   for (int i=0;i<100000;i++) {
      u = gRandom->Rndm();
      v = 10*gRandom->Rndm();
      if (cut->IsInside(u,v)) h2->Fill(u,v);
   }
   TCanvas *c1 = new TCanvas("c1","show profile",600,900);
   c1->Divide(1,2);
   c1->cd(1);
   h2->Draw();
   c1->cd(2);

   //use a TProfile to convert the 2-d to 1-d problem
   TProfile *prof = h2->ProfileX();
   prof->Fit("pol1");
}

