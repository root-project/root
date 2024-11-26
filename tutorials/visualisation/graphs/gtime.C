/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Example of TGraphTime.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

#include "TCanvas.h"
#include "TRandom3.h"
#include "TMath.h"
#include "TText.h"
#include "TArrow.h"
#include "TGraphTime.h"
#include <vector>

void gtime(Int_t nsteps = 500, Int_t np = 100)
{
   if (np > 1000)
      np = 1000;
   std::vector<Int_t> color(np);
   std::vector<Double_t> rr(np), phi(np), dr(np), size(np);
   TRandom3 r;
   Double_t xmin = -10, xmax = 10, ymin = -10, ymax = 10;
   auto g = new TGraphTime(nsteps, xmin, ymin, xmax, ymax);
   g->SetTitle("TGraphTime demo;X domain;Y domain");
   for (Int_t i = 0; i < np; i++) { // calculate some object parameters
      rr[i] = r.Uniform(0.1 * xmax, 0.2 * xmax);
      phi[i] = r.Uniform(0, 2 * TMath::Pi());
      dr[i] = r.Uniform(0, 1) * 0.9 * xmax / Double_t(nsteps);
      Double_t rc = r.Rndm();
      if (rc > 0.7)
         color[i] = kYellow;
      else if (rc > 0.3)
         color[i] = kBlue;
      else
         color[i] = kRed;

      size[i] = r.Uniform(0.5, 6);
   }
   for (Int_t s = 0; s < nsteps; s++) { // fill the TGraphTime step by step
      for (Int_t i = 0; i < np; i++) {
         Double_t newr = rr[i] + dr[i] * s;
         Double_t newsize = 0.2 + size[i] * TMath::Abs(TMath::Sin(newr + 10));
         Double_t newphi = phi[i] + 0.01 * s;
         Double_t xx = newr * TMath::Cos(newphi);
         Double_t yy = newr * TMath::Sin(newphi);
         TMarker *m = new TMarker(xx, yy, 20);
         m->SetMarkerColor(color[i]);
         m->SetMarkerSize(newsize);
         g->Add(m, s);
         if (i == np - 1)
            g->Add(new TArrow(xmin, ymax, xx, yy, 0.02, "-|>"), s);
      }
      g->Add(new TPaveLabel(.90, .92, .98, .97, TString::Format("%d", s + 1), "brNDC"), s);
   }

   g->Draw();

   // save object as animated gif
   // g->SaveAnimatedGif("gtime.gif");

   // save object to a file
   auto f = TFile::Open("gtime.root", "recreate");
   f->WriteObject(g, "g");
   delete f;

   // to view this object in another session do
   //   TFile::Open("gtime.root");
   //   g->Draw();
}
