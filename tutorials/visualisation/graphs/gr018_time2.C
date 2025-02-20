/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// \preview Example of TGraphTime showing how the class could be used to visualize
/// a set of particles with their time stamp in a MonteCarlo program.
///
/// See the [TGraphTime documentation](https://root.cern/doc/master/classTGraphTime.html)
///
/// \macro_image
/// \macro_code
/// \author Rene Brun

#include "TRandom3.h"
#include "TMath.h"
#include "TMarker.h"
#include "TPaveLabel.h"
#include "TArrow.h"
#include "TGraphTime.h"
#include <vector>

void gr018_time2(Int_t nsteps = 200, Int_t np = 5000)
{
   if (np > 5000) np = 5000;
   std::vector<Int_t> color(np);
   std::vector<Double_t> cosphi(np), sinphi(np), speed(np);
   TRandom3 r;
   Double_t xmin = 0, xmax = 10, ymin = -10, ymax = 10;
   TGraphTime *g = new TGraphTime(nsteps,xmin,ymin,xmax,ymax);
   g->SetTitle("TGraphTime demo 2;X;Y");
   Double_t fact = xmax/Double_t(nsteps);
   for (Int_t i = 0; i < np; i++) { // calculate some object parameters
      speed[i] = r.Uniform(0.5, 1);
      Double_t phi = r.Gaus(0, TMath::Pi() / 6.);
      cosphi[i] = fact * speed[i] * TMath::Cos(phi);
      sinphi[i] = fact * speed[i] * TMath::Sin(phi);
      Double_t rc = r.Rndm();
      color[i] = kRed;
      if (rc > 0.3) color[i] = kBlue;
      if (rc > 0.7) color[i] = kYellow;
   }
   for (Int_t s = 0; s < nsteps; s++) { // fill the TGraphTime step by step
      for (Int_t i = 0; i < np; i++) {
         Double_t xx = s*cosphi[i];
         if (xx < xmin) continue;
         Double_t yy = s*sinphi[i];
         TMarker *m = new TMarker(xx,yy,25);
         m->SetMarkerColor(color[i]);
         m->SetMarkerSize(1.5 -s/(speed[i]*nsteps));
         g->Add(m, s);
      }
      g->Add(new TPaveLabel(.70,.92,.98,.99,TString::Format("shower at %5.3f nsec",3.*s/nsteps),"brNDC"),s);
   }

   g->Draw();

   // save object as animated gif
   // g->SaveAnimatedGif("gr18_time2.gif");

   // start animation, can be stopped with g->Animate(kFALSE);
   // g->Animate();
}



