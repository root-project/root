#include "TRandom.h"
#include "TCanvas.h"
#include "TView.h"
#include "TPolyMarker3D.h"
   
void spheres(Int_t nspheres=20, Int_t npoints=100000) {
   // generate random points uniformly distributed over the surface
   // of spheres generated at random positions.

   TCanvas *c1 = new TCanvas("c1","spheres",600,600);
   c1->SetFillColor(kBlack);
   TView *view = new TView(1);
   Double_t k=0.8;
   view->SetRange( -k, -k, -k, k, k, k);
   
   //generate sphere coordinates and radius
   TRandom r;
   if (nspheres <=0) nspheres = 10;
   Double_t *xs = new Double_t[nspheres];
   Double_t *ys = new Double_t[nspheres];
   Double_t *zs = new Double_t[nspheres];
   Double_t *rs = new Double_t[nspheres];
   Int_t i;
   for (i=0;i<nspheres;i++) {
      xs[i] = r.Uniform(-1,1);
      ys[i] = r.Uniform(-1,1);
      zs[i] = r.Uniform(-1,1);
      rs[i] = r.Uniform(0.05,0.25);
   }

   //generate points
   TPolyMarker3D *pm = new TPolyMarker3D(npoints);
   pm->SetMarkerColor(kRed);
   Double_t x,y,z;
   for (Int_t j=0;j<npoints;j++) {
       i = (Int_t)r.Uniform(0,nspheres);  //choose sphere
       r.Sphere(x,y,z,rs[i]);
       pm->SetPoint(j,xs[i]+x, ys[i]+y,zs[i]+z);
   }
   
   delete [] xs;
   delete [] ys;
   delete [] zs;
   delete [] rs;
    
   pm->Draw();
}
