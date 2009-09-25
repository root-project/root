// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstrates usage of class TEveStraightLineSet.

TEveStraightLineSet* lineset(Int_t nlines = 40, Int_t nmarkers = 4) 
{
   TEveManager::Create();

   TRandom r(0);
   Float_t s = 100;

   TEveStraightLineSet* ls = new TEveStraightLineSet();

   for(Int_t i = 0; i<nlines; i++)
   {
      ls->AddLine( r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s),
                   r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
      // add random number of markers
      Int_t nm = Int_t(nmarkers* r.Rndm());
      for(Int_t m = 0; m < nm; m++) {
         ls->AddMarker(i, r.Rndm());
      }
   }

   ls->SetMarkerSize(1.5);
   ls->SetMarkerStyle(4);

   gEve->AddElement(ls);
   gEve->Redraw3D();

   return ls;
}

TEveStraightLineSet* lineset_2d(Int_t nlines = 40, Int_t nmarkers = 4) 
{
   TEveManager::Create();

   TRandom r(0);
   Float_t s = 100;

   TEveStraightLineSet* ls = new TEveStraightLineSet();

   for(Int_t i = 0; i<nlines; i++)
   {
      ls->AddLine( r.Uniform(-s,s), r.Uniform(-s,s), 0,
                   r.Uniform(-s,s), r.Uniform(-s,s), 0);
      // add random number of markers
      Int_t nm = Int_t(nmarkers* r.Rndm());
      for(Int_t m = 0; m < nm; m++) {
         ls->AddMarker(i, r.Rndm());
      }
   }

   ls->SetMarkerSize(1.5);
   ls->SetMarkerStyle(4);

   gEve->AddElement(ls);
   gEve->Redraw3D();

   return ls;
}
