// @(#)root/eve:$Id: text_test.C 26717 2008-12-07 22:07:55Z matevz $
// Author: Alja Mrak-Tadel

// Makes some tracks with three different magnetic field types.

#include "TEveTrackPropagator.h"
#include "TEveTrack.h"
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TSystem.h"
#include "TGLViewer.h"
#include "TMath.h"

#include "TEveViewer.h"
#include "TEvePointSet.h"

class GappedField : public TEveMagField
{
public:
   GappedField():TEveMagField(){}
   ~GappedField(){};
   using   TEveMagField::GetField;

   virtual TEveVector GetField(Float_t /*x*/, Float_t /*y*/, Float_t z) const
   {
      if (TMath::Abs(z) < 300) return TEveVector(0, 0, -4);
      if (TMath::Abs(z) < 600) return TEveVector(0, 0, 0);
      return TEveVector(0, 0, 4);
   }
};

//______________________________________________________________________________
TEveTrack* make_track(TEveTrackPropagator* prop, Int_t sign)
{
  // Make track with given propagator.
  // Add to math-marks to test fit.

  TEveRecTrack *rc = new TEveRecTrack();
  rc->fV.Set(0.028558, -0.000918, 3.691919);
  rc->fP.Set(0.767095, -2.400006, -0.313103);
  rc->fSign = sign;


  TEveTrack* track = new TEveTrack(rc, prop);
  track->SetName(Form("Charge %d", sign));
  // daughter 0
  TEvePathMark* pm1 = new TEvePathMark(TEvePathMark::kDaughter);
  pm1->fV.Set(1.479084, -4.370661, 3.119761);
  track->AddPathMark(*pm1);
  // daughter 1
  TEvePathMark* pm2 = new TEvePathMark(TEvePathMark::kDaughter);
  pm2->fV.Set(57.72345, -89.77011, -9.783746);
  track->AddPathMark(*pm2);

  return track;
}


void track(Int_t bCase = 3, Bool_t isRungeKutta = kTRUE)
{
#if defined (__CINT__)
   Error("track.C", "Must be run in compiled mode!");
   return;
#endif

   gSystem->IgnoreSignal(kSigSegmentationViolation, true);
   TEveManager::Create();

   TEveTrackList *list = new TEveTrackList();
   TEveTrackPropagator* prop = list->GetPropagator();
   prop->SetFitDaughters(kFALSE);
   prop->SetMaxZ(1000);

   if (isRungeKutta)
   {
      prop->SetStepper(TEveTrackPropagator::kRungeKutta);
      list->SetName("RK Propagator");
   }
   else
   {
      list->SetName("Heix Propagator");
   }

   TEveTrack *track = 0;
   switch (bCase)
   {
      case 0:
      {
         // B = 0 no difference btween signed and charge particles
         prop->SetMagField(0.);
         list->SetElementName(Form("%s, zeroB", list->GetElementName()));
         track = make_track(prop, 1);
         break;
      }

      case 1:
      {
         // constant B field, const angle
         prop->SetMagFieldObj(new TEveMagFieldConst(0., 0., -3.8));
         list->SetElementName(Form("%s, constB", list->GetElementName()));
         track = make_track(prop, 1);
         break;
      }
      case 2:
      {
         // variable B field, sign change at  R = 200 cm
         prop->SetMagFieldObj(new TEveMagFieldDuo(200, -4.4, 2));
         list->SetElementName(Form("%s, duoB", list->GetElementName()));
         track = make_track(prop, 1);
         break;
      }
      case 3:
      {
         // gapped field
         prop->SetMagFieldObj(new GappedField());
         list->SetElementName(Form("%s, gappedB", list->GetElementName()));

      
         TEveRecTrack *rc = new TEveRecTrack();
         rc->fV.Set(0.028558, -0.000918, 3.691919);
         rc->fP.Set(0.767095, -0.400006, 2.313103);
         rc->fSign = 1;
         track = new TEveTrack(rc, prop);

         TEvePointSet* marker = new TEvePointSet(2);  
         marker->SetElementName("B field break points");
         marker->SetPoint(0, 0., 0., 300.f);
         marker->SetPoint(1, 0., 0., 600.f);
         marker->SetMarkerColor(3);
         gEve->AddElement(marker);
      }
   };
       
   if (isRungeKutta)
      list->SetLineColor(kMagenta);
   else 
      list->SetLineColor(kCyan);

   track->SetLineColor(list->GetLineColor());
 
   gEve->AddElement(track, list);
   gEve->AddElement(list);
   track->MakeTrack();

   TEveViewer* v = gEve->GetDefaultViewer();
   v->GetGLViewer()->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);
   gEve->Redraw3D(1);
}
