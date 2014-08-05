// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

#include <stdlib.h>

#include <TSystem.h>
#include <TRandom.h>
#include <TParticle.h>
#include "MyParticle.h"
#include "TVector3.h"
#include "constants.h"
#include "RootShower.h"
#include "ParticlesDef.h"

//______________________________________________________________________________
//
// MyParticle class implementation
//______________________________________________________________________________

ClassImp(MyParticle)

//______________________________________________________________________________
MyParticle::MyParticle() : TParticle()
{
   // MyParticle constructor.

   // initialize members to zero
   fStatus = 0;
   fNChildren = 0;
   fDecayType = UNDEFINE;
   fLocation = new TVector3();
   fPassed = 0.0;
   fEloss = 0.0;
   fDecayLength = 0.0;
   fChild[0] = 0;
   fChild[1] = 0;
   fChild[2] = 0;
   fChild[3] = 0;
   fChild[4] = 0;
   fChild[5] = 0;
   fTimeOfDecay = 0.0;
   fTracks = new TObjArray(1);
   fNtrack = 0;
}

//______________________________________________________________________________
MyParticle::MyParticle(Int_t id, Int_t pType,Int_t pStat,Int_t pDecayType,const TVector3 &pLocation,
                       const TVector3 &pMomentum,Double_t pPassed, Double_t pDecayLen,
                       Double_t pEnergy)
          : TParticle(pType, pStat, 1, 0, 0, 0, pMomentum.x(), pMomentum.y(),
                        pMomentum.z(), pEnergy, pLocation.x(), pLocation.y(),
                        pLocation.z(), 0.0)
{
   // MyParticle constructor.

   // initialize members with parameters passed in argument
   fId = id;
   fStatus = pStat;
   fNChildren = 0;
   fDecayType = pDecayType;
   fLocation = new TVector3(pLocation);
   fPassed = pPassed;
   fDecayLength = pDecayLen;
   fChild[0] = 0;
   fChild[1] = 0;
   fChild[2] = 0;
   fChild[3] = 0;
   fChild[4] = 0;
   fChild[5] = 0;
   fTimeOfDecay = 0.0;
   fTracks = new TObjArray(1);
   fNtrack = 0;
}

//______________________________________________________________________________
MyParticle::MyParticle(Int_t id, Int_t pType,Int_t pStat,Int_t pDecayType,const TVector3 &pLocation,
                       const TVector3 &pMomentum)
          : TParticle(pType, pStat, 1, 0, 0, 0, pMomentum.x(), pMomentum.y(),
                        pMomentum.z(), 0.0, pLocation.x(), pLocation.y(),
                        pLocation.z(), 0.0)
{
   // MyParticle constructor.

   // initialize members with parameters passed in argument
   fId = id;
   Double_t energy;
   fStatus = pStat;
   fNChildren = 0;
   fDecayType = pDecayType;
   fLocation = new TVector3(pLocation);
   fPassed = 0.0;
   fEloss = 0.0;
   fDecayLength = 0.0;
   fChild[0] = 0;
   fChild[1] = 0;
   fChild[2] = 0;
   fChild[3] = 0;
   fChild[4] = 0;
   fChild[5] = 0;
   if (GetMass() == 0.0)
      energy = pMomentum.Mag();
   else
      energy = TMath::Sqrt((pMomentum * pMomentum)
               +  (GetMass() * GetMass()));
   TParticle::SetMomentum(pMomentum.x(),pMomentum.y(),pMomentum.z(),energy);
   fTimeOfDecay = 0.0;
   fTracks = new TObjArray(1);
   fNtrack = 0;
}

//______________________________________________________________________________
char *MyParticle::GetObjectInfo(Int_t, Int_t) const
{
   // Returns particle information.

   static char info[64];
   sprintf(info,"Particle = %s, E = %1.3e", GetName(), Energy());
   return info;
}

//______________________________________________________________________________
void MyParticle::SetMoment(const TVector3 &mom)
{
   // Set particle momentum with TVector3 members

   Double_t energy;
   if (GetMass() == 0.0)
      energy = mom.Mag();
   else
      energy = TMath::Sqrt((mom * mom) +  (GetMass() * GetMass()));
   SetMomentum(mom.x(), mom.y(), mom.z(), energy);
}

//______________________________________________________________________________
void MyParticle::GenerateTimeOfDecay()
{
   // Generates time of decay for this type of particle.

   Int_t i;
   fTimeOfDecay = (GetPDG()->Lifetime() > 0 && GetPDG()->Lifetime() < 1e8)
                   ? gRandom->Exp(GetPDG()->Lifetime())
                   : GetPDG()->Lifetime();
   if (fTimeOfDecay == 0.0) {
      Int_t my_code = GetPdgCode();
      fTimeOfDecay = 1.0e-20;
      for (i=0;i<total_defs;i++) {
         if (particle_def[i].code == my_code) {
            fTimeOfDecay = particle_def[i].lifetime;
            break;
         }
      }
      if (fTimeOfDecay <= 1.0e-20) {
         fTimeOfDecay = gRandom->Exp(1.0e-8);
      }
      else if (fTimeOfDecay > 0 && fTimeOfDecay < 1e8)
         fTimeOfDecay = gRandom->Exp(fTimeOfDecay);
   }
   // Apply Lorentz transformation for time dilatation
   fTimeOfDecay /= TMath::Sqrt(1.0 - (0.996 * 0.996));
}

//______________________________________________________________________________
const Char_t *MyParticle::GetName() const
{
   // Get name of particle with its PDG number, or Unknown if
   // particle name is not defined into ParticlesDef.h

   Int_t i;
   Int_t my_code = GetPdgCode();

   for (i=0;i<total_defs;i++) {
      if (particle_def[i].code == my_code)
         return(particle_def[i].name);
   }
   return ("Unknown");
}

//______________________________________________________________________________
TPolyLine3D *MyParticle::AddTrack(const TVector3 &pos, Int_t color)
{
   // Add a new track to the list of tracks for this particle.

   TPolyLine3D *poly;
   fTracks->Add(new TPolyLine3D());
   fNtrack = fTracks->GetLast();
   poly = (TPolyLine3D *)fTracks->At(fNtrack);
   poly->SetPoint(0, pos.x(), pos.y(), pos.z());
   poly->SetLineColor(color);
   return poly;
}

//______________________________________________________________________________
TPolyLine3D *MyParticle::AddTrack(Double_t x, Double_t y, Double_t z, Int_t col)
{
   // Add a new track to the list of tracks for this particle.

   TPolyLine3D *poly;
   fTracks->Add(new TPolyLine3D());
   fNtrack = fTracks->GetLast();
   poly = (TPolyLine3D *)fTracks->At(fNtrack);
   poly->SetPoint(0, x, y, z);
   poly->SetLineColor(col);
   return poly;
}

//______________________________________________________________________________
void MyParticle::SetNextPoint(Int_t color)
{
   // Set next polyline point for the current track if the color did not change
   // or add a new polyline with the new color.

   TPolyLine3D *poly;
   poly = (TPolyLine3D *)fTracks->At(fNtrack);
   poly->SetNextPoint(fLocation->x(), fLocation->y(), fLocation->z());
   if (color != poly->GetLineColor())
      AddTrack(fLocation->x(), fLocation->y(), fLocation->z(), color);
}

//______________________________________________________________________________
MyParticle::~MyParticle()
{
   // destructor

   fTracks->Delete();
   delete fTracks;
   delete fLocation;
}

//______________________________________________________________________________
void MyParticle::HighLight()
{
   // HighLight this particle track in the display.

   Int_t i;
   TPolyLine3D *poly;
   TIter next(fTracks);
   for (i=0;i<5;i++) {
      next.Reset();
      while ((poly = (TPolyLine3D *)next())) {
         poly->SetLineColor(poly->GetLineColor() + 25);
         poly->SetLineWidth(3);
         gSystem->ProcessEvents();
      }
      gRootShower->UpdateDisplay();
      gSystem->ProcessEvents();
      next.Reset();
      while ((poly = (TPolyLine3D *)next())) {
         poly->SetLineColor(poly->GetLineColor() - 25);
         poly->SetLineWidth(1);
         gSystem->ProcessEvents();
      }
      gRootShower->UpdateDisplay();
      gSystem->ProcessEvents();
   }
   gRootShower->UpdateDisplay();
}


