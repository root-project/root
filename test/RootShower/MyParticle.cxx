// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

#include <stdlib.h>

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
    // MyParticle constructor
    // initialize members to zero
    fstatus = 0;
    fnChildren = 0;
    fdecay_type = UNDEFINE;
    flocation = new TVector3();
    fpassed = 0.0;
    fEloss = 0.0;
    fdecay_length = 0.0;
    fChild[0] = 0;
    fChild[1] = 0;
    fChild[2] = 0;
    fChild[3] = 0;
    fChild[4] = 0;
    fChild[5] = 0;
    ftimeOfDecay = 0.0;
}

//______________________________________________________________________________
MyParticle::MyParticle(Int_t id, Int_t pType,Int_t pStat,Int_t pDecayType,const TVector3 &pLocation,
                       const TVector3 &pMomentum,Double_t pPassed, Double_t pDecayLen, 
                       Double_t pEnergy)
          : TParticle(pType, pStat, 1, 0, 0, 0, pMomentum.x(), pMomentum.y(), 
                        pMomentum.z(), pEnergy, pLocation.x(), pLocation.y(), 
                        pLocation.z(), 0.0)
{
    // MyParticle constructor
    // initialize members with parameters passed in argument
    fId = id;
    fstatus = pStat;
    fnChildren = 0;
    fdecay_type = pDecayType;
    flocation = new TVector3(pLocation);
    fpassed = pPassed;
    fdecay_length = pDecayLen;
    fChild[0] = 0;
    fChild[1] = 0;
    fChild[2] = 0;
    fChild[3] = 0;
    fChild[4] = 0;
    fChild[5] = 0;
    ftimeOfDecay = 0.0;
}

//______________________________________________________________________________
MyParticle::MyParticle(Int_t id, Int_t pType,Int_t pStat,Int_t pDecayType,const TVector3 &pLocation,
                       const TVector3 &pMomentum)
          : TParticle(pType, pStat, 1, 0, 0, 0, pMomentum.x(), pMomentum.y(), 
                        pMomentum.z(), 0.0, pLocation.x(), pLocation.y(),
                        pLocation.z(), 0.0)
{
    // MyParticle constructor
    // initialize members with parameters passed in argument
    fId = id;
    Double_t energy;
    fstatus = pStat;
    fnChildren = 0;
    fdecay_type = pDecayType;
    flocation = new TVector3(pLocation);
    fpassed = 0.0;
    fEloss = 0.0;
    fdecay_length = 0.0;
    fChild[0] = 0;
    fChild[1] = 0;
    fChild[2] = 0;
    fChild[3] = 0;
    fChild[4] = 0;
    fChild[5] = 0;
    if(GetMass() == 0.0)
        energy = pMomentum.Mag();
    else
        energy = sqrt((pMomentum * pMomentum)
                 +  (GetMass() * GetMass()));
    TParticle::SetMomentum(pMomentum.x(),pMomentum.y(),pMomentum.z(),energy);
    ftimeOfDecay = 0.0;
}

//______________________________________________________________________________
char *MyParticle::GetObjectInfo(Int_t, Int_t) const
{
   static char info[64];
   sprintf(info,"Particle = %s, E = %1.3e", GetName(), Energy());
   return info;
}

void MyParticle::SetMoment(const TVector3 &mom)
{
    // Set particle momentum with TVector3 members
    Double_t energy;
    if(GetMass() == 0.0)
        energy = mom.Mag();
    else
        energy = sqrt((mom * mom) +  (GetMass() * GetMass()));
    SetMomentum(mom.x(), mom.y(), mom.z(), energy);
}

void MyParticle::GenerateTimeOfDecay()
{
    // Generates time of decay for this type of particle.
    Int_t i;
    ftimeOfDecay = (GetPDG()->Lifetime() > 0 && GetPDG()->Lifetime() < 1e8)
	  ? gRandom->Exp(GetPDG()->Lifetime())
	  : GetPDG()->Lifetime();
    if(ftimeOfDecay == 0.0) {
        Int_t my_code = GetPdgCode();
        ftimeOfDecay = 1.0e-20;
        for(i=0;i<total_defs;i++) {
            if(particle_def[i].code == my_code) {
                ftimeOfDecay = particle_def[i].lifetime;
                break;
            }
        }
//        ftimeOfDecay *= 1.0e3;
        if (ftimeOfDecay <= 1.0e-20) {
            ftimeOfDecay = gRandom->Exp(1.0e-8);
        }
        else if (ftimeOfDecay > 0 && ftimeOfDecay < 1e8)
            ftimeOfDecay = gRandom->Exp(ftimeOfDecay);
    }
    // Apply Lorentz transformation for time dilatation
    ftimeOfDecay /= TMath::Sqrt(1.0 - (0.996 * 0.996));
}

//______________________________________________________________________________
const Char_t *MyParticle::GetName() const
{
    // get name of particle with its PDG number, or Unknown if
    // particle name is not defined into ParticlesDef.h
    Int_t i;
    Char_t *pdg_name = new Char_t[40];
    Int_t my_code = GetPdgCode();

    sprintf(pdg_name,"Unknown");
    for(i=0;i<total_defs;i++) {
        if(particle_def[i].code == my_code) {
            sprintf(pdg_name,"%s", particle_def[i].name);
            break;
        }
    }
    return pdg_name;
}

//______________________________________________________________________________
MyParticle::~MyParticle()
{
    // destructor
    delete   flocation;
}

