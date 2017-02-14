// @(#)root/pythia6:$Id$
// Author: Christian Holm Christensen   22/04/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPythia6Decayer
#define ROOT_TPythia6Decayer

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// TPythia6Decayer                                                           //
//                                                                           //
// This implements the TVirtualMCDecayer interface.  The TPythia6            //
// singleton object is used to decay particles.  Note, that since this       //
// class modifies common blocks (global variables) it is defined as a        //
// singleton.                                                                //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TVirtualMCDecayer.h"
#include "TString.h"
#include "TArrayF.h"


class TPythia6Decayer : public TVirtualMCDecayer {

public:
   // enum of decay mode types
   enum EDecayType {
      kSemiElectronic,
      kDiElectron,
      kSemiMuonic,
      kDiMuon,
      kBJpsiDiMuon,
      kBJpsiDiElectron,
      kBPsiPrimeDiMuon,
      kBPsiPrimeDiElectron,
      kPiToMu,
      kKaToMu,
      kNoDecay,
      kHadronicD,
      kOmega,
      kPhiKK,
      kAll,
      kNoDecayHeavy,
      kHardMuons,
      kBJpsi,
      kWToMuon,
      kWToCharm,
      kWToCharmToMuon,
      kZDiMuon,
      kMaxDecay
   };

protected:
   TString    fDecayTableFile; // File to read decay table from
   EDecayType fDecay;          // Forced decay mode
   TArrayF    fBraPart;        //! Branching ratios

   static TPythia6Decayer *fgInstance;

   // Extra functions
   void ForceHadronicD();
   void ForceOmega();
   Int_t CountProducts(Int_t channel, Int_t particle);

public:
   TPythia6Decayer();
   virtual ~TPythia6Decayer() { }
   virtual void    Init();
   virtual void    Decay(Int_t idpart, TLorentzVector* p);
   virtual Int_t   ImportParticles(TClonesArray *particles);
   virtual void    SetForceDecay(Int_t type);
   virtual void    ForceDecay();
   void ForceParticleDecay(Int_t particle, Int_t* products,
                           Int_t* mult, Int_t npart);
   void ForceParticleDecay(Int_t particle, Int_t product, Int_t mult);
   virtual Float_t GetPartialBranchingRatio(Int_t ipart);
   virtual Float_t GetLifetime(Int_t kf);
   virtual void    ReadDecayTable();
   // Extension member functions
   virtual void    SetDecayTableFile(const char* name);
   virtual void    WriteDecayTable();
   virtual void    SetForceDecay(EDecayType type) { fDecay = type; }

   static  TPythia6Decayer *Instance();

   ClassDef(TPythia6Decayer,1) // Particle Decayer Base Class
};

inline void TPythia6Decayer::SetDecayTableFile(const char *name)
{
   fDecayTableFile = name;
}

#endif
