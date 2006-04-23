// @(#)root/pythia6:$Name:  $:$Id: TPythia6.h,v 1.8 2006/01/24 05:59:27 brun Exp $
// Author: Christian Holm Christensen   22/04/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TPythia6Decayer
#define ROOT_TPythia6Decayer
////////////////////////////////////////////////////////////////////////////////
//                                                                            
// TPythia6Decayer                                                                   
//
// This implements the TVirtualMCDecayer interface.  The TPythia6
// singleton object is used to decay particles.  Note, that since this
// class modifies common blocks (global variables) it is defined as a
// singleton.   
//
//////////////////////////////////////////////////////////////////////
#ifndef ROOT_TVirtualMCDecayer
# include <TVirtualMCDecayer.h>
#endif
#ifndef ROOT_TString
# include <TString.h>
#endif
#ifndef ROOT_TArrayF
# include <TArrayF.h>
#endif


class TPythia6Decayer : public TVirtualMCDecayer
{
public:
   // enum of decay mode types 
   enum Decay_t {
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

   virtual ~TPythia6Decayer(){;}
   static  TPythia6Decayer* Instance();
   virtual void    Init();
   virtual void    Decay(Int_t idpart, TLorentzVector* p);
   virtual Int_t   ImportParticles(TClonesArray *particles);
   virtual void    SetForceDecay(Int_t type);
   virtual void    ForceDecay();
   virtual Float_t GetPartialBranchingRatio(Int_t ipart);
   virtual Float_t GetLifetime(Int_t kf);
   virtual void    ReadDecayTable();
   // Extension member functions 
   virtual void    SetDecayTableFile(const char* name);
   virtual void    WriteDecayTable();
   virtual void    SetForceDecay(Decay_t type) { fDecay = type; }
protected:
   TPythia6Decayer();
   static TPythia6Decayer* fgInstance;
   TString fDecayTableFile; // File to read decay table from 
   Decay_t fDecay;          // Forced decay mode
   TArrayF fBraPart;        //! Branching ratios

   // Extra functions 
   void ForceParticleDecay(Int_t particle, Int_t* products, 
			    Int_t* mult, Int_t npart);
   void ForceParticleDecay(Int_t particle, Int_t product, Int_t mult);
   void ForceHadronicD();
   void ForceOmega();
   Int_t CountProducts(Int_t channel, Int_t particle);
   
   ClassDef(TPythia6Decayer,1) // Particle Decayer Base Class
};

inline
void
TPythia6Decayer::SetDecayTableFile(const char* name)
{
   fDecayTableFile = name;
}

#endif
// Local Variables:
//   mode: C++ 
//   c-basic-offset: 3
// End:
//
// EOF
//
