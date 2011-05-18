// @(#)root/eg:$Id$
// Author: Pasha Murat   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef TParticlePDG_hh
#define TParticlePDG_hh

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TDecayChannel;

class TParticlePDG : public TNamed {
public:
//------------------------------------------------------------------------------
//     data members
//------------------------------------------------------------------------------
protected:
  Int_t            fPdgCode;                    // PDG code of the particle
  Double_t         fMass;                       // particle mass in GeV
  Double_t         fCharge;                     // charge in units of |e|/3
  Double_t         fLifetime;                   // proper lifetime in seconds
  Double_t         fWidth;                      // total width in GeV
  Int_t            fParity;
  Double_t         fSpin;
  Double_t         fIsospin;                    // isospin
  Double_t         fI3;                         // i3
  Int_t            fStrangeness;                // flavours are defined if i3 != -1
  Int_t            fCharm;                      // 1 or -1 for C-particles,
                                                // 0 for others
  Int_t            fBeauty;                     //
  Int_t            fTop;                        //
  Int_t            fY;                          // X,Y: quantum numbers for the 4-th generation
  Int_t            fX;                          //
  Int_t            fStable;                     // 1 if stable, 0 otherwise
                                        
  TObjArray*       fDecayList;                  // array of decay channels
                                
  TString          fParticleClass;              // lepton, meson etc

  Int_t            fTrackingCode;               // G3 tracking code of the particle
  TParticlePDG*    fAntiParticle;               // pointer to antiparticle
//------------------------------------------------------------------------------
// functions
//------------------------------------------------------------------------------
   TParticlePDG(const TParticlePDG&); 
   TParticlePDG& operator=(const TParticlePDG&);

public:
   // ****** constructors  and destructor
   TParticlePDG();
   TParticlePDG(int pdg_code);
   TParticlePDG(const char* Name, const char* Title, Double_t Mass,
                Bool_t Stable, Double_t Width, Double_t Charge,
                const char* ParticleClass, Int_t PdgCode, Int_t Anti, 
                Int_t TrackingCode);

   virtual ~TParticlePDG();
   // ****** access methods
  
   Int_t           PdgCode      () const { return fPdgCode; }
   Double_t        Mass         () const { return fMass; }
   Double_t        Charge       () const { return fCharge; } //charge in units of |e|/3
   Double_t        Lifetime     () const { return fLifetime; }
   Double_t        Width        () const { return fWidth; }
   Int_t           Parity       () const { return fParity; }
   Double_t        Spin         () const { return fSpin; }
   Double_t        Isospin      () const { return fIsospin; }
   Double_t        I3           () const { return fI3; }
   Int_t           Strangeness  () const { return fStrangeness; }
   Int_t           Charm        () const { return fCharm; }
   Int_t           Beauty       () const { return fBeauty; }
   Int_t           Top          () const { return fTop; }
   Int_t           X            () const { return fX; }
   Int_t           Y            () const { return fY; }
   Int_t           Stable       () const { return fStable; }
   const char*     ParticleClass() const { return fParticleClass.Data(); }

   TObjArray*      DecayList    () { return fDecayList; }

   Int_t   NDecayChannels () const { 
     return (fDecayList) ? fDecayList->GetEntriesFast() : 0;
   }

   Int_t   TrackingCode() const { return fTrackingCode; }

   TDecayChannel* DecayChannel(Int_t i);

   TParticlePDG* AntiParticle() { return fAntiParticle; }

   // ****** modifiers

   void   SetAntiParticle(TParticlePDG* ap) { fAntiParticle = ap; }

   Int_t  AddDecayChannel(Int_t        Type, 
                          Double_t     BranchingRatio, 
                          Int_t        NDaughters, 
                          Int_t*       DaughterPdgCode);

   virtual void  PrintDecayChannel(TDecayChannel* dc, Option_t* opt = "") const; 

   virtual void  Print(Option_t* opt = "") const; // *MENU*

   ClassDef(TParticlePDG,2)  // PDG static particle definition
};

#endif
