// @(#)root/eg:$Id$
// Author: Ola Nordmann   29/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAttParticle                                                         //
//                                                                      //
// Particle definition, based on GEANT3 particle definition             //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TAttParticle
#define ROOT_TAttParticle

#include "TNamed.h"
#include "TAttLine.h"

class THashList;

class TAttParticle  : public TNamed {
private:
   Double_t     fPDGMass;         //Mass of the particle in GeV/c2
   Bool_t       fPDGStable;       //Logical indicator, if TRUE the particle can not decay
   Double_t     fPDGDecayWidth;   //Life time of the particle in sec.
   Double_t     fPDGCharge;       //Charge of the particle in units of e
   TString      fParticleType;    //Text indicator for the particle family
   Int_t        fMCnumberOfPDG;   //PDG MC number followed by
                                  //http://pdg.lbl.gov/rpp/mcdata/all.mc
   Double_t     fEnergyCut;       //Lower energy cut off, the default is 10 keV
   Double_t     fEnergyLimit;     //High energy cut off, the default is 10 TeV
   Double_t     fGranularity;     //Granularity of the fLogEScale

public:
   TAttParticle();
   TAttParticle(const char *name, const char *title,
                Double_t Mass, Bool_t Stable,
                Double_t DecayWidth, Double_t Charge, const char *Type,
                Int_t MCnumber, Int_t granularity=90,
                Double_t LowerCutOff=1.e-5, Double_t HighCutOff=1.e4);
   virtual ~TAttParticle();
   static  THashList     *fgList;
   static  Int_t          ConvertISAtoPDG(Int_t isaNumber);
   static  void           DefinePDG();
   virtual Double_t       GetCharge() const { return fPDGCharge; }
   virtual Double_t       GetEnergyCut() const { return fEnergyCut; }
   virtual Double_t       GetEnergyLimit() const { return fEnergyLimit; }
   virtual Double_t       GetGranularity() const { return fGranularity; }
   virtual Double_t       GetDecayWidth() const { return fPDGDecayWidth; }
   virtual Double_t       GetMass() const { return fPDGMass; }
   virtual Int_t          GetMCNumber() const { return fMCnumberOfPDG; }
   static  TAttParticle  *GetParticle(const char *name);
   static  TAttParticle  *GetParticle(Int_t mcnumber);
   virtual const char    *GetParticleType() const { return fParticleType.Data(); }
   virtual Bool_t         GetStable() const { return fPDGStable; }
   virtual void           Print(Option_t *option="") const ;
   virtual Double_t       SampleMass() const ;
   virtual Double_t       SampleMass(Double_t widthcut) const ;

   ClassDef(TAttParticle,1)  //Particle definition
};

#endif
