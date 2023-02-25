// @(#)root/pythia8:$Name$:$Id$
// Author: Andreas Morsch   04/07/2008

/* Copyright(c) 1998-2008, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#ifndef TPYTHIA8DECAYER_H
#define TPYTHIA8DECAYER_H

#include "TVirtualMCDecayer.h"

class TClonesArrray;
class TLorentzVector;
class TPythia8;

class TPythia8Decayer : public TVirtualMCDecayer {
public:
   TPythia8Decayer();
   ~TPythia8Decayer() override{;}
   void    Init() override;
   void    Decay(Int_t pdg, TLorentzVector* p) override;
   Int_t   ImportParticles(TClonesArray *particles) override;
   void    SetForceDecay(Int_t type) override;
   void    ForceDecay() override;
   Float_t GetPartialBranchingRatio(Int_t ipart) override;
   Float_t GetLifetime(Int_t kf) override;
   void    ReadDecayTable() override;

   virtual void    SetDebugLevel(Int_t debug) {fDebug = debug;}
protected:
   void AppendParticle(Int_t pdg, TLorentzVector* p);
   void ClearEvent();
private:
   TPythia8* fPythia8;          // Pointer to pythia8
   Int_t     fDebug;            // Debug level

   ClassDefOverride(TPythia8Decayer, 1) // Particle Decayer using Pythia8

};
#endif







