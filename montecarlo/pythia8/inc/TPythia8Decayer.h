// @(#)root/pythia8:$Name$:$Id$
// Author: Andreas Morsch   04/07/2008

/* Copyright(c) 1998-2008, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

#ifndef TPYTHIA8DECAYER_H
#define TPYTHIA8DECAYER_H

#include <TVirtualMCDecayer.h>

class TClonesArrray;
class TLorentzVector;
class TPythia8;

class TPythia8Decayer : public TVirtualMCDecayer {
public:
   TPythia8Decayer();
   virtual ~TPythia8Decayer(){;}
   virtual void    Init();
   virtual void    Decay(Int_t pdg, TLorentzVector* p);
   virtual Int_t   ImportParticles(TClonesArray *particles);
   virtual void    SetForceDecay(Int_t type);
   virtual void    ForceDecay();
   virtual Float_t GetPartialBranchingRatio(Int_t ipart);
   virtual Float_t GetLifetime(Int_t kf);
   virtual void    ReadDecayTable();

   virtual void    SetDebugLevel(Int_t debug) {fDebug = debug;}
protected:
   void AppendParticle(Int_t pdg, TLorentzVector* p);
   void ClearEvent(); 
private:
   TPythia8* fPythia8;          // Pointer to pythia8
   Int_t     fDebug;            // Debug level
   
   ClassDef(TPythia8Decayer, 1) // Particle Decayer using Pythia8
    
};
#endif







