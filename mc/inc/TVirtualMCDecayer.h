// Author: Andreas Morsch  13/04/2002
   
#ifndef ROOT_TVirtualMCDecayer
#define ROOT_TVirtualMCDecayer
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

// Abstract base class for particle decays.
// Clients are the transport code and the primary particle generators

#include "TObject.h"
class TClonesArray;
class TLorentzVector;

typedef enum
{ kSemiElectronic, kDiElectron, kSemiMuonic, kDiMuon,
  kBJpsiDiMuon, kBJpsiDiElectron, 
  kBPsiPrimeDiMuon, kBPsiPrimeDiElectron, kPiToMu, kKaToMu, kNoDecay, 
  kHadronicD, kOmega, kAll}
Decay_t;

class TVirtualMCDecayer : public TObject {
 public:
//
    virtual ~TVirtualMCDecayer(){;}
    virtual void    Init()                                     =0;
    virtual void    Decay(Int_t idpart, TLorentzVector* p)     =0;
    virtual Int_t   ImportParticles(TClonesArray *particles)   =0;
    virtual void    SetForceDecay(Decay_t type)                =0;
    virtual void    ForceDecay()                               =0;
    virtual Float_t GetPartialBranchingRatio(Int_t ipart)      =0;
    virtual Float_t GetLifetime(Int_t kf)                      =0;
    virtual void    ReadDecayTable()                           =0;
    ClassDef(TVirtualMCDecayer,1) // Particle Decayer Base Class
};
#endif //ROOT_TVirtualMCDecayer







