/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooBlindTools.rdl,v 1.1 2001/05/07 06:14:53 verkerke Exp $
 * Authors:
 *   AR, Aaron Roodman, Stanford University, roodman@slac.stanford.edu 
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   13-Dec-1999 AR Created initial version
 *   28-Mar-2001 WV Migrate to RooFitCore and rename from RooBlind to RooBlindTools
 *                  Convert basic types to ROOT names
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#ifndef ROO_BLIND_TOOLS
#define ROO_BLIND_TOOLS

//		---------------------
// 		-- Class Interface --
//		---------------------

#include "Rtypes.h" 
#include "TString.h"

class RooBlindTools{

//--------------------
// Instance Members --
//--------------------

public:

  enum blindMode {
    full=0,	// blind both parameters and data
    dataonly	// blind data only
    
  };

public:

  // Constructors
  RooBlindTools
  (const char *stSeed,	// blinding seed string
   blindMode Mode=full	// blinding mode
   );

  // Copy
  RooBlindTools(const RooBlindTools& );
  
  // Destructor
  virtual ~RooBlindTools( );

  // Operators

  Double_t Randomizer(char *StringAlphabet) const;

  Double_t PseudoRandom(Int_t Seed) const;

  Double_t MakeOffset(char *StringAlphabet) const;

  Double_t MakeGaussianOffset(char *StringAlphabet) const;

  Double_t MakeSignFlip(char *StringAlphabet) const;

  Int_t SignOfTag(Double_t STag) const;
  
  Double_t HideDeltaZ(Double_t DeltaZ, Double_t STag) const;

  Double_t HiDelZPdG(Double_t DeltaZ, Double_t STag, Double_t PdG) const;

  Double_t UnHideDeltaZ(Double_t DeltaZPrime, Double_t STag) const;

  Double_t UnHiDelZPdG(Double_t DeltaZPrime, Double_t STag, Double_t PdG) const;

  Double_t HideAsym(Double_t Asym) const;

  Double_t HiAsPdG(Double_t Asym, Double_t PdG) const;

  Double_t UnHideAsym(Double_t AsymPrime) const;

  Double_t UnHiAsPdG(Double_t AsymPrime, Double_t PdG) const;

  Double_t HideDeltaM(Double_t DeltaM) const;

  Double_t UnHideDeltaM(Double_t DeltaMPrime) const;

  Double_t MysteryPhase() const;

  Double_t RandomizeTag(Double_t STag, Int_t EventNumber) const;

  const char *stSeed()const {return _stSeed;}

  const blindMode& mode()const {return _mode;}

  void setMode(blindMode mode) {_mode=mode;}

private:

  // Friends
  // Data members

  TString _stSeed ; 
  Double_t _DeltaZOffset;
  Double_t _DeltaZSignFlip;
  Double_t _DeltaZScale;
  Double_t _AsymOffset;
  Double_t _AsymSignFlip;
  Double_t _DeltaMScale;
  Double_t _DeltaMOffset;
  Double_t _MysteryPhase;
  Double_t _STagConstant;
  blindMode _mode;
  
  // setup data members from string seed  
  void setup(const char *stSeed);

protected:

    // Helper functions
  ClassDef(RooBlindTools,1) // Root implementation of BlindTools
};

#endif 


