/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooBlindTools.h,v 1.10 2007/05/11 10:15:52 verkerke Exp $
 * Authors:                                                                  *
 *   AR, Aaron Roodman, Stanford University, roodman@slac.stanford.edu       *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_BLIND_TOOLS
#define ROO_BLIND_TOOLS

//      ---------------------
//       -- Class Interface --
//      ---------------------

#include "Rtypes.h"
#include "TString.h"

class RooBlindTools {

//--------------------
// Instance Members --
//--------------------

public:

  enum blindMode {
    full=0,   // blind both parameters and data
    dataonly   // blind data only

  };

public:

  // Constructors
  RooBlindTools() {} ;

  RooBlindTools
  (const char *stSeed,   // blinding seed string
   blindMode Mode=full,   // blinding mode
   Double_t centralValue=0.0,     // Central value for Precision measurements
   Double_t sigmaOffset=1.0,      // range for Offset
   bool s2bMode=false          // Use sin2beta modifications?
   );

  // Copy
  RooBlindTools(const RooBlindTools& );

  // Destructor
  virtual ~RooBlindTools( );

  // Operators

  Double_t Randomizer(const char *StringAlphabet) const;

  Double_t PseudoRandom(Int_t Seed) const;

  Double_t MakeOffset(const char *StringAlphabet) const;

  Double_t MakeGaussianOffset(const char *StringAlphabet) const;

  Double_t MakeSignFlip(const char *StringAlphabet) const;

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

  Double_t HidePrecision(Double_t Precision) const;

  Double_t UnHidePrecision(Double_t PrecisionPrime) const;

  Double_t HideOffset( Double_t Precision ) const;

  Double_t UnHideOffset( Double_t PrecisionBlind ) const;

  Double_t HideUniform( Double_t Precision ) const;

  Double_t UnHideUniform( Double_t PrecisionBlind ) const;


  const char *stSeed()const {return _stSeed;}

  const blindMode& mode()const {return _mode;}

  void setMode(blindMode bmode) {_mode=bmode;}

  Double_t getPrecisionCentralValue() const {return _PrecisionCentralValue;}

  Double_t getPrecisionOffsetScale() const {return _PrecisionOffsetScale;}

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
  Double_t _PrecisionSignFlip;
  Double_t _PrecisionOffsetScale;
  Double_t _PrecisionOffset;
  Double_t _PrecisionUniform;
  Double_t _PrecisionCentralValue;
  blindMode _mode;
  bool   _s2bMode ;

  // setup data members from string seed
  void setup(const char *stSeed);

protected:

    // Helper functions
  ClassDef(RooBlindTools,1) // Root implementation of BlindTools
};

#endif
