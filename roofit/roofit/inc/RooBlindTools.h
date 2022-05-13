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
   double centralValue=0.0,     // Central value for Precision measurements
   double sigmaOffset=1.0,      // range for Offset
   bool s2bMode=false          // Use sin2beta modifications?
   );

  // Copy
  RooBlindTools(const RooBlindTools& );

  // Destructor
  virtual ~RooBlindTools( );

  // Operators

  double Randomizer(const char *StringAlphabet) const;

  double PseudoRandom(Int_t Seed) const;

  double MakeOffset(const char *StringAlphabet) const;

  double MakeGaussianOffset(const char *StringAlphabet) const;

  double MakeSignFlip(const char *StringAlphabet) const;

  Int_t SignOfTag(double STag) const;

  double HideDeltaZ(double DeltaZ, double STag) const;

  double HiDelZPdG(double DeltaZ, double STag, double PdG) const;

  double UnHideDeltaZ(double DeltaZPrime, double STag) const;

  double UnHiDelZPdG(double DeltaZPrime, double STag, double PdG) const;

  double HideAsym(double Asym) const;

  double HiAsPdG(double Asym, double PdG) const;

  double UnHideAsym(double AsymPrime) const;

  double UnHiAsPdG(double AsymPrime, double PdG) const;

  double HideDeltaM(double DeltaM) const;

  double UnHideDeltaM(double DeltaMPrime) const;

  double MysteryPhase() const;

  double RandomizeTag(double STag, Int_t EventNumber) const;

  double HidePrecision(double Precision) const;

  double UnHidePrecision(double PrecisionPrime) const;

  double HideOffset( double Precision ) const;

  double UnHideOffset( double PrecisionBlind ) const;

  double HideUniform( double Precision ) const;

  double UnHideUniform( double PrecisionBlind ) const;


  const char *stSeed()const {return _stSeed;}

  const blindMode& mode()const {return _mode;}

  void setMode(blindMode bmode) {_mode=bmode;}

  double getPrecisionCentralValue() const {return _PrecisionCentralValue;}

  double getPrecisionOffsetScale() const {return _PrecisionOffsetScale;}

private:

  // Friends
  // Data members

  TString _stSeed ;
  double _DeltaZOffset;
  double _DeltaZSignFlip;
  double _DeltaZScale;
  double _AsymOffset;
  double _AsymSignFlip;
  double _DeltaMScale;
  double _DeltaMOffset;
  double _MysteryPhase;
  double _STagConstant;
  double _PrecisionSignFlip;
  double _PrecisionOffsetScale;
  double _PrecisionOffset;
  double _PrecisionUniform;
  double _PrecisionCentralValue;
  blindMode _mode;
  bool   _s2bMode ;

  // setup data members from string seed
  void setup(const char *stSeed);

protected:

    // Helper functions
  ClassDef(RooBlindTools,1) // Root implementation of BlindTools
};

#endif
