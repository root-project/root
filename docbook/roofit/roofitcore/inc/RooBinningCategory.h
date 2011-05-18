/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_BINNING_CATEGORY
#define ROO_BINNING_CATEGORY

#include "TSortedList.h"
#include "RooAbsCategory.h"
#include "RooRealProxy.h"
#include "RooCatType.h"

class RooBinningCategory : public RooAbsCategory {

public:
  // Constructors etc.
  inline RooBinningCategory() { }
  RooBinningCategory(const char *name, const char *title, RooAbsRealLValue& inputVar, const char* binningName=0);
  RooBinningCategory(const RooBinningCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooBinningCategory(*this, newname); }
  virtual ~RooBinningCategory();

  // Printing interface (human readable)
  virtual void printMultiline(ostream& os, Int_t content, Bool_t verbose=kFALSE, TString indent="") const ;

protected:
  
  void initialize() ;

  RooRealProxy _inputVar ; // Input variable that is mapped
  TString _bname ;         // Name of the binning specification to be used to perform the mapping

  virtual RooCatType evaluate() const ; 

  ClassDef(RooBinningCategory,1) // RealVar-to-Category function defined by bin boundaries on input var
};

#endif
