/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooThresholdCategory.h,v 1.8 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_THRESHOLD_CATEGORY
#define ROO_THRESHOLD_CATEGORY

#include "TSortedList.h"
#include "RooAbsCategory.h"
#include "RooRealProxy.h"
#include "RooCatType.h"

class RooThresholdCategory : public RooAbsCategory {

public:
  // Constructors etc.
  inline RooThresholdCategory() { }
  RooThresholdCategory(const char *name, const char *title, RooAbsReal& inputVar, const char* defCatName="Default", Int_t defCatIdx=0);
  RooThresholdCategory(const RooThresholdCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooThresholdCategory(*this, newname); }
  virtual ~RooThresholdCategory();

  // Mapping function
  Bool_t addThreshold(Double_t upperLimit, const char* catName, Int_t catIdx=-99999) ;

  // Printing interface (human readable)
  virtual void printMultiline(ostream& os, Int_t content, Bool_t verbose=kFALSE, TString indent="") const ;

  void writeToStream(ostream& os, Bool_t compact) const ;

protected:
  
  RooRealProxy _inputVar ;
  RooCatType* _defCat ;
  TSortedList _threshList ;
  TIterator* _threshIter ; //! do not persist 

  virtual RooCatType evaluate() const ; 

  ClassDef(RooThresholdCategory,1) // Real-to-Category function defined by series of threshold
};

#endif
