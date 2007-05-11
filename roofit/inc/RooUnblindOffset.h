/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooUnblindOffset.rdl,v 1.5 2005/02/25 14:25:06 wverkerke Exp $
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
#ifndef ROO_UNBLIND_OFFSET
#define ROO_UNBLIND_OFFSET

#include "RooAbsHiddenReal.h"
#include "RooRealProxy.h"
#include "RooBlindTools.h"

class RooUnblindOffset : public RooAbsHiddenReal {
public:
  // Constructors, assignment etc
  RooUnblindOffset() ;
  RooUnblindOffset(const char *name, const char *title, 
		   const char *blindString, Double_t scale, RooAbsReal& blindValue);
  RooUnblindOffset(const char *name, const char *title, 
		   const char *blindString, Double_t scale, RooAbsReal& blindValue,
		   RooAbsCategory& blindState);
  RooUnblindOffset(const RooUnblindOffset& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooUnblindOffset(*this,newname); }  
  virtual ~RooUnblindOffset();

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;

  RooRealProxy _value ;
  RooBlindTools _blindEngine ;

  ClassDef(RooUnblindOffset,1) // Offset unblinding transformation
};

#endif
