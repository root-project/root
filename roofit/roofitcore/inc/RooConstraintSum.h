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
#ifndef ROO_CONSTRAINT_SUM
#define ROO_CONSTRAINT_SUM

#include "RooAbsReal.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooConstraintSum : public RooAbsReal {
public:

  RooConstraintSum() ;
  RooConstraintSum(const char *name, const char *title, const RooArgSet& constraintSet) ;
  virtual ~RooConstraintSum() ;

  RooConstraintSum(const RooConstraintSum& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooConstraintSum(*this, newname); }

protected:

  RooListProxy _set1 ;
  TIterator* _setIter1 ;  //! do not persist

  Double_t evaluate() const;

  ClassDef(RooConstraintSum,1) // Sum of RooAbsReal terms
};

#endif
