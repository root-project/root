/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: PiecewiseInterpolation.h,v 1.3 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ADDITION
#define ROO_ADDITION

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class PiecewiseInterpolation : public RooAbsReal {
public:

  PiecewiseInterpolation() ;
  PiecewiseInterpolation(const char *name, const char *title, const RooAbsReal& nominal, const RooArgList& lowSet, const RooArgList& highSet, const RooArgList& paramSet, Bool_t takeOwnerShip=kFALSE) ;
  virtual ~PiecewiseInterpolation() ;

  PiecewiseInterpolation(const PiecewiseInterpolation& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new PiecewiseInterpolation(*this, newname); }

  //  virtual Double_t defaultErrorLevel() const ;

  //  void printMetaArgs(ostream& os) const ;

  const RooArgList& lowList() const { return _lowSet ; }
  const RooArgList& highList() const { return _highSet ; }
  const RooArgList& paramList() const { return _paramSet ; }

protected:

  RooRealProxy _nominal;           // The nominal value
  RooArgList   _ownedList ;       // List of owned components
  RooListProxy _lowSet ;            // Low-side variation
  RooListProxy _highSet ;            // High-side varaition
  RooListProxy _paramSet ;            // interpolation parameters
  mutable TIterator* _paramIter ;  //! Iterator over paramSet
  mutable TIterator* _lowIter ;  //! Iterator over lowSet
  mutable TIterator* _highIter ;  //! Iterator over highSet

  Double_t evaluate() const;

  ClassDef(PiecewiseInterpolation,1) // Sum of RooAbsReal objects
};

#endif
