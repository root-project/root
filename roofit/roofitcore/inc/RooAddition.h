/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAddition.h,v 1.3 2007/05/11 09:11:30 verkerke Exp $
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
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooAddition : public RooAbsReal {
public:

  RooAddition() ;
  RooAddition(const char *name, const char *title, const RooArgSet& sumSet, Bool_t takeOwnerShip=kFALSE) ;
  RooAddition(const char *name, const char *title, const RooArgList& sumSet1, const RooArgList& sumSet2, Bool_t takeOwnerShip=kFALSE) ;
  virtual ~RooAddition() ;

  RooAddition(const RooAddition& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooAddition(*this, newname); }

  virtual Double_t defaultErrorLevel() const ;

protected:

  RooArgList   _ownedList ;
  RooListProxy _set1 ;
  RooListProxy _set2 ;
  TIterator* _setIter1 ;  //! do not persist
  TIterator* _setIter2 ;  //! do not persist

  Double_t evaluate() const;

  ClassDef(RooAddition,1) // Sum of RooAbsReal terms
};

#endif
