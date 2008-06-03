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
#ifndef ROO_CHANGE_TRACKER
#define ROO_CHANGE_TRACKER

#include "RooAbsReal.h"
#include "RooListProxy.h"
#include <vector>

class RooRealVar;
class RooArgList ;

class RooChangeTracker : public RooAbsReal {
public:

  RooChangeTracker() ;
  RooChangeTracker(const char *name, const char *title, const RooArgSet& trackSet, Bool_t checkValues=kFALSE) ;
  virtual ~RooChangeTracker() ;

  RooChangeTracker(const RooChangeTracker& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooChangeTracker(*this, newname); }

  virtual Double_t getVal(const RooArgSet* /*set*/=0) const { return 0 ; }

  Bool_t hasChanged(Bool_t clearState) ;


protected:

  RooListProxy     _realSet ;
  RooListProxy     _catSet ;
  std::vector<Double_t> _realRef ;
  std::vector<Int_t>    _catRef ;
  Bool_t       _checkVal ; // Check contents as well if true

  mutable TIterator* _realSetIter ;     //! do not persist
  mutable TIterator* _catSetIter ;     //! do not persist

  Double_t evaluate() const;

  ClassDef(RooChangeTracker,1) // Meta object that tracks changes in set of other arguments
};

#endif
