/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOPROFILELL
#define ROOPROFILELL

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include <map>
#include <string>

class RooMinuit ;
 
class RooProfileLL : public RooAbsReal {
public:

  RooProfileLL(const char *name, const char *title, RooAbsReal& nll, const RooArgSet& observables);
  RooProfileLL(const RooProfileLL& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooProfileLL(*this,newname); }
  virtual ~RooProfileLL() ;

protected:

  RooRealProxy _nll ;
  RooSetProxy _obs ;
  RooSetProxy _par ;

  TIterator* _piter ; //! 
  TIterator* _oiter ; //!

  mutable RooMinuit* _minuit ; //!

  mutable Bool_t _absMinValid ; // flag if absmin is up-to-date
  mutable Double_t _absMin ; // absolute minimum of -log(L)
  
  Double_t evaluate() const ;
  mutable std::map<std::string,bool> _paramFixed ;

private:

  ClassDef(RooProfileLL,0) // Real-valued function representing profile likelihood of external (likelihood) function
};
 
#endif
