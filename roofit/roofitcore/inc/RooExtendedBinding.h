/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOEXTENDEDBINDING
#define ROOEXTENDEDBINDING

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsPdf.h"
#include "RooAbsCategory.h"

class RooExtendedBinding : public RooAbsReal {
public:
  RooExtendedBinding() {} ;
  RooExtendedBinding(const char *name, const char *title, RooAbsPdf& _pdf);
  RooExtendedBinding(const RooExtendedBinding& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooExtendedBinding(*this,newname); }

protected:

  RooRealProxy pdf ;

  double evaluate() const override ;

private:

  ClassDefOverride(RooExtendedBinding,1);
};

#endif
