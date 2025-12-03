/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOHISTCONSTRAINT
#define ROOHISTCONSTRAINT

#include <RooAbsPdf.h>
#include <RooListProxy.h>

class RooHistConstraint : public RooAbsPdf {
public:
  RooHistConstraint() {} ;
  RooHistConstraint(const char *name, const char *title, const RooArgSet& phfSet, int threshold=1000000);
  RooHistConstraint(const RooHistConstraint& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooHistConstraint(*this,newname); }

  double getLogVal(const RooArgSet* set=nullptr) const override ;

  /// It makes only sense to use the RooHistConstraint when normalized over the
  /// set of all gammas, in which case it is self-normalized because the used
  /// TMath::Poisson function is normalized.
  bool selfNormalized() const override { return true; }

protected:

  RooListProxy _gamma ;
  RooListProxy _nominal ;
  bool _relParam ;

  double evaluate() const override ;

private:

  ClassDefOverride(RooHistConstraint, 2)
};

#endif
