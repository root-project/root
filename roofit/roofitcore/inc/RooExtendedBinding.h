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

#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooRealProxy.h>
#include <RooSetProxy.h>

class RooExtendedBinding : public RooAbsReal {
public:
   RooExtendedBinding() {}
   RooExtendedBinding(const char *name, const char *title, RooAbsPdf &_pdf);
   RooExtendedBinding(const char *name, const char *title, RooAbsPdf &_pdf, const RooArgSet &_obs);
   RooExtendedBinding(const RooExtendedBinding &other, const char *name = nullptr);
   TObject *clone(const char *newname = nullptr) const override { return new RooExtendedBinding(*this, newname); }

   double evaluate() const override;

private:
   RooRealProxy pdf;
   std::unique_ptr<RooSetProxy> _obsList;

   ClassDefOverride(RooExtendedBinding, 2);
};

#endif
