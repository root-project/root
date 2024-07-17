/*
 * Project: RooFit
 * Authors:
 *   Robin Syring, CERN 2024
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOPYBINDFUNCTION_H
#define ROOPYBINDFUNCTION_H

#include "RooAbsReal.h"
#include "RooListProxy.h"
#include "RooArgList.h"

namespace RooFit {
namespace Detail {

class RooPyBindFunction : public RooAbsReal {
public:
   RooPyBindFunction(const char *name, const char *title, RooArgList &varlist);
   RooPyBindFunction(const RooPyBindFunction &right, const char *name = nullptr);

   RooPyBindFunction *clone(const char *name) const override;
   double evaluate() const override;
   const RooArgList &varlist() const;

protected:
   RooListProxy m_varlist; // all variables as list of variables

   ClassDefOverride(RooPyBindFunction, 0);
};

} // namespace Detail
} // namespace RooFit

#endif
