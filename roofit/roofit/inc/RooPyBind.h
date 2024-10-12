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

/** \class RooPyBind
    \ingroup Roofit
    \brief A RooFit class for wrapping python functions.

This clsss provides the functionality to wrap arbitrary python functions in
RooFit.
*/

#ifndef ROOPYBINDFUNCTION_H
#define ROOPYBINDFUNCTION_H

#include "RooAbsReal.h"
#include "RooListProxy.h"
#include "RooArgList.h"

namespace RooFit {
namespace Detail {

template <class BaseClass>
class RooPyBind : public BaseClass {
public:
   RooPyBind(const char *name, const char *title, RooArgList &varlist)
      : BaseClass(name, title), _varlist("!varlist", "All variables(list)", this)
   {
      _varlist.add(varlist);
   }

   RooPyBind(const RooPyBind &right, const char *name = nullptr)
      : BaseClass(right, name), _varlist("!varlist", this, right._varlist)
   {
   }

   RooPyBind *clone(const char *name) const override { return new RooPyBind(*this, name); }
   // This function should be redefined in Python
   double evaluate() const override { return 1.; }
   const RooArgList &varlist() const { return _varlist; }

   virtual double *doEvalPy(RooFit::EvalContext &) const
   {
      throw std::runtime_error("not implemented");
   }

protected:
   void doEval(RooFit::EvalContext &ctx) const override
   {
      std::span<double> output = ctx.output();
      std::span<const double> result{doEvalPy(ctx), output.size()};
      for (std::size_t i = 0; i < result.size(); ++i) {
         output[i] = result[i];
      }
   }

   RooListProxy _varlist; // all variables as list of variables

   ClassDefOverride(RooPyBind, 0);
};

} // namespace Detail
} // namespace RooFit

#endif
