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

#include "RooPyBindFunction.h"

/** \class RooPyBindFunction
    \ingroup Roofit
    \brief A RooFit class for wrapping python functions.

This clsss provides the functionality to wrap arbitrary python functions in
RooFit.
*/

namespace RooFit {
namespace Detail {

RooPyBindFunction::RooPyBindFunction(const char *name, const char *title, RooArgList &varlist)
   : RooAbsReal(name, title), m_varlist("!varlist", "All variables(list)", this)
{
   m_varlist.add(varlist);
}

RooPyBindFunction::RooPyBindFunction(const RooPyBindFunction &right, const char *name)
   : RooAbsReal(right, name), m_varlist("!varlist", this, right.m_varlist)
{
}

RooPyBindFunction *RooPyBindFunction::clone(const char *name) const
{
   return new RooPyBindFunction(*this, name);
}

double RooPyBindFunction::evaluate() const
{
   // This function should be redefined in Python
   return 1;
}

const RooArgList &RooPyBindFunction::varlist() const
{
   return m_varlist;
}

} // namespace Detail
} // namespace RooFit


