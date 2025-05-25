/*
 * Project: RooFit
 * Authors:
 *   Kevin Belasco, 2009
 *   Kyle Cranmer, 2009
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooStats/Heaviside.h"
#include "RooAbsReal.h"

/** \class RooStats::Heaviside
    \ingroup Roostats

Represents the Heaviside function.
Evaluates to 1.0 when ((double)x) >= ((double)c), 0.0 otherwise.

*/


using namespace RooFit;
using namespace RooStats;

////////////////////////////////////////////////////////////////////////////////

Heaviside::Heaviside(const char *name, const char *title,
                       RooAbsReal& _x,
                       RooAbsReal& _c) :
  RooAbsReal(name,title),
  x("x","x",this,_x),
  c("c","c",this,_c)
{
}

////////////////////////////////////////////////////////////////////////////////

Heaviside::Heaviside(const Heaviside& other, const char* name) :
  RooAbsReal(other,name),
  x("x",this,other.x),
  c("c",this,other.c)
{
}

////////////////////////////////////////////////////////////////////////////////

double Heaviside::evaluate() const
{
  // ENTER EXPRESSION IN TERMS OF VARIABLE ARGUMENTS HERE
  if (((double)x) >= ((double)c)) {
     return 1.0;
  } else {
     return 0.0;
  }
}
