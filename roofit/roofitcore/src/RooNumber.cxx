/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

/**
\file RooNumber.cxx
\class RooNumber
\ingroup Roofitcore

Provides numeric constants used in \ref Roofitmain.
**/

#include <RooNumber.h>

/// @brief  Returns an std::to_string compatible number (i.e. rounds infinities back to the nearest representable
/// value). This function is primarily used in the code-squashing for AD and as such encodes infinities to double's
/// maximum value. We do this because 1, std::to_string cannot handle infinities correctly on some platforms
/// (e.g. 32 bit debian) and 2, Clad (the AD tool) cannot handle differentiating std::numeric_limits::infinity directly.
std::string RooNumber::toString(double x)
{
   int sign = isInfinite(x);
   double out = x;
   if (sign)
      out = sign == 1 ? std::numeric_limits<double>::max() : std::numeric_limits<double>::min();
   return std::to_string(out);
}

double &RooNumber::staticRangeEpsRel()
{
   static double epsRel = 0.0;
   return epsRel;
}

double &RooNumber::staticRangeEpsAbs()
{
   static double epsAbs = 0.0;
   return epsAbs;
}
