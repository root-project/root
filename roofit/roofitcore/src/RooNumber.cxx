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
