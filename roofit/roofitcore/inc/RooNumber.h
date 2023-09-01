/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNumber.h,v 1.7 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_NUMBER
#define ROO_NUMBER

#include <limits>
#include <string>

class RooNumber {
public:
   /// Return internal infinity representation.
   constexpr static double infinity() { return std::numeric_limits<double>::infinity(); }
   /// Return true if x is infinite by RooNumber internal specification.
   constexpr static int isInfinite(double x) { return (x >= +infinity()) ? +1 : ((x <= -infinity()) ? -1 : 0); }

   /// Set the relative epsilon that is used by range checks in RooFit,
   /// e.g., in RooAbsRealLValue::inRange().
   inline static void setRangeEpsRel(double epsRel) { staticRangeEpsRel() = epsRel; }
   /// Get the relative epsilon that is used by range checks in RooFit,
   /// e.g., in RooAbsRealLValue::inRange().
   inline static double rangeEpsRel() { return staticRangeEpsRel(); }

   /// Set the absolute epsilon that is used by range checks in RooFit,
   /// e.g., in RooAbsRealLValue::inRange().
   inline static void setRangeEpsAbs(double epsRel) { staticRangeEpsAbs() = epsRel; }
   /// Get the absolute epsilon that is used by range checks in RooFit,
   /// e.g., in RooAbsRealLValue::inRange().
   inline static double rangeEpsAbs() { return staticRangeEpsAbs(); }
   static std::string toString(double x);

private:
   static double &staticRangeEpsRel();
   static double &staticRangeEpsAbs();
};

#endif
