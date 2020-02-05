// Author: Stephan Hageboeck, CERN  2 Sep 2019

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOFIT_ROOFITCORE_ROOVDTHEADERS_H_
#define ROOFIT_ROOFITCORE_ROOVDTHEADERS_H_

/**
 * VDT headers for RooFit. Since RooFit cannot directly depend on VDT (it might not be available),
 * this layer can be used to switch between different implementations.
 */

#include "ROOT/RConfig.hxx"

#if defined(R__HAS_VDT)
#include "vdt/exp.h"
#include "vdt/log.h"
#include "vdt/sqrt.h"

inline double _rf_fast_exp(double x) {
  return vdt::fast_exp(x);
}

inline double _rf_fast_log(double x) {
  return vdt::fast_log(x);
}

inline double _rf_fast_isqrt(double x) {
  return vdt::fast_isqrt(x);
}

#else
#include <cmath>

inline double _rf_fast_exp(double x) {
  return std::exp(x);
}

inline double _rf_fast_log(double x) {
  return std::exp(x);
}

inline double _rf_fast_isqrt(double x) {
  return 1/std::sqrt(x);
}

#endif


#endif /* ROOFIT_ROOFITCORE_ROOVDTHEADERS_H_ */
