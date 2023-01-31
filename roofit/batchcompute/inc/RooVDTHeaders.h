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

#ifndef ROOFIT_BATCHCOMPUTE_ROOVDTHEADERS_H
#define ROOFIT_BATCHCOMPUTE_ROOVDTHEADERS_H

/*
 * VDT headers for RooFit. Since RooFit cannot directly depend on VDT (it might not be available),
 * this layer can be used to switch between different implementations.
 */

#include "RConfigure.h"

#if defined(R__HAS_VDT) && !defined(__CUDACC__)
#include <vdt/cos.h>
#include <vdt/exp.h>
#include <vdt/log.h>
#include <vdt/sin.h>
#include <vdt/sqrt.h>

namespace RooBatchCompute {

inline double fast_exp(double x)
{
   return vdt::fast_exp(x);
}

inline double fast_sin(double x)
{
   return vdt::fast_sin(x);
}

inline double fast_cos(double x)
{
   return vdt::fast_cos(x);
}

inline double fast_log(double x)
{
   return vdt::fast_log(x);
}

inline double fast_isqrt(double x)
{
   return vdt::fast_isqrt(x);
}

} // namespace RooBatchCompute

#else
#include <cmath>

namespace RooBatchCompute {

__roodevice__ inline double fast_exp(double x)
{
   return std::exp(x);
}

__roodevice__ inline double fast_sin(double x)
{
   return std::sin(x);
}

__roodevice__ inline double fast_cos(double x)
{
   return std::cos(x);
}

__roodevice__ inline double fast_log(double x)
{
   return std::log(x);
}

__roodevice__ inline double fast_isqrt(double x)
{
   return 1. / std::sqrt(x);
}

} // namespace RooBatchCompute

#endif // defined(R__HAS_VDT) && !defined(__CUDACC__)

#endif // ROOFIT_BATCHCOMPUTE_ROOVDTHEADERS_H_
