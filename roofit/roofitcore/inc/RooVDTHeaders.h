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

#if defined(R__HAS_VDT)
#include "vdt/exp.h"
#include "vdt/log.h"
#else
#define vdt::fast_exp std::exp
#define vdt::fast_log std::log
#endif


#endif /* ROOFIT_ROOFITCORE_ROOVDTHEADERS_H_ */
