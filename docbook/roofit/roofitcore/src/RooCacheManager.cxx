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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// Template class RooCacheManager manages the storage of any type of data indexed on
// the choice of normalization and optionally the set of integrated observables.
// The purpose of this class is to faciliate storage of intermediate results
// in operator p.d.f.s whose value and inner working are often highly dependent
// on the user provided choice of normalization in getVal(). 
//
// For efficiency reasons these normalization set pointer are
// derefenced as little as possible. This class contains a lookup
// table for RooArgSet pointer pairs -> normalization lists.  Distinct
// pointer pairs that represent the same normalization/projection are
// recognized and will all point to the same normalization list. Lists
// for up to 'maxSize' different normalization/ projection
// configurations can be cached.  
// END_HTML
//
// 

#include "RooFit.h"
#include <vector>
#include "RooCacheManager.h"

using namespace std ;

#ifndef ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
#define ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
templateClassImp(RooCacheManager) 
#endif 





