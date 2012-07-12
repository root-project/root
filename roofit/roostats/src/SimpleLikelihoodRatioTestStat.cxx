// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/SimpleLikelihoodRatioTestStat.h"

Bool_t RooStats::SimpleLikelihoodRatioTestStat::fAlwaysReuseNll = kTRUE ;

void RooStats::SimpleLikelihoodRatioTestStat::SetAlwaysReuseNLL(Bool_t flag) { fAlwaysReuseNll = flag ; }
