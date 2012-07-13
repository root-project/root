// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"

#include "RooArgSet.h"
#include "RooAbsData.h"

Bool_t RooStats::RatioOfProfiledLikelihoodsTestStat::fgAlwaysReuseNll = kTRUE ;

void RooStats::RatioOfProfiledLikelihoodsTestStat::SetAlwaysReuseNLL(Bool_t flag) { fgAlwaysReuseNll = flag ; }
