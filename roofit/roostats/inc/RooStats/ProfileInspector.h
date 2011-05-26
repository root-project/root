// @(#)root/roostats:$Id: ProfileInspector.h 31793 2009-12-10 14:43:51Z cranmer $

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke       *
 *   Akira Shibata
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ProfileInspector
#define ROOSTATS_ProfileInspector

#include "TList.h"
#include "RooStats/ModelConfig.h"
#include "RooAbsData.h"

namespace RooStats {

  class ProfileInspector {

   public:
    ProfileInspector();

    // Destructor of SamplingDistribution
    virtual ~ProfileInspector();

    TList * GetListOfProfilePlots( RooAbsData& data, RooStats::ModelConfig * config);



  protected:

    ClassDef(ProfileInspector,1)  // Class containing the results of the IntervalCalculator
  };
}

#endif
