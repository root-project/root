// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HypoTestInverterPlot
#define ROOSTATS_HypoTestInverterPlot

#include "TNamed.h"

class TGraphErrors; 


namespace RooStats {

  class HypoTestInverterResult; 

  class HypoTestInverterPlot : public TNamed {

  public:

    // constructor
    HypoTestInverterPlot( const char* name, 
			  const char* title,
			  HypoTestInverterResult* results ) ;

    TGraphErrors* MakePlot() ;

    // destructor
    ~HypoTestInverterPlot() ;

  private:

    HypoTestInverterResult* fResults;

  protected:

    ClassDef(HypoTestInverterPlot,1)  // HypoTestInverterPlot class

  };
}

#endif
