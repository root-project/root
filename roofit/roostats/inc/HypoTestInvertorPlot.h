// @(#)root/roostats:$Id: SimpleInterval.h 30478 2009-09-25 19:42:07Z schott $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HypoTestInvertorPlot
#define ROOSTATS_HypoTestInvertorPlot

#include "TNamed.h"

class TGraph; 


namespace RooStats {

  class HypoTestInvertorResult; 

  class HypoTestInvertorPlot : public TNamed {

  public:

    // constructor
    HypoTestInvertorPlot( const char* name, 
			  const char* title,
			  HypoTestInvertorResult* results ) ;

    TGraph* MakePlot() ;

    // destructor
    ~HypoTestInvertorPlot() ;

  private:

    HypoTestInvertorResult* fResults;

  protected:

    ClassDef(HypoTestInvertorPlot,1)  // HypoTestInvertorPlot class

  };
}

#endif
