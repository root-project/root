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
class TMultiGraph;


namespace RooStats {

   class HypoTestInverterResult;
   class SamplingDistPlot;


   class HypoTestInverterPlot : public TNamed {

   public:

      // constructor
      HypoTestInverterPlot(HypoTestInverterResult* results ) ;

      HypoTestInverterPlot( const char* name,
                            const char* title,
                            HypoTestInverterResult* results ) ;

      /// return a TGraphErrors with the obtained observed p-values
      /// resultinf from the scan
      /// By default (Option = "") return CLs or CLsb depending if the flag UseCLs is set
      /// If Option = "CLb"   return  CLb plot
      ///           = "CLs+b" return  CLs+b plot  independently of the flag
      ///           = "CLs"   return  CLs plot  independently of the flag
      TGraphErrors* MakePlot(Option_t *opt="") ;

      /// Make the expected plot and the bands
      /// nsig1 and nsig2 indicates the n-sigma value for the bands
      /// if nsig1 = 0 no band is computed (only expected value)
      /// if nsig2 > nsig1 (default is nsig1=1 and nsig2=2) the second band is also done.
      /// The first band is drawn in green while the second in yellow
      /// The plot (expected value + bands) is returned as a TMultiGraph object
      TMultiGraph* MakeExpectedPlot(double sig1=1, double sig2=2) ;

      /// Plot the test statistic distributions
      SamplingDistPlot * MakeTestStatPlot(int index, int type=0, int nbins = 100);


      /// Draw the scan result in the current canvas
      /// Possible options:
      ///   ""  (default): draw observed + expected with 1 and 2 sigma bands
      ///   SAME : draw in the current axis
      ///   OBS  :  draw only the observed plot
      ///   EXP  :  draw only the expected plot
      ///   CLB  : draw also  CLb
      ///   2CL  : drow both  CLs+b and CLs
      void Draw(Option_t *opt="");

      /// destructor
      ~HypoTestInverterPlot() ;

   private:

      HypoTestInverterResult* fResults;

   protected:

      ClassDef(HypoTestInverterPlot,1)  // HypoTestInverterPlot class

   };
}

#endif
