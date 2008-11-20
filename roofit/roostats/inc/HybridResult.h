// @(#)root/roostats:$Id$

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke       *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HybridResult
#define ROOSTATS_HybridResult

#ifndef ROOSTATS_HypoTestResult
#include "RooStats/HypoTestResult.h"
#endif

namespace RooStats {

   class HybridPlot; 

   class HybridResult /*: public HypoTestResult*/ {  /// TO DO: inheritance

   public:

      /// Constructor for HybridResult
      HybridResult(const char *name,const char *title,std::vector<float>& testStat_sb_vals,
                   std::vector<float>& testStat_b_vals,
                   float testStat_data_val );

      /// Destructor of HybridResult
      virtual ~HybridResult();

      /// TO DO: use from HypoTestResult
      double CLb();
      double CLsplusb();
      double CLs();

      void Add(HybridResult* other);
      HybridPlot* GetPlot(const char* name,const char* title, int n_bins);
      void Print(const char* options);

   private:
      const char* fName; /// TO DO: put to inherited (TNamed for write to file)
      const char* fTitle; /// TO DO: put to inherited (TNamed for write to file)

      std::vector<float> fTestStat_b; // results for B-only toy-MC
      std::vector<float> fTestStat_sb; // results for S+B toy-MC
      float fTestStat_data; // results for data

   protected:

      ClassDef(HybridResult,1)   // Class containing the results of the HybridCalculator
   };
}

#endif
