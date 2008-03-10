// @(#)root/graf:$Id$
// Author: L. Moneta Thu Nov 15 17:04:20 2007

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TGraphFitInterface

#ifndef ROOT_TGraphFitInterface
#define ROOT_TGraphFitInterface

#ifndef ROOT_Fit_DataVectorfwd
#include "Fit/DataVectorfwd.h"
#endif

class TGraph; 
class TMultiGraph; 
class TF1; 

namespace ROOT { 

   namespace Fit { 



      /** 
          fill the data vector from a TGraph. Pass also the TF1 function which is 
          needed in case to exclude points rejected by the function
      */ 
      void FillData ( BinData  & dv, const TGraph * gr, TF1 * func = 0 ); 
      /** 
          fill the data vector from a TMultiGraph. Pass also the TF1 function which is 
          needed in case to exclude points rejected by the function
      */ 
      void FillData ( BinData  & dv, const TMultiGraph * gr,  TF1 * func = 0); 

      
      

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_TGraphFitInterface */
