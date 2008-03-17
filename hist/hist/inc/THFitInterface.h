// @(#)root/hist:$Id$
// Author: L. Moneta Thu Aug 31 10:40:20 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TH1Interface

#ifndef ROOT_THFitInterface
#define ROOT_THFitInterface

// #ifndef ROOT_Fit_DataVectorfwd
// #include "Fit/DataVectorfwd.h"
// #endif

class TH1; 
class TF1;
class TGraph2D;  

namespace ROOT { 

   namespace Fit { 

      class BinData; 


      /** 
          fill the data vector from a TH1. Pass also the TF1 function which is 
          needed in case of integral option and to reject points rejected by the function
      */ 
      void FillData ( BinData  & dv, const TH1 * hist, TF1 * func = 0); 

      /** 
          fill the data vector from a TGraph2D. Pass also the TF1 function which is 
          needed in case of integral option and to reject points rejected by the function
      */ 
      void FillData ( BinData  & dv, const TGraph2D * gr, TF1 * func = 0); 
      

      /** 
          compute initial parameter for gaussian function given the fit data
          Set the sigma limits for zero top 10* initial rms values 
          Set the initial parameter values in the TF1
       */ 
      void InitGaus(const ROOT::Fit::BinData & data, TF1 * f1 ); 
      

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_TH1Interface */
