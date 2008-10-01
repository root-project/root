// @(#)root/hist:$Id$
// Author: L. Moneta Thu Aug 31 10:40:20 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TH1Interface

#ifndef ROOT_HFitInterface
#define ROOT_HFitInterface

#ifndef ROOT_Fit_BinData
#include "Fit/BinData.h"
#endif

class TH1; 
class TF1;
class TF2;
class TGraph; 
class TGraphErrors; 
class TGraph2D;  
class TMultiGraph; 
struct Foption_t; 


namespace ROOT { 

   namespace Math { 
      class MinimizerOptions; 
   }

   namespace Fit { 

      //class BinData; 
      class FitResult;
      class DataRange; 

#ifndef __CINT__  // does not link on Windows (why ??)

      /**
         fitting function for a TH1 (called from TH1::Fit)
       */
      int FitObject(TH1 * h1, TF1 *f1, Foption_t & option, const ROOT::Math::MinimizerOptions & moption, const char *goption, ROOT::Fit::DataRange & range); 

      /**
         fitting function for a TGraph (called from TGraph::Fit)
       */
      int FitObject(TGraph * gr, TF1 *f1 , Foption_t & option , const ROOT::Math::MinimizerOptions & moption, const char *goption, ROOT::Fit::DataRange & range);  

      /**
         fitting function for a MultiGraph (called from TMultiGraph::Fit)
       */
      int FitObject(TMultiGraph * mg, TF1 *f1 , Foption_t & option , const ROOT::Math::MinimizerOptions & moption, const char *goption, ROOT::Fit::DataRange & range);  

      /**
         fitting function for a TGraph2D (called from TGraph2D::Fit)
       */
      int FitObject(TGraph2D * gr, TF1 *f1 , Foption_t & option , const ROOT::Math::MinimizerOptions & moption, const char *goption, ROOT::Fit::DataRange & range); 
#endif

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
          fill the data vector from a TGraph. Pass also the TF1 function which is 
          needed in case to exclude points rejected by the function
      */ 
      void FillData ( BinData  & dv, const TGraph * gr, TF1 * func = 0 ); 
      /** 
          fill the data vector from a TMultiGraph. Pass also the TF1 function which is 
          needed in case to exclude points rejected by the function
      */ 
      void FillData ( BinData  & dv, const TMultiGraph * gr,  TF1 * func = 0); 

      

      /** 
          compute initial parameter for gaussian function given the fit data
          Set the sigma limits for zero top 10* initial rms values 
          Set the initial parameter values in the TF1
       */ 
      void InitGaus(const ROOT::Fit::BinData & data, TF1 * f1 ); 

      template <class FitObject> 
      void InitGaus(FitObject * obj, TF1 * f1) { 
         BinData dv; 
         FillData(dv, obj); // no need to function 
         InitGaus(dv, f1);
      }

      /**
         compute confidence intervals at level cl for a fitted histogram h1 in a TGraphErrors gr
      */
      bool GetConfidenceIntervals(const TH1 * h1, const ROOT::Fit::FitResult & r, TGraphErrors * gr, double cl = 0.95); 
      

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_TH1Interface */
