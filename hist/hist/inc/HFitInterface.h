// @(#)root/hist:$Id$
// Author: L. Moneta Thu Aug 31 10:40:20 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class HFitInterface
// set of free functions used to couple the ROOT data object with the fitting classes

// avoid including this file when running CINT since free functions cannot be re-defined

#ifndef ROOT_HFitInterface
#define ROOT_HFitInterface


class TH1;
class THnBase;
class TF1;
class TF2;
class TGraph;
class TGraphErrors;
class TGraph2D;
class TMultiGraph;
struct Foption_t;

#include "TFitResultPtr.h"

namespace ROOT {

   namespace Math {
      class MinimizerOptions;
   }

   namespace Fit {

      //class BinData;

      class FitResult;
      class DataRange;
      class BinData;
      class UnBinData;
      class SparseData;

      enum EFitObjectType {
         kHistogram,
         kGraph
      };


      //#ifndef __CINT__  // does not link on Windows (why ??)

      /**
         Decode list of options into fitOption
       */
      void FitOptionsMake(EFitObjectType type, const char *option, Foption_t &fitOption);

      /**
         fitting function for a TH1 (called from TH1::Fit)
       */
      TFitResultPtr FitObject(TH1 * h1, TF1 *f1, Foption_t & option, const ROOT::Math::MinimizerOptions & moption, const char *goption, ROOT::Fit::DataRange & range);

      /**
         fitting function for a TGraph (called from TGraph::Fit)
       */
      TFitResultPtr FitObject(TGraph * gr, TF1 *f1 , Foption_t & option , const ROOT::Math::MinimizerOptions & moption, const char *goption, ROOT::Fit::DataRange & range);

      /**
         fitting function for a MultiGraph (called from TMultiGraph::Fit)
       */
      TFitResultPtr FitObject(TMultiGraph * mg, TF1 *f1 , Foption_t & option , const ROOT::Math::MinimizerOptions & moption, const char *goption, ROOT::Fit::DataRange & range);

      /**
         fitting function for a TGraph2D (called from TGraph2D::Fit)
       */
      TFitResultPtr FitObject(TGraph2D * gr, TF1 *f1 , Foption_t & option , const ROOT::Math::MinimizerOptions & moption, const char *goption, ROOT::Fit::DataRange & range);

      /**
         fitting function for a THn / THnSparse (called from THnBase::Fit)
       */
      TFitResultPtr FitObject(THnBase * s1, TF1 *f1, Foption_t & option, const ROOT::Math::MinimizerOptions & moption, const char *goption, ROOT::Fit::DataRange & range);


      /**
          fit an unbin data set (from tree or from histogram buffer)
          using a TF1 pointer and fit options.
          N.B. ownership of fit data is passed to the UnBinFit function which will be responsible of
          deleting the data after the fit. User calling this function MUST NOT delete UnBinData after
          calling it.
      */
      TFitResultPtr UnBinFit(ROOT::Fit::UnBinData * data, TF1 * f1 , Foption_t & option , const ROOT::Math::MinimizerOptions & moption);

      /**
          fill the data vector from a TH1. Pass also the TF1 function which is
          needed in case of integral option and to reject points rejected by the function
      */
      void FillData ( BinData  & dv, const TH1 * hist, TF1 * func = nullptr);

      /**
          fill the data vector from a TH1 with sparse data. Pass also the TF1 function which is
          needed in case of integral option and to reject points rejected by the function
      */
      void FillData ( SparseData  & dv, const TH1 * hist, TF1 * func = nullptr);

      /**
          fill the data vector from a THnBase. Pass also the TF1 function which is
          needed in case of integral option and to reject points rejected by the function
      */
      void FillData ( SparseData  & dv, const THnBase * hist, TF1 * func = nullptr);

      /**
          fill the data vector from a THnBase. Pass also the TF1 function which is
          needed in case of integral option and to reject points rejected by the function
      */
      void FillData ( BinData  & dv, const THnBase * hist, TF1 * func = nullptr);

      /**
          fill the data vector from a TGraph2D. Pass also the TF1 function which is
          needed in case of integral option and to reject points rejected by the function
      */
      void FillData ( BinData  & dv, const TGraph2D * gr, TF1 * func = nullptr);


      /**
          fill the data vector from a TGraph. Pass also the TF1 function which is
          needed in case to exclude points rejected by the function
      */
      void FillData ( BinData  & dv, const TGraph * gr, TF1 * func = nullptr);
      /**
          fill the data vector from a TMultiGraph. Pass also the TF1 function which is
          needed in case to exclude points rejected by the function
      */
      void FillData ( BinData  & dv, const TMultiGraph * gr,  TF1 * func = nullptr);


      /**
          compute initial parameter for an exponential function given the fit data
          Set the constant and slope assuming a simple exponential going through xmin and xmax
          of the data set
       */
      void InitExpo(const ROOT::Fit::BinData & data, TF1 * f1);


      /**
          compute initial parameter for gaussian function given the fit data
          Set the sigma limits for zero top 10* initial rms values
          Set the initial parameter values in the TF1
       */
      void InitGaus(const ROOT::Fit::BinData & data, TF1 * f1);

      /**
          compute initial parameter for 2D gaussian function given the fit data
          Set the sigma limits for zero top 10* initial rms values
          Set the initial parameter values in the TF1
       */
      void Init2DGaus(const ROOT::Fit::BinData & data, TF1 * f1);

      /**
         compute confidence intervals at level cl for a fitted histogram h1 in a TGraphErrors gr
      */
      bool GetConfidenceIntervals(const TH1 * h1, const ROOT::Fit::FitResult & r, TGraphErrors * gr, double cl = 0.95);

      /**
         compute the chi2 value for an histogram given a function  (see TH1::Chisquare for the documentation)
      */
      double Chisquare(const TH1 & h1, TF1 & f1, bool useRange, bool usePL = false);

      /**
         compute the chi2 value for a graph given a function (see TGraph::Chisquare)
      */
      double Chisquare(const TGraph & h1, TF1 & f1, bool useRange);


   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_TH1Interface */


//#endif  /* not CINT OR MAKE_CINT */
