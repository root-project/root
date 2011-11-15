// @(#)root/roostats:$Id$
// Author: Sven Kreiss, Kyle Cranmer   Nov 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_AsymptoticCalculator
#define ROOSTATS_AsymptoticCalculator

//_________________________________________________
/*
BEGIN_HTML
<p>
Calculator based on Asymptotic formula, 
 on the ProfileLikelihood and the Asimov data set
</p>
END_HTML
*/
//



#ifndef ROOSTATS_HypoTestCalculatorGeneric
#include "RooStats/HypoTestCalculatorGeneric.h"
#endif

class RooArgSet;
class RooPoisson;
class RooProdPdf;


namespace RooStats {

   class AsymptoticCalculator : public HypoTestCalculatorGeneric {

   public:
      AsymptoticCalculator(
                        const RooAbsData &data,
                        const ModelConfig &altModel,
                        const ModelConfig &nullModel
         );
      //    HypoTestCalculatorGeneric(data, altModel, nullModel, 0)
      // {
      // }

      ~AsymptoticCalculator() {
      }



      // re-implement HypoTest computation using the asymptotic 
      virtual HypoTestResult *GetHypoTest() const; 

      RooAbsData * MakeAsimovData(const RooArgSet & paramValues, RooArgSet & globObs); 

      static RooAbsData * GenerateAsimovData(const RooAbsPdf & pdf, const RooArgSet & observables ); 

      // function given the null and the alt p value - return the expected one given the N - sigma value
      static double GetExpectedPValues(double pnull, double palt, double nsigma, bool usecls ); 

      // get expected limit 
//      static void GetExpectedLimit(double nsigma, double alpha, double &clsblimit, double &clslimit);

      void SetOneSided(bool on) { fOneSided = on; }

      // set using of qtilde, by default is controlled if RoORealVar is limited or not 
      void SetQTilde(bool on) { fUseQTilde = on; }

      static void SetPrintLevel(int level);

   protected:
      // // configure TestStatSampler for the Null run
      // int PreNullHook(RooArgSet *parameterPoint, double obsTestStat) const;

      // // configure TestStatSampler for the Alt run
      // int PreAltHook(RooArgSet *parameterPoint, double obsTestStat) const;

      
      static RooAbsData * GenerateAsimovDataSinglePdf(const RooAbsPdf & pdf, const RooArgSet & obs,  const RooRealVar & weightVar,
                                                      RooCategory * channelCat = 0);

      static RooAbsData * GenerateCountingAsimovData(RooAbsPdf & pdf, const RooArgSet & obs,  const RooRealVar & weightVar,
                                                      RooCategory * channelCat = 0);


      static void FillBins(const RooAbsPdf & pdf, const RooArgList &obs, RooAbsData & data, int &index,  double
                           &binVolume, int &ibin); 

      static double EvaluateNLL(RooAbsPdf & pdf, RooAbsData& data, const RooArgSet *poiSet = 0); 

      static void SetObsToExpected(RooProdPdf &prod, const RooArgSet &obs); 
      static void SetObsToExpected(RooPoisson &pois, const RooArgSet &obs);

   protected:
      ClassDef(AsymptoticCalculator,1)

   private: 

      bool fOneSided;
      mutable int fUseQTilde;              // flag to indicate if using qtilde or not (-1 (default based on RooRealVar)), 0 false, 1 (true)
      static int fgPrintLevel;     // control print level  (0 minimal, 1 normal, 2 debug)
      mutable double fNLLObs; 
      mutable double fNLLAsimov; 

      mutable RooAbsData * fAsimovData;   // asimov data set 
      RooArgSet  fAsimovGlobObs;  // snapshot of Asimov global observables 
      mutable RooArgSet  fBestFitPoi;       // snapshot of best fitted POI values
      
      
   };
}

#endif
