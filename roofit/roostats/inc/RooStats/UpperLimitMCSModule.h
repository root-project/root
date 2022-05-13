// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke, Nils Ruthmann
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef UPPER_LIMIT_MCS_MODULE
#define UPPER_LIMIT_MCS_MODULE

#include "RooAbsMCStudyModule.h"
#include <string>

class RooArgSet;
class RooDataSet;
class RooRealVar;
class RooAbsPdf;

namespace RooStats {

   class ProfileLikelihoodCalculator;

class UpperLimitMCSModule : public RooAbsMCStudyModule {
public:


   UpperLimitMCSModule(const RooArgSet* poi, double CL=0.95) ;
   UpperLimitMCSModule(const UpperLimitMCSModule& other) ;
   ~UpperLimitMCSModule() override ;

   bool initializeInstance() override ;

   bool initializeRun(Int_t /*numSamples*/) override ;
   RooDataSet* finalizeRun() override ;

   //bool processAfterFit(Int_t /*sampleNum*/)  ;
   bool processBetweenGenAndFit(Int_t /*sampleNum*/) override ;

private:

   std::string _parName ;  ///< Name of Nsignal parameter
   RooStats::ProfileLikelihoodCalculator* _plc;
   RooRealVar* _ul ;

   const RooArgSet* _poi;  ///< parameters of interest
   RooDataSet* _data ;     ///< Summary dataset to store results
   double _cl;
   RooAbsPdf* _model;

   ClassDefOverride(UpperLimitMCSModule,0) // MCStudy module to calculate upper limit of a given poi
};

}

#endif

