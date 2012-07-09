// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Sven Kreiss   23/05/10
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
Same purpose as HybridCalculatorOriginal, but different implementation.
*/

#include "RooStats/HybridCalculator.h"
#include "RooStats/ToyMCSampler.h"


ClassImp(RooStats::HybridCalculator)

using namespace RooStats;
using namespace std;



int HybridCalculator::CheckHook(void) const {

   if( fPriorNuisanceNull && (!fNullModel->GetNuisanceParameters() || fNullModel->GetNuisanceParameters()->getSize() == 0) ) {
      oocoutE((TObject*)0,InputArguments)  << "HybridCalculator - Nuisance PDF has been specified, but is unaware of which parameters are the nuisance parameters. Must set nuisance parameters in the Null ModelConfig." << endl;
      return -1; // error
   }
   if( fPriorNuisanceAlt && (!fAltModel->GetNuisanceParameters() || fAltModel->GetNuisanceParameters()->getSize() == 0) ) {
      oocoutE((TObject*)0,InputArguments)  << "HybridCalculator - Nuisance PDF has been specified, but is unaware of which parameters are the nuisance parameters. Must set nuisance parameters in the Alt ModelConfig" << endl;
      return -1; // error
   }

   return 0; // ok
}


int HybridCalculator::PreNullHook(RooArgSet* /*parameterPoint*/, double obsTestStat) const {


   // ****** any TestStatSampler ********

   if(fPriorNuisanceNull) {
      // Setup Priors for ad hoc Hybrid
      fTestStatSampler->SetPriorNuisance(fPriorNuisanceNull);
   } else if(
      fNullModel->GetNuisanceParameters() == NULL ||
      fNullModel->GetNuisanceParameters()->getSize() == 0
   ) {
      oocoutI((TObject*)0,InputArguments)
       << "HybridCalculator - No nuisance parameters specified for Null model and no prior forced. "
       << "Case is reduced to simple hypothesis testing with no uncertainty." << endl;
   } else {
      oocoutI((TObject*)0,InputArguments) << "HybridCalculator - Using uniform prior on nuisance parameters (Null model)." << endl;
   }


   // ***** ToyMCSampler specific *******

   // check whether TestStatSampler is a ToyMCSampler
   ToyMCSampler *toymcs = dynamic_cast<ToyMCSampler*>(GetTestStatSampler());
   if(toymcs) {
      oocoutI((TObject*)0,InputArguments) << "Using a ToyMCSampler. Now configuring for Null." << endl;

      // variable number of toys
      if(fNToysNull >= 0) toymcs->SetNToys(fNToysNull);

      // adaptive sampling
      if(fNToysNullTail) {
         oocoutI((TObject*)0,InputArguments) << "Adaptive Sampling" << endl;
         if(GetTestStatSampler()->GetTestStatistic()->PValueIsRightTail()) {
            toymcs->SetToysRightTail(fNToysNullTail, obsTestStat);
         }else{
            toymcs->SetToysLeftTail(fNToysNullTail, obsTestStat);
         }
      }else{
         toymcs->SetToysBothTails(0, 0, obsTestStat); // disable adaptive sampling
      }

      GetNullModel()->LoadSnapshot();
   }

   return 0;
}


int HybridCalculator::PreAltHook(RooArgSet* /*parameterPoint*/, double obsTestStat) const {

   // ****** any TestStatSampler ********

   if(fPriorNuisanceAlt){
     // Setup Priors for ad hoc Hybrid
     fTestStatSampler->SetPriorNuisance(fPriorNuisanceAlt);
   } else if (
      fAltModel->GetNuisanceParameters()==NULL ||
      fAltModel->GetNuisanceParameters()->getSize()==0
   ) {
      oocoutI((TObject*)0,InputArguments)
       << "HybridCalculator - No nuisance parameters specified for Alt model and no prior forced. "
       << "Case is reduced to simple hypothesis testing with no uncertainty." << endl;
   } else {
      oocoutI((TObject*)0,InputArguments) << "HybridCalculator - Using uniform prior on nuisance parameters (Alt model)." << endl;
   }


   // ***** ToyMCSampler specific *******

   // check whether TestStatSampler is a ToyMCSampler
   ToyMCSampler *toymcs = dynamic_cast<ToyMCSampler*>(GetTestStatSampler());
   if(toymcs) {
      oocoutI((TObject*)0,InputArguments) << "Using a ToyMCSampler. Now configuring for Alt." << endl;

      // variable number of toys
      if(fNToysAlt >= 0) toymcs->SetNToys(fNToysAlt);

      // adaptive sampling
      if(fNToysAltTail) {
         oocoutI((TObject*)0,InputArguments) << "Adaptive Sampling" << endl;
         if(GetTestStatSampler()->GetTestStatistic()->PValueIsRightTail()) {
            toymcs->SetToysLeftTail(fNToysAltTail, obsTestStat);
         }else{
            toymcs->SetToysRightTail(fNToysAltTail, obsTestStat);
         }
      }else{
         toymcs->SetToysBothTails(0, 0, obsTestStat); // disable adaptive sampling
      }
   }

   return 0;
}




