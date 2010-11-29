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


int HybridCalculator::PreNullHook(double obsTestStat) const {
   // check whether TestStatSampler is a ToyMCSampler
   ToyMCSampler *toymcs = dynamic_cast<ToyMCSampler*>(GetTestStatSampler());
   if(toymcs) {
      oocoutI((TObject*)0,InputArguments) << "Using a ToyMCSampler. Now configuring for Null." << endl;

      // variable number of toys
      if(fNToysNull) toymcs->SetNToys(fNToysNull);

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

      // importance sampling
      if(fNullImportanceDensity) {
         oocoutI((TObject*)0,InputArguments) << "Importance Sampling" << endl;
         toymcs->SetImportanceDensity(fNullImportanceDensity);
         if(fNullImportanceSnapshot) toymcs->SetImportanceSnapshot(*fNullImportanceSnapshot);
      }else{
         toymcs->SetImportanceDensity(NULL);       // disable importance sampling
      }
      GetNullModel()->LoadSnapshot();
   }

   return 0;
}


int HybridCalculator::PreAltHook(double obsTestStat) const {
   // check whether TestStatSampler is a ToyMCSampler
   ToyMCSampler *toymcs = dynamic_cast<ToyMCSampler*>(GetTestStatSampler());
   if(toymcs) {
      oocoutI((TObject*)0,InputArguments) << "Using a ToyMCSampler. Now configuring for Alt." << endl;

      // variable number of toys
      if(fNToysAlt) toymcs->SetNToys(fNToysAlt);

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


      // importance sampling
      if(fAltImportanceDensity) {
         oocoutI((TObject*)0,InputArguments) << "Importance Sampling" << endl;
         toymcs->SetImportanceDensity(fAltImportanceDensity);
         if(fAltImportanceSnapshot) toymcs->SetImportanceSnapshot(*fAltImportanceSnapshot);
      }else{
         toymcs->SetImportanceDensity(NULL);       // disable importance sampling
      }
   }

   return 0;
}




