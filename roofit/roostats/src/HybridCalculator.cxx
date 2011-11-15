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



int HybridCalculator::CheckHook(void) const {
   if( (fNullModel->GetNuisanceParameters()
        && fNullModel->GetNuisanceParameters()->getSize()>0
        && !fPriorNuisanceNull)
    || (fAltModel->GetNuisanceParameters()
        && fAltModel->GetNuisanceParameters()->getSize()>0
        && !fPriorNuisanceAlt)
   ){
      oocoutE((TObject*)0,InputArguments)  << "Must ForceNuisancePdf, inferring posterior from ModelConfig is not yet implemented" << endl;
      return -1; // error
   }

   if(    (!fNullModel->GetNuisanceParameters() && fPriorNuisanceNull)
       || (!fAltModel->GetNuisanceParameters()  && fPriorNuisanceAlt)
       || (fNullModel->GetNuisanceParameters()  && fNullModel->GetNuisanceParameters()->getSize()==0 && fPriorNuisanceNull)
       || (fAltModel->GetNuisanceParameters()  && fAltModel->GetNuisanceParameters()->getSize()>0   && !fPriorNuisanceAlt)
   ){
      oocoutE((TObject*)0,InputArguments)  << "Nuisance PDF specified, but the pdf doesn't know which parameters are the nuisance parameters.  Must set nuisance parameters in the ModelConfig" << endl;
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
      fNullModel->GetNuisanceParameters()==NULL ||
      fNullModel->GetNuisanceParameters()->getSize()==0
   ) {
      oocoutI((TObject*)0,InputArguments)
       << "No nuisance parameters specified and no prior forced, reduces "
       << "to simple hypothesis testing with no uncertainty" << endl;
   } else {
      // TODO principled case:
      // must create posterior from Model.PriorPdf and Model.Pdf

      // Note, we do not want to use "prior" for nuisance parameters:
      // fTestStatSampler->SetPriorNuisance(const_cast<RooAbsPdf*>(model.GetPriorPdf()));

      oocoutE((TObject*)0,InputArguments) << "inferring posterior from ModelConfig is not yet implemented" << endl;
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
         << "No nuisance parameters specified and no prior forced, reduces "
         << "to simple hypothesis testing with no uncertainty" << endl;
   } else {
      // TODO principled case:
      // must create posterior from Model.PriorPdf and Model.Pdf

      // Note, we do not want to use "prior" for nuisance parameters:
      // fTestStatSampler->SetPriorNuisance(const_cast<RooAbsPdf*>(model.GetPriorPdf()));

      oocoutE((TObject*)0,InputArguments) << "inferring posterior from ModelConfig is not yet implemented" << endl;
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




