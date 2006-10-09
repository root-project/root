// @(#)root/tmva $Id: SimulatedAnnealingBase.cxx,v 1.7 2006/08/30 22:19:59 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SimulatedAnnealingBase                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Implementation of Simulated Annealing fitter  
//_______________________________________________________________________

#include "Riostream.h"
#include "TMVA/SimulatedAnnealingBase.h"

ClassImp(TMVA::SimulatedAnnealingBase)

TMVA::SimulatedAnnealingBase::SimulatedAnnealingBase( std::vector<LowHigh_t*>& ranges )
   : fRanges( ranges )
{   
   fRandom                 = new TRandom();

   // set default options
   fMaxCalls               = 500000;
   fTemperatureGradient    = 0.3;
   fUseAdaptiveTemperature = kFALSE;
   fInitialTemperature     = 1000;
   fMinTemperature         = 0;
   fEps                    = 1e-04;
   fNFunLoops              = 25;
   fNEps                   = 4; // needs to be at leas 2 !
}

TMVA::SimulatedAnnealingBase::~SimulatedAnnealingBase()
{}

Double_t TMVA::SimulatedAnnealingBase::Minimize( std::vector<Double_t>& parameters )
{
   // speed up loops
   Int_t npar = parameters.size();
      
   // set values
   Double_t deltaT = fTemperatureGradient;

   // step width of temperature reduction
   Double_t stepWidth = 3*npar;
   if (npar < 20) stepWidth = 50; // don't go below some minimum 

   Int_t i; // predefine for compatibility with old platforms (eg, SunOS5)
   std::vector <Double_t> diffFCN; 
   for (i = 0; i < fNEps; ++i) diffFCN.push_back(1e20);

   // result vector
   std::vector <Double_t> bestParameters( parameters );

   // auxiliary vectors
   std::vector <Double_t> xPars( parameters );
   std::vector <Double_t> yPars( parameters );
   
   // initialize adaptive errors with bold guess
   std::vector <Double_t> adaptiveErrors;
   std::vector <Int_t>    nAccepted;
   for (Int_t k = 0; k < npar; k++ ) {
      adaptiveErrors.push_back( (fRanges[k]->second - fRanges[k]->first)/10.0 );
      nAccepted.push_back(0);
   }

   // starting point (note that simulated annealing searches for maximum!)
   Double_t retFCN = - MinimizeFunction( xPars );
   Int_t    nCalls = 1;
   Double_t maxFCN = retFCN;
   diffFCN[1] = retFCN;

   // set initial temperature
   Double_t temperature = fInitialTemperature;

   //---------------------------------------------------------------------------------
   // perform the optimization until maximum iteration or required accuracy is reached
   Int_t nIteration = 0;

   // required for adaptive annealing
   std::pair<Int_t,Int_t> optPoint;
   std::pair<Int_t,Int_t> optPoint_previous;

   // begin with new temperature
   Bool_t continueWhile = kTRUE;
   while (continueWhile) {

      // iteration counter
      nIteration++;

      if (fUseAdaptiveTemperature) {
         if (nIteration>0 && ( TMath::Abs( ( (float)nIteration/2. ) - nIteration/2 ) < 0.01 ) ) {
            if (optPoint.first          > 0 && optPoint.second          == 0 && 
                optPoint_previous.first > 0 && optPoint_previous.second == 0) {
               if (stepWidth > 20) {
                  deltaT = sqrt( deltaT );
                  stepWidth = 0.5*stepWidth;
               }
            }
         }
      }

      // loop over the iterations before temperature reduction:
      for (Int_t m = 0; m < stepWidth; m++) {

         // adaptive annealing steps
         optPoint_previous.first  = optPoint.first;
         optPoint_previous.second = optPoint.second;
         optPoint.first  = 0;
         optPoint.second = 0;

         // loop over the accepted-function evaluations
         for (Int_t j = 0; j < fNFunLoops; j++) {

            // loop over parameters
            for (Int_t h = 0; h < npar; h++) {

               // randomize parameter h
               yPars[h] = xPars[h] + gRandom->Uniform(-1.0,1.0)*adaptiveErrors[h];

               // retry if randomising has thrown yPars out of its bounds
               while (yPars[h] < fRanges[h]->first || yPars[h] > fRanges[h]->second) {
                  yPars[h] = gRandom->Uniform(-2.0,2.0)*adaptiveErrors[h] + parameters[h];
               }

               // recover previous parameter setting 
               if (h >= 1) yPars[h-1] = xPars[h-1];

               // compute estimator for given variable set (again, searches for maximum)
               Double_t retFCNi = - MinimizeFunction( yPars ); 

               // too many function evaluations ? --> stop simulated annealing
               ++nCalls;
               if (nCalls >= fMaxCalls) {
                  for (i = 0; i < npar; i++) parameters[i] = bestParameters[i];
                  return -maxFCN;
               }

               // accept new solution if better FCN value
               if (retFCNi > retFCN) {
                  for (i = 0; i < npar; ++i) xPars[i] = yPars[i];
                  retFCN = retFCNi;
                  ++nAccepted[h];

                  // best FCN value so far, record as new best parameter set
                  if (retFCNi > maxFCN) {
                     for (i = 0; i < npar; ++i) bestParameters[i] = yPars[i];
                     
                     if (m <= stepWidth/2) optPoint.first++;
                     else                  optPoint.second++;
                     
                     maxFCN = retFCNi;
                  }
               } 
               else {

                  // original Metropolis et al. scheme to decide whether a solution is 
                  // accepted if FCN is worse than before;
                  // criterion following:
                  //   N. Metropolis, A. Rosenbluth, M. Rosenbluth, A. Teller, E. Teller, 
                  //   "Equation of State Calculations by Fast Computing Machines", 
                  //   J. Chem. Phys., 21, 6, 1087-1092, 1953
                  if (gRandom->Uniform(1.0) < this->GetPerturbationProbability( retFCNi, retFCN, 
                                                                                temperature )) {
                     for (i = 1; i < npar; ++i) xPars[i] = yPars[i];
                     retFCN = retFCNi;
                     ++nAccepted[h];
                  } 
               }
            } 
         } 
         // adjust the adaptiveErrors and the temperature
         // adjust adaptiveErrors to accept between 0.4 and 0.6 of the trial 
         // points at the given temperature
         for (i = 0; i < npar; ++i) {
            Double_t ratio = Double_t(nAccepted[i])/Double_t(fNFunLoops);
            // many FCN improvements, ie, far from minimum --> enhance error
            if      (ratio > 0.6) adaptiveErrors[i] *= (2.0 * (ratio - 0.6) / 0.4 + 1.0);
            // few FCN improvements, ie, closer to minimum --> reduce error
            else if (ratio < 0.4) adaptiveErrors[i] /= (2.0 * (0.4 - ratio) / 0.4 + 1.0);
            // else don't touch the error

            // the error shouldn't be larger than the full variable range
            if (adaptiveErrors[i] > fRanges[i]->second - fRanges[i]->first) 
               adaptiveErrors[i] = fRanges[i]->second - fRanges[i]->first;
         }

         // reset number of accepted functions
         for (i = 0; i < npar; ++i) nAccepted[i] = 0;
      } 

      //  terminate simulated annealing if appropriate 
      diffFCN[1] = retFCN;
      if (TMath::Abs(maxFCN - diffFCN[1]) < fEps) 
      for (i = 0; i < fNEps; ++i) if (TMath::Abs(retFCN - diffFCN[i]) > fEps) continueWhile = kTRUE;

      // more quite criteria
      if (TMath::Abs(maxFCN) < 1e-06   ) continueWhile = kFALSE;
      if (temperature < fMinTemperature) continueWhile = kFALSE;

      if (continueWhile) {
         // continue annealing
         temperature = deltaT * temperature;
         for (i = fNEps-1; i >= 1; --i) diffFCN[i] = diffFCN[i - 1];
         for (i = 0; i < npar; ++i) xPars[i] = bestParameters[i];
         retFCN = maxFCN;
         printf( "new temp: %g --> maxFCN: %f10.10\n" , temperature, maxFCN );
      }
   } // end of while loop

   // return best parameter set
   for (i = 0; i < npar; i++ ) parameters[i] = bestParameters[i];

   return - maxFCN;

}

Double_t TMVA::SimulatedAnnealingBase::GetPerturbationProbability( Double_t E, Double_t Eref, 
                                                                   Double_t temperature )
{
   return (temperature > 0) ? TMath::Exp( (E - Eref)/temperature ) : 0;
}
