// @(#)root/tmva $Id$ 
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MCFitter                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer    <Joerg.Stelzer@cern.ch>  - CERN, Switzerland             *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
/*
  MCFitter
  
  Fitter using Monte Carlo sampling of parameters 
*/
//_______________________________________________________________________


#include "TMVA/MCFitter.h"
#include "TMVA/GeneticRange.h"
#include "TMVA/Interval.h"
#include "TMVA/Timer.h"
#include "TRandom3.h"

ClassImp(TMVA::MCFitter)

//_______________________________________________________________________
TMVA::MCFitter::MCFitter( IFitterTarget& target, 
                          const TString& name, 
                          const std::vector<Interval*>& ranges, 
                          const TString& theOption ) 
   : TMVA::FitterBase( target, name, ranges, theOption ),
     fSamples( 0 ),
     fSigma  ( 1 ),
     fSeed   ( 0 )
{
   // constructor
   DeclareOptions();
   ParseOptions();
}            

//_______________________________________________________________________
void TMVA::MCFitter::DeclareOptions() 
{
   // Declare MCFitter options
   DeclareOptionRef( fSamples = 100000, "SampleSize", "Number of Monte Carlo events in toy sample" );  
   DeclareOptionRef( fSigma   = -1.0,   "Sigma", 
                    "If > 0: new points are generated according to Gauss around best value and with \"Sigma\" in units of interval length" );  
   DeclareOptionRef( fSeed    = 100,    "Seed",       "Seed for the random generator (0 takes random seeds)" );  
}

//_______________________________________________________________________
void TMVA::MCFitter::SetParameters( Int_t samples )
{
   // set MC fitter configuration parameters
   fSamples = samples;
}

//_______________________________________________________________________
Double_t TMVA::MCFitter::Run( std::vector<Double_t>& pars )
{
   // Execute fitting
   Log() << kINFO << "<MCFitter> Sampling, please be patient ..." << Endl;
   
   // sanity check
   if ((Int_t)pars.size() != GetNpars())
      Log() << kFATAL << "<Run> Mismatch in number of parameters: "
              << GetNpars() << " != " << pars.size() << Endl;

   // timing of MC
   Timer timer( fSamples, GetName() ); 
   
   std::vector<Double_t> parameters;
   std::vector<Double_t> bestParameters;

   TRandom3*rnd = new TRandom3( fSeed );
   rnd->Uniform(0.,1.);
      
   std::vector<TMVA::GeneticRange*> rndRanges;

   // initial parameters (given by argument) are ignored
   std::vector< TMVA::Interval* >::const_iterator rIt; 
   Double_t val;
   for (rIt = fRanges.begin(); rIt<fRanges.end(); rIt++) {
      rndRanges.push_back( new TMVA::GeneticRange( rnd, (*rIt) ) );
      val = rndRanges.back()->Random();
      parameters.push_back( val );
      bestParameters.push_back( val );
   }

   std::vector<Double_t>::iterator parIt;
   std::vector<Double_t>::iterator parBestIt;
      
   Double_t estimator = 0;
   Double_t bestFit   = 0;

   // loop over all MC samples
   for (Int_t sample = 0; sample < fSamples; sample++) {

      // dice the parameters
      parIt = parameters.begin();
      if (fSigma > 0.0) {
         parBestIt = bestParameters.begin();
         for (std::vector<TMVA::GeneticRange*>::iterator rndIt = rndRanges.begin(); rndIt<rndRanges.end(); rndIt++) {
            (*parIt) = (*rndIt)->Random( kTRUE, (*parBestIt), fSigma );
            parIt++;
            parBestIt++;
         }
      }
      else {
         for (std::vector<TMVA::GeneticRange*>::iterator rndIt = rndRanges.begin(); rndIt<rndRanges.end(); rndIt++) {
            (*parIt) = (*rndIt)->Random();
            parIt++;
         }
      }

      // test the estimator value for the parameters
      estimator = EstimatorFunction( parameters );

      // if the estimator ist better (=smaller), take the new parameters as the best ones
      if (estimator < bestFit || sample==0) {
         bestFit = estimator;
         bestParameters.swap( parameters );
      }

      // whats the time please?
      if ((fSamples<100) || sample%Int_t(fSamples/100.0) == 0) timer.DrawProgressBar( sample );
   }
   pars.swap( bestParameters ); // return best parameters found

   // get elapsed time
   Log() << kINFO << "Elapsed time: " << timer.GetElapsedTime() 
           << "                           " << Endl;  
   
   return bestFit;
}
