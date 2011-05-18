// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Krzysztof Danielowski, Kamil Kraszewski, Maciej Kruk

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SimulatedAnnealing                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Krzysztof Danielowski <danielow@cern.ch>       - IFJ & AGH, Poland        *
 *      Kamil Kraszewski      <kalq@cern.ch>           - IFJ & UJ, Poland         *
 *      Maciej Kruk           <mkruk@cern.ch>          - IFJ & AGH, Poland        *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      IFJ-Krakow, Poland                                                        *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// Implementation of Simulated Annealing fitter
//_______________________________________________________________________
#include "TMVA/SimulatedAnnealing.h"

#include "TRandom3.h"
#include "TMath.h"

#include "TMVA/Interval.h"
#include "TMVA/IFitterTarget.h"
#include "TMVA/GeneticRange.h"
#include "TMVA/Timer.h"
#include "TMVA/MsgLogger.h"

ClassImp(TMVA::SimulatedAnnealing)

//_______________________________________________________________________
TMVA::SimulatedAnnealing::SimulatedAnnealing( IFitterTarget& target, const std::vector<Interval*>& ranges )
   : fKernelTemperature     (kIncreasingAdaptive),
     fFitterTarget          ( target ),
     fRandom                ( new TRandom3(100) ),
     fRanges                ( ranges ),
     fMaxCalls              ( 500000 ),
     fInitialTemperature    ( 1000 ),
     fMinTemperature        ( 0 ),
     fEps                   ( 1e-10 ),
     fTemperatureScale      ( 0.06 ),
     fAdaptiveSpeed         ( 1.0 ),
     fTemperatureAdaptiveStep( 0.0 ),
     fUseDefaultScale       ( kFALSE ),
     fUseDefaultTemperature ( kFALSE ),
     fLogger( new MsgLogger("SimulatedAnnealing") ),
     fProgress(0.0)
{
   // constructor
   fKernelTemperature = kIncreasingAdaptive;
}

//_______________________________________________________________________
void TMVA::SimulatedAnnealing::SetOptions( Int_t    maxCalls,
                                           Double_t initialTemperature,
                                           Double_t minTemperature,
                                           Double_t eps,
                                           TString  kernelTemperatureS,
                                           Double_t temperatureScale,
                                           Double_t adaptiveSpeed,
                                           Double_t temperatureAdaptiveStep,
                                           Bool_t   useDefaultScale,
                                           Bool_t   useDefaultTemperature)   
{
   // option setter

   fMaxCalls = maxCalls;
   fInitialTemperature = initialTemperature;
   fMinTemperature = minTemperature;
   fEps = eps;

   if      (kernelTemperatureS == "IncreasingAdaptive") {
      fKernelTemperature = kIncreasingAdaptive; 
      Log() << kINFO << "Using increasing adaptive algorithm" << Endl;
   }
   else if (kernelTemperatureS == "DecreasingAdaptive") {
      fKernelTemperature = kDecreasingAdaptive; 
      Log() << kINFO << "Using decreasing adaptive algorithm" << Endl;
   }
   else if (kernelTemperatureS == "Sqrt") {
      fKernelTemperature = kSqrt; 
      Log() << kINFO << "Using \"Sqrt\" algorithm" << Endl;
   }
   else if (kernelTemperatureS == "Homo") {
      fKernelTemperature = kHomo; 
      Log() << kINFO << "Using \"Homo\" algorithm" << Endl;
   }
   else if (kernelTemperatureS == "Log") {
      fKernelTemperature = kLog;  
      Log() << kINFO << "Using \"Log\" algorithm" << Endl;
   }
   else if (kernelTemperatureS == "Sin") {
      fKernelTemperature = kSin;
      Log() << kINFO << "Using \"Sin\" algorithm" << Endl;
   }

   fTemperatureScale        = temperatureScale;
   fAdaptiveSpeed           = adaptiveSpeed;
   fTemperatureAdaptiveStep = temperatureAdaptiveStep;

   fUseDefaultScale         = useDefaultScale;
   fUseDefaultTemperature   = useDefaultTemperature;
}
//_______________________________________________________________________
TMVA::SimulatedAnnealing::~SimulatedAnnealing()
{
   // destructor
}

//_______________________________________________________________________
void TMVA::SimulatedAnnealing::FillWithRandomValues( std::vector<Double_t>& parameters )
{
   // random starting parameters
   for (UInt_t rIter = 0; rIter < parameters.size(); rIter++) {
      parameters[rIter] = fRandom->Uniform(0.0,1.0)*(fRanges[rIter]->GetMax() - fRanges[rIter]->GetMin()) + fRanges[rIter]->GetMin();
   }
}

//_______________________________________________________________________
void TMVA::SimulatedAnnealing::ReWriteParameters( std::vector<Double_t>& from, std::vector<Double_t>& to)
{
   // copy parameters
   for (UInt_t rIter = 0; rIter < from.size(); rIter++) to[rIter] = from[rIter];
}

//_______________________________________________________________________
void TMVA::SimulatedAnnealing::GenerateNeighbour( std::vector<Double_t>& parameters, std::vector<Double_t>& oldParameters, 
                                                  Double_t currentTemperature )
{
   // generate adjacent parameters
   ReWriteParameters( parameters, oldParameters );

   for (UInt_t rIter=0;rIter<parameters.size();rIter++) {
      Double_t uni,distribution,sign;
      do {
         uni = fRandom->Uniform(0.0,1.0);
         sign = (uni - 0.5 >= 0.0) ? (1.0) : (-1.0);
         distribution = currentTemperature * (TMath::Power(1.0 + 1.0/currentTemperature, TMath::Abs(2.0*uni - 1.0)) -1.0)*sign;
         parameters[rIter] = oldParameters[rIter] +  (fRanges[rIter]->GetMax()-fRanges[rIter]->GetMin())*0.1*distribution;
      }
      while (parameters[rIter] < fRanges[rIter]->GetMin() || parameters[rIter] > fRanges[rIter]->GetMax() );
   }
}
//_______________________________________________________________________
std::vector<Double_t> TMVA::SimulatedAnnealing::GenerateNeighbour( std::vector<Double_t>& parameters, Double_t currentTemperature )
{
   // generate adjacent parameters
   std::vector<Double_t> newParameters( fRanges.size() );   

   for (UInt_t rIter=0; rIter<parameters.size(); rIter++) {
      Double_t uni,distribution,sign;
      do {
         uni = fRandom->Uniform(0.0,1.0);
         sign = (uni - 0.5 >= 0.0) ? (1.0) : (-1.0);
         distribution = currentTemperature * (TMath::Power(1.0 + 1.0/currentTemperature, TMath::Abs(2.0*uni - 1.0)) -1.0)*sign;
         newParameters[rIter] = parameters[rIter] +  (fRanges[rIter]->GetMax()-fRanges[rIter]->GetMin())*0.1*distribution;
      }
      while (newParameters[rIter] < fRanges[rIter]->GetMin() || newParameters[rIter] > fRanges[rIter]->GetMax() );
   }

   return newParameters;
}

//_______________________________________________________________________
void TMVA::SimulatedAnnealing::GenerateNewTemperature( Double_t& currentTemperature, Int_t Iter )
{
   // generate new temperature
   if      (fKernelTemperature == kSqrt) {
         currentTemperature = fInitialTemperature/(Double_t)TMath::Sqrt(Iter+2) * fTemperatureScale;
   }
   else if (fKernelTemperature == kLog) {
      currentTemperature = fInitialTemperature/(Double_t)TMath::Log(Iter+2) * fTemperatureScale;
   }
   else if (fKernelTemperature == kHomo) {
      currentTemperature = fInitialTemperature/(Double_t)(Iter+2) * fTemperatureScale;
   }
   else if (fKernelTemperature == kSin) {
      currentTemperature = (TMath::Sin( (Double_t)Iter / fTemperatureScale ) + 1.0 )/ (Double_t)(Iter+1.0) * fInitialTemperature + fEps;
   }
   else if (fKernelTemperature == kGeo) {
      currentTemperature = currentTemperature*fTemperatureScale;
   }
   else if (fKernelTemperature == kIncreasingAdaptive) {
      currentTemperature = fMinTemperature + fTemperatureScale*TMath::Log(1.0+fProgress*fAdaptiveSpeed);
   }
   else if (fKernelTemperature == kDecreasingAdaptive) {
      currentTemperature = currentTemperature*fTemperatureScale;
   }
   else Log() << kFATAL << "No such kernel!" << Endl;
}

//________________________________________________________________________
Bool_t TMVA::SimulatedAnnealing::ShouldGoIn( Double_t currentFit, Double_t localFit, Double_t currentTemperature )
{
   // result checker
   if (currentTemperature < fEps) return kFALSE;
   Double_t lim  = TMath::Exp( -TMath::Abs( currentFit - localFit ) / currentTemperature );
   Double_t prob = fRandom->Uniform(0.0, 1.0);
   return (prob < lim) ? kTRUE : kFALSE;
}

//_______________________________________________________________________
void TMVA::SimulatedAnnealing::SetDefaultScale()
{
   // setting of default scale
   if      (fKernelTemperature == kSqrt) fTemperatureScale = 1.0;
   else if (fKernelTemperature == kLog)  fTemperatureScale = 1.0;
   else if (fKernelTemperature == kHomo) fTemperatureScale = 1.0;
   else if (fKernelTemperature == kSin)  fTemperatureScale = 20.0;
   else if (fKernelTemperature == kGeo)  fTemperatureScale = 0.99997;
   else if (fKernelTemperature == kDecreasingAdaptive) {
      fTemperatureScale = 1.0;
      while (TMath::Abs(TMath::Power(fTemperatureScale,fMaxCalls) * fInitialTemperature - fMinTemperature) >
             TMath::Abs(TMath::Power(fTemperatureScale-0.000001,fMaxCalls) * fInitialTemperature - fMinTemperature)) {
         fTemperatureScale -= 0.000001;
      }
   }
   else if (fKernelTemperature == kIncreasingAdaptive) fTemperatureScale = 0.15*( 1.0 / (Double_t)(fRanges.size() ) );
   else Log() << kFATAL << "No such kernel!" << Endl;
}

//_______________________________________________________________________
Double_t TMVA::SimulatedAnnealing::GenerateMaxTemperature( std::vector<Double_t>& parameters  )
{
   // maximum temperature
   Int_t equilibrium;
   Bool_t stopper = 0; 
   Double_t t, dT, cold, delta, deltaY, y, yNew, yBest, yOld;
   std::vector<Double_t> x( fRanges.size() ), xNew( fRanges.size() ), xBest( fRanges.size() ), xOld( fRanges.size() );
   t = fMinTemperature;
   deltaY = cold = 0.0;
   dT = fTemperatureAdaptiveStep;
   for (UInt_t rIter = 0; rIter < x.size(); rIter++)
      x[rIter] = ( fRanges[rIter]->GetMax() + fRanges[rIter]->GetMin() ) / 2.0;
   y = yBest = 1E10;
   for (Int_t i=0; i<fMaxCalls/50; i++) {
      if ((i>0) && (deltaY>0.0)) {
         cold = deltaY;
         stopper = 1;
      }
      t += dT*i;
      x = xOld = GenerateNeighbour(x,t);
      y = yOld = fFitterTarget.EstimatorFunction( xOld );   
      equilibrium = 0;
      for ( Int_t k=0; (k<30) && (equilibrium<=12); k++ ) {
         xNew = GenerateNeighbour(x,t);
         //"energy"
         yNew = fFitterTarget.EstimatorFunction( xNew );
         deltaY = yNew - y;
         if (deltaY < 0.0) {     // keep xnew if energy is reduced
            std::swap(x,xNew);
            std::swap(y,yNew);
            if (y < yBest) {
               xBest = x;
               yBest = y;
            }
            delta = TMath::Abs( deltaY );
            if      (y    != 0.0) delta /= y;
            else if (yNew != 0.0) delta /= y;

            // equilibrium is defined as a 10% or smaller change in 10 iterations 
            if (delta < 0.1) equilibrium++;
            else             equilibrium = 0;
         }
         else equilibrium++;
      }

      // "energy"
      yNew = fFitterTarget.EstimatorFunction( xNew ); 
      deltaY = yNew - yOld;
      if ( (deltaY < 0.0 )&&( yNew < yBest)) {
         xBest=x;
         yBest = yNew;
      }
      y = yNew;
      if ((stopper) && (deltaY >= (100.0 * cold))) break;  // phase transition with another parameter to change
   }
   parameters = xBest;
   return t;
}

//_______________________________________________________________________
Double_t TMVA::SimulatedAnnealing::Minimize( std::vector<Double_t>& parameters )
{
   // minimisation algorithm
   std::vector<Double_t> bestParameters(fRanges.size());
   std::vector<Double_t> oldParameters (fRanges.size());

   Double_t currentTemperature, bestFit, currentFit;
   Int_t optimizeCalls, generalCalls, equals;

   equals = 0;

   if (fUseDefaultTemperature) {
      if (fKernelTemperature == kIncreasingAdaptive) {
         fMinTemperature = currentTemperature = 1e-06; 
         FillWithRandomValues( parameters );
      }
      else fInitialTemperature = currentTemperature = GenerateMaxTemperature( parameters );
   }
   else {
      if (fKernelTemperature == kIncreasingAdaptive)
         currentTemperature = fMinTemperature; 
      else
         currentTemperature = fInitialTemperature;
      FillWithRandomValues( parameters ); 
   }

   if (fUseDefaultScale) SetDefaultScale();

   Log() << kINFO
           << "Temperatur scale = "      << fTemperatureScale  
           << ", current temperature = " << currentTemperature  << Endl;

   bestParameters = parameters;
   bestFit        = currentFit = fFitterTarget.EstimatorFunction( bestParameters );

   optimizeCalls = fMaxCalls/100;             //use 1% calls to optimize best founded minimum
   generalCalls  = fMaxCalls - optimizeCalls; //and 99% calls to found that one
   fProgress = 0.0;

   Timer timer( fMaxCalls, fLogger->GetSource().c_str() );

   for (Int_t sample = 0; sample < generalCalls; sample++) {
      GenerateNeighbour( parameters, oldParameters, currentTemperature );
      Double_t localFit = fFitterTarget.EstimatorFunction( parameters );
      
      if (localFit < currentFit || TMath::Abs(currentFit-localFit) < fEps) { // if not worse than last one
         if (TMath::Abs(currentFit-localFit) < fEps) { // if the same as last one
            equals++;
            if (equals >= 3) //if we still at the same level, we should increase temperature
               fProgress+=1.0;
         }
         else {
            fProgress = 0.0;
            equals = 0;
         }
         
         currentFit = localFit;
         
         if (currentFit < bestFit) {
            ReWriteParameters( parameters, bestParameters );
            bestFit = currentFit;
         }
      }
      else {
         if (!ShouldGoIn(localFit, currentFit, currentTemperature))
            ReWriteParameters( oldParameters, parameters );
         else
            currentFit = localFit;
         
         fProgress+=1.0;
         equals = 0;
      }
      
      GenerateNewTemperature( currentTemperature, sample );
      
      if ((fMaxCalls<100) || sample%Int_t(fMaxCalls/100.0) == 0) timer.DrawProgressBar( sample );
   }

   // get elapsed time   
   Log() << kINFO << "Elapsed time: " << timer.GetElapsedTime() 
           << "                            " << Endl;  
   
   // supose this minimum is the best one, now just try to improve it

   Double_t startingTemperature = fMinTemperature*(fRanges.size())*2.0; 
   currentTemperature = startingTemperature;

   Int_t changes = 0;
   for (Int_t sample=0;sample<optimizeCalls;sample++) {
      GenerateNeighbour( parameters, oldParameters, currentTemperature );
      Double_t localFit = fFitterTarget.EstimatorFunction( parameters );
      
      if (localFit < currentFit) { //if better than last one
         currentFit = localFit;
         changes++;
         
         if (currentFit < bestFit) {
            ReWriteParameters( parameters, bestParameters );
            bestFit = currentFit;
         }
      }
      else ReWriteParameters( oldParameters, parameters ); //we never try worse parameters

      currentTemperature-=(startingTemperature - fEps)/optimizeCalls;
   }

   ReWriteParameters( bestParameters, parameters );

   return bestFit; 
}

