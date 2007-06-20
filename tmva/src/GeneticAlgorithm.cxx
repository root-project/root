// @(#)root/tmva $Id: GeneticAlgorithm.cxx,v 1.7 2007/06/07 13:19:11 speckmayer Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::GeneticAlgorithm                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
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
//                                                                      
// Base definition for genetic algorithm                                
//_______________________________________________________________________

#include "Riostream.h"

#include "TMVA/GeneticAlgorithm.h"
#include "TMVA/Interval.h"
#include "TMVA/IFitterTarget.h"

namespace TMVA {
   const Bool_t GeneticAlgorithm__DEBUG__ = kFALSE;
}

ClassImp(TMVA::GeneticAlgorithm)
   
//_______________________________________________________________________
TMVA::GeneticAlgorithm::GeneticAlgorithm( IFitterTarget& target, Int_t populationSize, 
                                          const std::vector<Interval*>& ranges, UInt_t seed )
   : fFitterTarget( target ),
     fRanges( ranges ),
     fLogger( "GeneticAlgorithm" )
{
   // Constructor
   // Parameters: 
   //     int populationSize : defines the number of "Individuals" which are created and tested 
   //                          within one Generation (Iteration of the Evolution)
   //     vector<TMVA::Interval*> ranges : Interval holds the information of an interval, where the GetMin 
   //                          gets the low and GetMax gets the high constraint of the variable
   //                          the size of "ranges" is the number of coefficients which are optimised
   // Purpose: 
   //     Creates a random population with individuals of the size ranges.size()
   if( seed != 0 ){
      fPopulation.SetRandomSeed( seed );
   }

   for (std::vector<TMVA::Interval*>::const_iterator it = fRanges.begin(); it<fRanges.end(); it++) {
      fPopulation.AddFactor( (*it) );
   }

   fPopulation.CreatePopulation( populationSize );
   fPopulationSize = populationSize;
   fSexual      = kTRUE;
   fFirstTime   = kTRUE;
   fSpread      = 0.1;
   fMirror      = kTRUE;
   fConvCounter = -1;
   fConvValue   = 0.;
}

//_______________________________________________________________________
void TMVA::GeneticAlgorithm::Init()
{
   // calls evolution, but if it is not the first time. 
   // If it's the first time, the random population created by the
   // constructor is still not evaluated, .. therefore we wait for the 
   // second time init is called. 
   if ( fFirstTime ) fFirstTime = kFALSE;
   else {
      Evolution();
      fPopulation.ClearResults();
   }
}

//_______________________________________________________________________
Double_t TMVA::GeneticAlgorithm::NewFitness( Double_t /*oldValue*/, Double_t newValue )
{
   // if the "fitnessFunction" is called multiple times for one set of 
   // factors (because i.e. each event of a TTree has to be assessed with 
   // each set of Factors proposed by the Genetic Algorithm) the value 
   // of the current calculation has to be added(? or else) to the value
   // obtained up to now. 
   // example: some chi-square is calculated for every event, 
   // after every event the new chi-square (newValue) has to be simply
   // added to the oldValue. 
   //
   // this function has to be overridden eventually 
   // it might contain only the following return statement.
   //        return oldValue + newValue;
   return newValue;
}

//_______________________________________________________________________
Double_t TMVA::GeneticAlgorithm::CalculateFitness()
{
   // starts the evaluation of the fitness of all different individuals of
   // the population. 
   //
   // this function calls implicitly (many times) the "fitnessFunction" which
   // has been overridden by the user. 
   GeneticGenes *genes;
   Double_t fitness = 0;
   fPopulation.Reset();
   do {
      genes = fPopulation.GetGenes();
      fitness = NewFitness( fPopulation.GetFitness(), fFitterTarget.EstimatorFunction( genes->GetFactors()) );
   } while( fPopulation.SetFitness( genes, fitness, kTRUE ) );

   return fPopulation.GetFitness( 0 );
}

//_______________________________________________________________________
Double_t TMVA::GeneticAlgorithm::DoRenewFitness()
{
   // the fitness values of every individual is stored ..
   // if the fitness has been evaluated for many events, all the results are 
   // internally stored. 
   //
   // this function allows to loop through all results of all individuals. 
   // it calls implicitly the function "renewFitness" 
   // 
   // the right place to call this function would be at the end of one "Generation"
   // to set the fitness of every individual new depending on all the results it obtained 
   // in this generation. 
   GeneticGenes *genes;
   Double_t fitness = 0;
   fPopulation.Reset();
   do {
      genes = fPopulation.GetGenes();
      fitness = RenewFitness( genes->GetFactors(), genes->GetResults() );
      genes->GetResults().clear();
   } while( fPopulation.SetFitness( genes, fitness, kFALSE ) );
   return fPopulation.GetFitness( 0 );
}

//_______________________________________________________________________
Double_t TMVA::GeneticAlgorithm::RenewFitness( std::vector<Double_t> /*factors*/, 
                                               std::vector<Double_t> /* results */)
{
   // this function has to be overridden if "doRenewFitness" is called
   // Parameters: 
   //         vector< double > factors : in this vector the factors of a specific individual
   //                      are given. 
   //         vector< double > results : in this vector the results obtained by the given
   //                     coefficients are given. 
   // 
   // out of this information (the quality of the results) a new? value for the quality 
   // (fitness) of the set of factors has to be given back.
   return 1.0;
}

//_______________________________________________________________________
void TMVA::GeneticAlgorithm::Evolution()
{
   // this function is called from "init" and controls the evolution of the 
   // individuals. 
   // the function can be overridden to change the parameters for mutation rate
   // sexual reproduction and so on.
   if (fSexual) {
      fPopulation.MakeCopies( 5 );  
      fPopulation.MakeChildren();
      fPopulation.NextGeneration();

      fPopulation.Mutate( 10, 3, kTRUE, fSpread, fMirror );
      fPopulation.Mutate( 40, fPopulation.GetPopulationSize()*3/4 );
   } 
   else {
      fPopulation.MakeCopies( 3 );  
      fPopulation.MakeMutants(100,true, 0.1, true);
      fPopulation.NextGeneration();
   }
}

//_______________________________________________________________________
Double_t TMVA::GeneticAlgorithm::SpreadControl( Int_t ofSteps, Int_t successSteps, Double_t factor )
{
   // this function provides the ability to change the stepSize of a mutation according to
   // the success of the last generations. 
   // 
   // Parameters:
   //      int ofSteps :  = if OF the number of STEPS given in this variable (ofSteps)
   //      int successSteps : >sucessSteps Generations could improve the result
   //      double factor : than multiply the stepSize ( spread ) by this factor
   // (if ofSteps == successSteps nothing is changed, if ofSteps < successSteps, the spread
   // is divided by the factor) 
   //
   // using this function one can increase the stepSize of the mutation when we have 
   // good success (to pass fast through the easy phase-space) and reduce the stepSize
   // if we are in a difficult "territory" of the phase-space. 
   //

   // < is valid for "less" comparison
   if ( fPopulation.GetFitness( 0 ) < fLastResult || fSuccessList.size() <=0 ) { 
      fLastResult = fPopulation.GetFitness( 0 );
      fSuccessList.push_front( 1 ); // it got better
   } 
   else {
      fSuccessList.push_front( 0 ); // it stayed the same
   }
   Int_t n = 0;
   Int_t sum = 0;
   std::deque<Int_t>::iterator vec = fSuccessList.begin();
   for (; vec<fSuccessList.end() ; vec++) {
      sum += *vec;
      n++;
   }

   if ( n >= ofSteps ) {
      fSuccessList.pop_back();
      if ( sum > successSteps ) { // too much success
         fSpread /= factor;
         if (GeneticAlgorithm__DEBUG__) fLogger << kINFO << ">" << flush;
      }
      else if ( sum == successSteps ) { // on the optimal path
         if (GeneticAlgorithm__DEBUG__) fLogger << "=" << flush;
      }
      else {        // not very successful
         fSpread *= factor;
         if (GeneticAlgorithm__DEBUG__) fLogger << "<" << flush;
      }
   }

   return fSpread;
}

//_______________________________________________________________________
Bool_t TMVA::GeneticAlgorithm::HasConverged( Int_t steps, Double_t improvement )
{
   // gives back true if the last "steps" steps have lead to an improvement of the
   // "fitness" of the "individuals" of at least "improvement"
   // 
   // this gives a simple measure of if the fitness of the individuals is
   // converging and no major improvement is to be expected soon. 
   //
   if (fConvCounter < 0) {
      fConvValue = fPopulation.GetFitness( 0 );
   }
   if (TMath::Abs(fPopulation.GetFitness( 0 )-fConvValue) <= improvement || steps<0) {
      fConvCounter ++;
   } 
   else {
      fConvCounter = 0;
      fConvValue = fPopulation.GetFitness( 0 );
   }
   if (GeneticAlgorithm__DEBUG__) fLogger << "." << flush;
   if (fConvCounter < steps) return kFALSE;
   return kTRUE;
}

//_______________________________________________________________________
void TMVA::GeneticAlgorithm::Finalize()
{
   // nothing so far...
}

