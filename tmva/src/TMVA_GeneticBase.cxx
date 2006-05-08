// @(#)root/tmva $Id: TMVA_GeneticBase.cxx,v 1.2 2006/05/08 12:59:13 brun Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GeneticBase                                                      *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-KP Heidelberg, Germany,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Base definition for genetic algorithm                                
//                                                                      
//_______________________________________________________________________

#include "TMVA_GeneticBase.h"
#include <iostream>

#define TMVA_GeneticBase__DEBUG__ kFALSE

using namespace std;

ClassImp(TMVA_GeneticBase)
   
//_______________________________________________________________________
TMVA_GeneticBase::TMVA_GeneticBase( Int_t populationSize, vector<LowHigh*> ranges ) 
{
  for( vector< LowHigh* >::iterator it = ranges.begin(); it<ranges.end(); it++ ) {
    fPopulation.AddFactor( (*it)->first, (*it)->second );
  }

  fPopulation.CreatePopulation( populationSize );
  fSexual = kTRUE;
  fFirstTime = kTRUE;
  fSpread = 0.1;
  fMirror = kTRUE;
  fConvCounter = -1;
  fConvValue = 0.;
}

//_______________________________________________________________________
void TMVA_GeneticBase::Init()
{
  if( fFirstTime ) fFirstTime = kFALSE;
  else {
    Evolution();
    fPopulation.ClearResults();
  }
}

//_______________________________________________________________________
Double_t TMVA_GeneticBase::Calc() 
{
  CalculateFitness( );
  return 0;
}

//_______________________________________________________________________
Double_t TMVA_GeneticBase::FitnessFunction( vector< Double_t > /* factors */)
{
  return 1;
}

//_______________________________________________________________________
Double_t TMVA_GeneticBase::NewFitness( Double_t /*oldValue*/, Double_t newValue )
{
  return newValue;
}

//_______________________________________________________________________
Double_t TMVA_GeneticBase::CalculateFitness()
{
  TMVA_GeneticGenes *genes;
  Double_t fitness = 0;
  fPopulation.Reset();
  do {
    genes = fPopulation.GetGenes();
    fitness = NewFitness( fPopulation.GetFitness(), FitnessFunction( genes->fFactors) );
  } while( fPopulation.SetFitness( genes, fitness, kTRUE ) );

  return fPopulation.GetFitness( 0 );
}

//_______________________________________________________________________
Double_t TMVA_GeneticBase::DoRenewFitness()
{
  TMVA_GeneticGenes *genes;
  Double_t fitness = 0;
  fPopulation.Reset();
  do {
    genes = fPopulation.GetGenes();
    fitness = RenewFitness( genes->fFactors, genes->fResults );
    genes->fResults.clear();
  } while( fPopulation.SetFitness( genes, fitness, kFALSE ) );
  return fPopulation.GetFitness( 0 );
}

//_______________________________________________________________________
Double_t TMVA_GeneticBase::RenewFitness( vector<Double_t>  /*factors*/, vector<Double_t> /* results */)
{
  return 1.0;
}

//_______________________________________________________________________
void TMVA_GeneticBase::Evolution()
{
  if( fSexual ) {
    fPopulation.MakeChildren();
    fPopulation.Mutate( 10, 1, kTRUE, fSpread, fMirror );
    fPopulation.Mutate( 40, fPopulation.fPopulationSize*3/4 );
  }
  else{
    fPopulation.MakeMutants();
  }
}

//_______________________________________________________________________
Double_t TMVA_GeneticBase::SpreadControl( Int_t ofSteps, Int_t successSteps, Double_t factor )
{
  // < is valid for "less" comparison
  if( fPopulation.GetFitness( 0 ) < fLastResult || fSuccessList.size() <=0 ){ 
    fLastResult = fPopulation.GetFitness( 0 );
    fSuccessList.push_front( 1 ); // it got better
  } else {
    fSuccessList.push_front( 0 ); // it stayed the same
  }
  Int_t n = 0;
  Int_t sum = 0;
  std::deque<Int_t>::iterator vec = fSuccessList.begin();
  for( ; vec<fSuccessList.end() ; vec++ ){
    sum += *vec;
    n++;
  }

  if( n >= ofSteps ){
    fSuccessList.pop_back();
    if( sum > successSteps ){ // too much success
      fSpread /= factor;
      if (TMVA_GeneticBase__DEBUG__) cout << ">"; cout.flush();
    }
    else if( sum == successSteps ){ // on the optimal path
      if (TMVA_GeneticBase__DEBUG__) cout << "="; cout.flush();
    }
    else{	// not very successful
      fSpread *= factor;
      if (TMVA_GeneticBase__DEBUG__) cout << "<"; cout.flush();
    }
  }

  return fSpread;
}

//_______________________________________________________________________
Bool_t TMVA_GeneticBase::HasConverged( Int_t steps, Double_t improvement )
{
  if( fConvCounter < 0 ) {
    fConvValue = fPopulation.GetFitness( 0 );
  }
  if( TMath::Abs(fPopulation.GetFitness( 0 )-fConvValue) <= improvement || steps<0){
    fConvCounter ++;
  } 
  else {
    fConvCounter = 0;
    fConvValue = fPopulation.GetFitness( 0 );
  }
  if (TMVA_GeneticBase__DEBUG__) cout << "."; cout.flush();
  if( fConvCounter < steps ) return kFALSE;
  return kTRUE;
}


//_______________________________________________________________________
void TMVA_GeneticBase::Finalize()
{}

