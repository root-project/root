// @(#)root/tmva $Id: TMVA_GeneticBase.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $    
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
 * File and Version Information:                                                  *
 * $Id: TMVA_GeneticBase.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $
 **********************************************************************************/

#include "TMVA_GeneticBase.h"
#include <iostream>

#define TMVA_GeneticBase__DEBUG__ kFALSE

using namespace std;

//_______________________________________________________________________
//                                                                      
// Base definition for genetic algorithm                                
//                                                                      
//_______________________________________________________________________

TMVA_GeneticBase::TMVA_GeneticBase( Int_t populationSize, vector<LowHigh*> ranges ) 
{
  for( vector< LowHigh* >::iterator it = ranges.begin(); it<ranges.end(); it++ ) {
    population.addFactor( (*it)->first, (*it)->second );
  }

  population.createPopulation( populationSize );
  sexual = kTRUE;
  firstTime = kTRUE;
  spread = 0.1;
  mirror = kTRUE;
  convCounter = -1;
  convValue = 0.;
}

void TMVA_GeneticBase::init()
{
  if( firstTime ) firstTime = kFALSE;
  else {
    evolution();
    population.clearResults();
  }
}

Double_t TMVA_GeneticBase::calc() 
{
  calculateFitness( );
  return 0;
}

Double_t TMVA_GeneticBase::fitnessFunction( vector< Double_t > /* factors */)
{
  return 1;
}

Double_t TMVA_GeneticBase::newFitness( Double_t /*oldValue*/, Double_t newValue )
{
  return newValue;
}

Double_t TMVA_GeneticBase::calculateFitness()
{
  TMVA_GeneticGenes *genes;
  Double_t fitness = 0;
  population.reset();
  do {
    genes = population.getGenes();
    fitness = newFitness( population.getFitness(), fitnessFunction( genes->factors) );
  } while( population.setFitness( genes, fitness, kTRUE ) );

  return population.getFitness( 0 );
}

Double_t TMVA_GeneticBase::doRenewFitness()
{
  TMVA_GeneticGenes *genes;
  Double_t fitness = 0;
  population.reset();
  do {
    genes = population.getGenes();
    fitness = renewFitness( genes->factors, genes->results );
    genes->results.clear();
  } while( population.setFitness( genes, fitness, kFALSE ) );
  return population.getFitness( 0 );
}

Double_t TMVA_GeneticBase::renewFitness( vector<Double_t>  /*factors*/, vector<Double_t> /* results */)
{
  return 1.0;
}

void TMVA_GeneticBase::evolution()
{
  if( sexual ) {
    population.makeChildren();
    population.mutate( 10, 1, kTRUE, spread, mirror );
    population.mutate( 40, population.populationSize*3/4 );
  }
  else{
    population.makeMutants();
  }
}

Double_t TMVA_GeneticBase::spreadControl( Int_t ofSteps, Int_t successSteps, Double_t factor )
{
  // < is valid for "less" comparison
  if( population.getFitness( 0 ) < lastResult || successList.size() <=0 ){ 
    lastResult = population.getFitness( 0 );
    successList.push_front( 1 ); // it got better
  } else {
    successList.push_front( 0 ); // it stayed the same
  }
  Int_t n = 0;
  Int_t sum = 0;
  std::deque<Int_t>::iterator vec = successList.begin();
  for( ; vec<successList.end() ; vec++ ){
    sum += *vec;
    n++;
  }

  if( n >= ofSteps ){
    successList.pop_back();
    if( sum > successSteps ){ // too much success
      spread /= factor;
      if (TMVA_GeneticBase__DEBUG__) cout << ">"; cout.flush();
    }
    else if( sum == successSteps ){ // on the optimal path
      if (TMVA_GeneticBase__DEBUG__) cout << "="; cout.flush();
    }
    else{	// not very successful
      spread *= factor;
      if (TMVA_GeneticBase__DEBUG__) cout << "<"; cout.flush();
    }
  }

  return spread;
}

Bool_t TMVA_GeneticBase::hasConverged( Int_t steps, Double_t improvement )
{
  if( convCounter < 0 ) {
    convValue = population.getFitness( 0 );
  }
  if( TMath::Abs(population.getFitness( 0 )-convValue) <= improvement || steps<0){
    convCounter ++;
  } 
  else {
    convCounter = 0;
    convValue = population.getFitness( 0 );
  }
  if (TMVA_GeneticBase__DEBUG__) cout << "."; cout.flush();
  if( convCounter < steps ) return kFALSE;
  return kTRUE;
}


void TMVA_GeneticBase::finalize()
{}

