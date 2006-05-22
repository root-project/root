// @(#)root/tmva $Id: GeneticPopulation.cxx,v 1.6 2006/05/21 22:46:37 andreas.hoecker Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::GeneticPopulation                                               *
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

#include "TMVA/GeneticPopulation.h"
#include "TMVA/GeneticGenes.h"
#include <iostream>
#include "TSystem.h"
#include "TRandom3.h"
#include "TH1.h"
#include <sstream>

using namespace std;

ClassImp(TMVA::GeneticPopulation)
   
//_______________________________________________________________________
//                                                                      
// Population definition for genetic algorithm                          
//                                                                      
//_______________________________________________________________________

//_______________________________________________________________________
TMVA::GeneticPopulation::GeneticPopulation()
{
  // Constructor
  // create a randomGenerator for this population and set a seed
  // create the genePools
  //
  fRandomGenerator = new TRandom3(0); //please check
  // gSystem->Sleep(2);
  // fRandomGenerator->SetSeed( (long)gSystem->Now() );
  // seed of random-generator to machine clock
  // --> if called twice within a second, the generated numbers will be the same.

  fRandomGenerator->Uniform(0.,1.);
  fGenePool    = new multimap<Double_t, TMVA::GeneticGenes>();
  fNewGenePool = new multimap<Double_t, TMVA::GeneticGenes>();

  fCounterFitness = 0;
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::CreatePopulation( Int_t size )
{
  // create a Population of individuals with the population size given
  // in the parameter
  //--> every coefficient gets a random value within the constraints
  // provided by the user
  //
  fPopulationSize = size;
  fGenePool->clear();
  fNewGenePool->clear();

  vector< TMVA::GeneticRange* >::iterator rIt;
  vector< Double_t > newEntry;

  for( Int_t i=0; i<fPopulationSize; i++ ){
    newEntry.clear();
    for( rIt = fRanges.begin(); rIt<fRanges.end(); rIt++ ){
      newEntry.push_back( (*rIt)->Random() );
    }
    entry e(0, TMVA::GeneticGenes( newEntry) );
    fGenePool->insert( e );
  }

  fCounter = fGenePool->begin();
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::AddPopulation( TMVA::GeneticPopulation *strangers )
{
  // allows to connect two populations (using the same number of coefficients
  // with the same ranges) 
  // this allows to calculate several populations on the same phase-space
  // or on different parts of the same phase-space and combine them afterwards
  // this improves the global outcome.
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it;
  for( it = strangers->fGenePool->begin(); it != strangers->fGenePool->end(); it++ ) {
    GiveHint( it->second.GetFactors(), it->first );
  }
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::TrimPopulation( )
{
  // after adding another population or givingHint, the true size of the population
  // may be bigger than the size which was given at createPopulation
  // trimPopulation should be called (if necessary) after having checked the 
  // individuals fitness with calculateFitness
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it = fGenePool->begin() ;
  for( Int_t i=0; i<fPopulationSize; i++ ) it++;
  fGenePool->erase( it, fGenePool->end()-- );
}

void TMVA::GeneticPopulation::MakeChildren() 
{
  // does what the name says,... it creates children out of members of the
  // current generation
  // children have a combination of the coefficients of their parents
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it;
  multimap<Double_t, TMVA::GeneticGenes >::iterator it2;
  Int_t pos = 0;
  Int_t n = 0;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    if( n< (Int_t)(fGenePool->size()/2) ){
      fNewGenePool->insert( entry(0, it->second) );
      pos = (Int_t)fRandomGenerator->Integer( fGenePool->size()/2 );
      it2 = fGenePool->begin();
      for( Int_t i=0; i<pos; i++) it2++;
      fNewGenePool->insert( entry( 0, MakeSex( it->second, it2->second ) ) );
    } else continue;
    n++;
  }
  fGenePool->swap( (*fNewGenePool) );
  fNewGenePool->clear();
}

//_______________________________________________________________________
TMVA::GeneticGenes TMVA::GeneticPopulation::MakeSex( TMVA::GeneticGenes male, 
                                                   TMVA::GeneticGenes female )
{
  // this function takes two individuals and produces offspring by mixing (recombining) their
  // coefficients
  //
  vector< Double_t > child;
  vector< Double_t >::iterator itM;
  vector< Double_t >::iterator itF = female.GetFactors().begin();
  for( itM = male.GetFactors().begin(); itM < male.GetFactors().end(); itM++ ){
    if( fRandomGenerator->Integer( 2 ) == 0 ){
      child.push_back( (*itM) );
    }else{
      child.push_back( (*itF) );
    }
    itF++;
  }
  return TMVA::GeneticGenes( child );
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::MakeMutants( Double_t probability, Bool_t near, 
                                          Double_t spread, Bool_t mirror )
{
  // produces offspring which is are mutated versions of their parents
  // Parameters:
  //         double probability : gives the probability (in percent) of a mutation of a coefficient
  //         bool near : if true, the mutation will produce a new coefficient which is "near" the old one
  //                     (gaussian around the current value)
  //         double spread : if near==true, spread gives the sigma of the gaussian
  //         bool mirror : if the new value obtained would be outside of the given constraints
  //                    the value is mapped between the constraints again. This can be done either
  //                    by a kind of periodic boundary conditions or mirrored at the boundary.
  //                    (mirror = true seems more "natural")
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it;
  Int_t n = 0;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    if( n< (fPopulationSize/2) ){
      fNewGenePool->insert( entry(0, it->second) );
      fNewGenePool->insert( entry(1, it->second) );
    } else continue;
    n++;
  }
  fGenePool->swap( (*fNewGenePool) );
  Mutate( probability, fPopulationSize/2, near, spread, mirror );
  fNewGenePool->clear();
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::Mutate( Double_t probability , Int_t startIndex, 
                                     Bool_t near, Double_t spread, Bool_t mirror ) 
{
  // mutates the individuals in the genePool
  // Parameters:
  //         double probability : gives the probability (in percent) of a mutation of a coefficient
  //         int startIndex : leaves unchanged (without mutation) the individuals which are better ranked
  //                     than indicated by "startIndex". This means: if "startIndex==3", the first (and best)
  //                     three individuals are not mutaded. This allows to preserve the best result of the 
  //                     current Generation for the next generation. 
  //         bool near : if true, the mutation will produce a new coefficient which is "near" the old one
  //                     (gaussian around the current value)
  //         double spread : if near==true, spread gives the sigma of the gaussian
  //         bool mirror : if the new value obtained would be outside of the given constraints
  //                    the value is mapped between the constraints again. This can be done either
  //                    by a kind of periodic boundary conditions or mirrored at the boundary.
  //                    (mirror = true seems more "natural")
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it;
  Int_t index = 0;
  vector< Double_t >::iterator vec;
  vector< TMVA::GeneticRange* >::iterator vecRange;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    if( index >= startIndex ){
      vecRange = fRanges.begin();
      for( vec = (it->second.GetFactors()).begin(); vec < (it->second.GetFactors()).end(); vec++ ){
        if( fRandomGenerator->Uniform( 100 ) <= probability ){
          (*vec) = (*vecRange)->Random( near, (*vec), spread, mirror );
        }
        vecRange++;
      }
    }
    index++;
  }
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::AddFactor( Double_t from, Double_t to )
{
  // adds a new coefficient to the individuals. 
  // Parameters:
  //          double from : minimum value of the coefficient
  //          double to : maximum value of the coefficient
  //
  fRanges.push_back( new TMVA::GeneticRange( fRandomGenerator, from, to ) );
}

//_______________________________________________________________________
TMVA::GeneticGenes* TMVA::GeneticPopulation::GetGenes( Int_t index )
{
  // gives back the "Genes" of the population with the given index.
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it = fGenePool->begin();
  for( Int_t i=0; i<index; i++) it++;
  return &(it->second);
}

//_______________________________________________________________________
Double_t TMVA::GeneticPopulation::GetFitness( Int_t index )
{
  // gives back the calculated fitness of the individual with the given index
  // (after the evaluation of the fitness ["calculateFitness"] index==0 
  // is the best individual.
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it = fGenePool->begin();
  for( Int_t i=0; i<index; i++) it++;
  return it->first;
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::ClearResults()
{
  // delete the results of the last calculation of the fitnesses of the
  // population.
  // (to prepare a new Generation)
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it;
  for( it = fGenePool->begin(); it!=fGenePool->end(); it++ ){
    it->second.ClearResults();
  }
}

//_______________________________________________________________________
TMVA::GeneticGenes* TMVA::GeneticPopulation::GetGenes()
{
  // get the Genes of where an internal pointer is pointing to in the population
  //
  TMVA::GeneticGenes *g;
  if( fCounter == fGenePool->end() ) {
    g = new TMVA::GeneticGenes();
    return g;
  }
  g = &(fCounter->second);
  fCounterFitness = fCounter->first;
  return g;
}

//_______________________________________________________________________
Double_t TMVA::GeneticPopulation::GetFitness()
{
  // gives back the currently calculated fitness
  //
  if( fCounter == fGenePool->end() ) {
    Reset();
    return -1.;
  }
  return fCounter->first;
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::Reset()
{
  // prepare for a new generation
  //
  fCounter = fGenePool->begin();
  fNewGenePool->clear();
}

//_______________________________________________________________________
Bool_t TMVA::GeneticPopulation::SetFitness( TMVA::GeneticGenes *g, Double_t fitness, Bool_t add )
{
  // set the fitness of "g" to the value "fitness". 
  // add==true indicates, that this individual is created newly in this generation
  // if add==false, this is a reavaluation of the fitness of the individual.
  //
  if (add) g->GetResults().push_back( fitness );
  fNewGenePool->insert( entry( fitness, *g) );
  fCounter++;
  if( fCounter == fGenePool->end() ){
    fGenePool->swap( (*fNewGenePool) );
    fCounter = fGenePool->begin();
    Reset();
    return kFALSE;
  }
  return kTRUE;
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::GiveHint( vector< Double_t > hint, Double_t fitness )
{
  // if there is some good configuration of coefficients one might give this Hint to
  // the genetic algorithm. 
  // Parameters:
  //       vector< double > hint : is the collection of coefficients
  //       double fitness : is the fitness this collection has got
  //
  TMVA::GeneticGenes g;
  g.GetFactors().assign( hint.begin(), hint.end() );             

  fGenePool->insert( entry( fitness, g ) );
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::Print( Int_t untilIndex)
{
  // make a little printout of the individuals up to index "untilIndex"
  // this means, .. write out the best "untilIndex" individuals.
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it;
  Int_t n;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    if( untilIndex >= -1 ) {
      if( untilIndex == -1 ) return;
      untilIndex--;
    }
    n = 0;
    for( vector< Double_t >::iterator vec = it->second.GetFactors().begin(); 
       vec < it->second.GetFactors().end(); vec++ ) {
       cout << "f_" << n++ << ": " << (*vec) << "     ";
    }
    cout << endl;
  }
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::Print( ostream & out, Int_t untilIndex )
{
  // make a little printout to the stream "out" of the individuals up to index "untilIndex"
  // this means, .. write out the best "untilIndex" individuals.
  //
  multimap<Double_t, TMVA::GeneticGenes >::iterator it;
  Int_t n;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    if( untilIndex > -1 ) {
      untilIndex--;
      if( untilIndex == -1 ) return;
    }
    n = 0;
    out << "fitness: " << it->first << "    ";
    for( vector< Double_t >::iterator vec = it->second.GetFactors().begin(); 
       vec < it->second.GetFactors().end(); vec++ ){
       out << "f_" << n++ << ": " << (*vec) << "     ";
    }
    out << endl;
  }
}

//_______________________________________________________________________
TH1F* TMVA::GeneticPopulation::VariableDistribution( Int_t varNumber, Int_t bins, 
                                                    Int_t min, Int_t max ) 
{
  // give back a histogram with the distribution of the coefficients
  // parameters:
  //          int bins : number of bins of the histogram
  //          int min : histogram minimum 
  //          int max : maximum value of the histogram
  //
  std::stringstream histName;
  histName.clear();
  histName.str("v");
  histName << varNumber;
  TH1F *hist = new TH1F( histName.str().c_str(),histName.str().c_str(), bins,min,max );
  hist->SetBit(TH1::kCanRebin);

  multimap<Double_t, TMVA::GeneticGenes >::iterator it;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    hist->Fill( it->second.GetFactors().at(varNumber));
  }
  return hist;
}

//_______________________________________________________________________
vector<Double_t> TMVA::GeneticPopulation::VariableDistribution( Int_t varNumber ) 
{
  // gives back all the values of coefficient "varNumber" of the current generation
  //
  vector< Double_t > varDist;
  multimap<Double_t, TMVA::GeneticGenes >::iterator it;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    varDist.push_back( it->second.GetFactors().at( varNumber ) );
  }
  return varDist;
}

//_______________________________________________________________________
TMVA::GeneticPopulation::~GeneticPopulation()
{
  // destructor
  if( fRandomGenerator != NULL ) delete fRandomGenerator;
  if( fGenePool != NULL ) delete fGenePool;
  if( fNewGenePool != NULL ) delete fNewGenePool;

  std::vector<GeneticRange*>::iterator it = fRanges.begin();
  for(;it!=fRanges.end(); it++) {
    delete *it;
  }

}



