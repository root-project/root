// @(#)root/tmva $Id: TMVA_GeneticPopulation.cxx,v 1.3 2006/05/09 08:37:06 brun Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GeneticPopulation                                                *
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

#include "TMVA_GeneticPopulation.h"
#include "TMVA_GeneticGenes.h"
#include <iostream>
#include "TSystem.h"
#include "TRandom3.h"
#include "TH1.h"
#include <sstream>

using namespace std;

ClassImp(TMVA_GeneticPopulation)
   
//_______________________________________________________________________
//                                                                      
// Population definition for genetic algorithm                          
//                                                                      
//_______________________________________________________________________

TMVA_GeneticPopulation::TMVA_GeneticPopulation()
{
  fRandomGenerator = new TRandom3(0); //please check
  //gSystem->Sleep(2);
  //fRandomGenerator->SetSeed( (long)gSystem->Now() );
  // seed of random-generator to machine clock
  // --> if called twice within a second, the generated numbers will be the same.

  fRandomGenerator->Uniform(0.,1.);
  fGenePool    = new multimap<Double_t, TMVA_GeneticGenes>();
  fNewGenePool = new multimap<Double_t, TMVA_GeneticGenes>();

  fCounterFitness = 0;
}

void TMVA_GeneticPopulation::CreatePopulation( Int_t size )
{
  fPopulationSize = size;
  fGenePool->clear();
  fNewGenePool->clear();

  vector< TMVA_GeneticRange* >::iterator rIt;
  vector< Double_t > newEntry;

  for( Int_t i=0; i<fPopulationSize; i++ ){
    newEntry.clear();
    for( rIt = fRanges.begin(); rIt<fRanges.end(); rIt++ ){
      newEntry.push_back( (*rIt)->Random() );
    }
    entry e(0, TMVA_GeneticGenes( newEntry) );
    fGenePool->insert( e );
  }

  fCounter = fGenePool->begin();
}

void TMVA_GeneticPopulation::AddPopulation( TMVA_GeneticPopulation *strangers )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  for( it = strangers->fGenePool->begin(); it != strangers->fGenePool->end(); it++ ) {
    GiveHint( it->second.fFactors, it->first );
  }
}

void TMVA_GeneticPopulation::TrimPopulation( )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it = fGenePool->begin() ;
  for( Int_t i=0; i<fPopulationSize; i++ ) it++;
  fGenePool->erase( it, fGenePool->end()-- );
}

void TMVA_GeneticPopulation::MakeChildren() 
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  multimap<Double_t, TMVA_GeneticGenes >::iterator it2;
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

TMVA_GeneticGenes TMVA_GeneticPopulation::MakeSex( TMVA_GeneticGenes male, 
                                                   TMVA_GeneticGenes female )
{
  vector< Double_t > child;
  vector< Double_t >::iterator itM;
  vector< Double_t >::iterator itF = female.fFactors.begin();
  for( itM = male.fFactors.begin(); itM < male.fFactors.end(); itM++ ){
    if( fRandomGenerator->Integer( 2 ) == 0 ){
      child.push_back( (*itM) );
    }else{
      child.push_back( (*itF) );
    }
    itF++;
  }
  return TMVA_GeneticGenes( child );
}

void TMVA_GeneticPopulation::MakeMutants( Double_t probability, Bool_t near, 
                                          Double_t spread, Bool_t mirror )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
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

void TMVA_GeneticPopulation::Mutate( Double_t probability , Int_t startIndex, 
                                     Bool_t near, Double_t spread, Bool_t mirror ) 
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  Int_t index = 0;
  vector< Double_t >::iterator vec;
  vector< TMVA_GeneticRange* >::iterator vecRange;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    if( index >= startIndex ){
      vecRange = fRanges.begin();
      for( vec = (it->second.fFactors).begin(); vec < (it->second.fFactors).end(); vec++ ){
        if( fRandomGenerator->Uniform( 100 ) <= probability ){
          (*vec) = (*vecRange)->Random( near, (*vec), spread, mirror );
        }
        vecRange++;
      }
    }
    index++;
  }
}

void TMVA_GeneticPopulation::AddFactor( Double_t from, Double_t to )
{
  fRanges.push_back( new TMVA_GeneticRange( fRandomGenerator, from, to ) );
}

TMVA_GeneticGenes* TMVA_GeneticPopulation::GetGenes( Int_t index )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it = fGenePool->begin();
  for( Int_t i=0; i<index; i++) it++;
  return &(it->second);
}

Double_t TMVA_GeneticPopulation::GetFitness( Int_t index )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it = fGenePool->begin();
  for( Int_t i=0; i<index; i++) it++;
  return it->first;
}

void TMVA_GeneticPopulation::ClearResults()
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  for( it = fGenePool->begin(); it!=fGenePool->end(); it++ ){
    it->second.ClearResults();
  }
}

TMVA_GeneticGenes* TMVA_GeneticPopulation::GetGenes()
{
  TMVA_GeneticGenes *g;
  if( fCounter == fGenePool->end() ) {
    g = new TMVA_GeneticGenes();
    return g;
  }
  g = &(fCounter->second);
  fCounterFitness = fCounter->first;
  return g;
}

Double_t TMVA_GeneticPopulation::GetFitness()
{
  if( fCounter == fGenePool->end() ) {
    Reset();
    return -1.;
  }
  return fCounter->first;
}

void TMVA_GeneticPopulation::Reset()
{
  fCounter = fGenePool->begin();
  fNewGenePool->clear();
}

Bool_t TMVA_GeneticPopulation::SetFitness( TMVA_GeneticGenes *g, Double_t fitness, Bool_t add )
{
  if( add ) g->fResults.push_back( fitness );
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

void TMVA_GeneticPopulation::GiveHint( vector< Double_t > hint, Double_t fitness )
{
  TMVA_GeneticGenes g;
  g.fFactors.assign( hint.begin(), hint.end() );             

  fGenePool->insert( entry( fitness, g ) );
}

void TMVA_GeneticPopulation::Print( Int_t untilIndex)
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  Int_t n;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    if( untilIndex >= -1 ) {
      if( untilIndex == -1 ) return;
      untilIndex--;
    }
    n = 0;
    for( vector< Double_t >::iterator vec = it->second.fFactors.begin(); 
       vec < it->second.fFactors.end(); vec++ ) {
       cout << "f_" << n++ << ": " << (*vec) << "     ";
    }
    cout << endl;
  }
}

void TMVA_GeneticPopulation::Print( ostream & out, Int_t untilIndex )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  Int_t n;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    if( untilIndex > -1 ) {
      untilIndex--;
      if( untilIndex == -1 ) return;
    }
    n = 0;
    out << "fitness: " << it->first << "    ";
    for( vector< Double_t >::iterator vec = it->second.fFactors.begin(); 
       vec < it->second.fFactors.end(); vec++ ){
       out << "f_" << n++ << ": " << (*vec) << "     ";
    }
    out << endl;
  }
}

TH1F* TMVA_GeneticPopulation::VariableDistribution( Int_t varNumber, Int_t bins, 
                                                    Int_t min, Int_t max ) 
{
  std::stringstream histName;
  histName.clear();
  histName.str("v");
  histName << varNumber;
  TH1F *hist = new TH1F( histName.str().c_str(),histName.str().c_str(), bins,min,max );
  hist->SetBit(TH1::kCanRebin);

  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    hist->Fill( it->second.fFactors.at(varNumber));
  }
  return hist;
}

vector<Double_t> TMVA_GeneticPopulation::VariableDistribution( Int_t varNumber ) 
{
  vector< Double_t > varDist;
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  for( it = fGenePool->begin(); it != fGenePool->end(); it++ ){
    varDist.push_back( it->second.fFactors.at( varNumber ) );
  }
  return varDist;
}

TMVA_GeneticPopulation::~TMVA_GeneticPopulation()
{
  if( fRandomGenerator != NULL ) delete fRandomGenerator;
  if( fGenePool != NULL ) delete fGenePool;
  if( fNewGenePool != NULL ) delete fNewGenePool;
}



