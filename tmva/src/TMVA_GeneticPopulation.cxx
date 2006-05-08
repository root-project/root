// @(#)root/tmva $Id: TMVA_GeneticPopulation.cpp,v 1.4 2006/05/02 12:01:35 andreas.hoecker Exp $    
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
 * File and Version Information:                                                  *
 * $Id: TMVA_GeneticPopulation.cpp,v 1.4 2006/05/02 12:01:35 andreas.hoecker Exp $
 **********************************************************************************/

#include "TMVA_GeneticPopulation.h"
#include "TMVA_GeneticGenes.h"
#include <iostream>
#include "TSystem.h"
#include "TRandom3.h"
#include "TH1.h"
#include <sstream>

using namespace std;

//_______________________________________________________________________
//                                                                      
// Population definition for genetic algorithm                          
//                                                                      
//_______________________________________________________________________

TMVA_GeneticPopulation::TMVA_GeneticPopulation()
{
  randomGenerator = new TRandom3();
  gSystem->Sleep(2);
  randomGenerator->SetSeed( (long)gSystem->Now() );
  // seed of random-generator to machine clock
  // --> if called twice within a second, the generated numbers will be the same.

  randomGenerator->Uniform(0.,1.);
  genePool    = new multimap<Double_t, TMVA_GeneticGenes>();
  newGenePool = new multimap<Double_t, TMVA_GeneticGenes>();

  counterFitness = 0;
}

void TMVA_GeneticPopulation::createPopulation( Int_t size )
{
  populationSize = size;
  genePool->clear();
  newGenePool->clear();

  vector< TMVA_GeneticRange* >::iterator rIt;
  vector< Double_t > newEntry;

  for( Int_t i=0; i<populationSize; i++ ){
    newEntry.clear();
    for( rIt = ranges.begin(); rIt<ranges.end(); rIt++ ){
      newEntry.push_back( (*rIt)->random() );
    }
    entry e(0, TMVA_GeneticGenes( newEntry) );
    genePool->insert( e );
  }

  counter = genePool->begin();
}

void TMVA_GeneticPopulation::addPopulation( TMVA_GeneticPopulation *strangers )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  for( it = strangers->genePool->begin(); it != strangers->genePool->end(); it++ ) {
    giveHint( it->second.factors, it->first );
  }
}

void TMVA_GeneticPopulation::trimPopulation( )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it = genePool->begin() ;
  for( Int_t i=0; i<populationSize; i++ ) it++;
  genePool->erase( it, genePool->end()-- );
}

void TMVA_GeneticPopulation::makeChildren() 
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  multimap<Double_t, TMVA_GeneticGenes >::iterator it2;
  Int_t pos = 0;
  Int_t n = 0;
  for( it = genePool->begin(); it != genePool->end(); it++ ){
    if( n< (Int_t)(genePool->size()/2) ){
      newGenePool->insert( entry(0, it->second) );
      pos = (Int_t)randomGenerator->Integer( genePool->size()/2 );
      it2 = genePool->begin();
      for( Int_t i=0; i<pos; i++) it2++;
      newGenePool->insert( entry( 0, makeSex( it->second, it2->second ) ) );
    } else continue;
    n++;
  }
  genePool->swap( (*newGenePool) );
  newGenePool->clear();
}

TMVA_GeneticGenes TMVA_GeneticPopulation::makeSex( TMVA_GeneticGenes male, 
						   TMVA_GeneticGenes female )
{
  vector< Double_t > child;
  vector< Double_t >::iterator itM;
  vector< Double_t >::iterator itF = female.factors.begin();
  for( itM = male.factors.begin(); itM < male.factors.end(); itM++ ){
    if( randomGenerator->Integer( 2 ) == 0 ){
      child.push_back( (*itM) );
    }else{
      child.push_back( (*itF) );
    }
    itF++;
  }
  return TMVA_GeneticGenes( child );
}

void TMVA_GeneticPopulation::makeMutants( Double_t probability, Bool_t near, 
					  Double_t spread, Bool_t mirror )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  Int_t n = 0;
  for( it = genePool->begin(); it != genePool->end(); it++ ){
    if( n< (populationSize/2) ){
      newGenePool->insert( entry(0, it->second) );
      newGenePool->insert( entry(1, it->second) );
    } else continue;
    n++;
  }
  genePool->swap( (*newGenePool) );
  mutate( probability, populationSize/2, near, spread, mirror );
  newGenePool->clear();
}

void TMVA_GeneticPopulation::mutate( Double_t probability , Int_t startIndex, 
				     Bool_t near, Double_t spread, Bool_t mirror ) 
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  Int_t index = 0;
  vector< Double_t >::iterator vec;
  vector< TMVA_GeneticRange* >::iterator vecRange;
  for( it = genePool->begin(); it != genePool->end(); it++ ){
    if( index >= startIndex ){
      vecRange = ranges.begin();
      for( vec = (it->second.factors).begin(); vec < (it->second.factors).end(); vec++ ){
	if( randomGenerator->Uniform( 100 ) <= probability ){
	  (*vec) = (*vecRange)->random( near, (*vec), spread, mirror );
	}
	vecRange++;
      }
    }
    index++;
  }
}

void TMVA_GeneticPopulation::addFactor( Double_t from, Double_t to )
{
  ranges.push_back( new TMVA_GeneticRange( randomGenerator, from, to ) );
}

TMVA_GeneticGenes* TMVA_GeneticPopulation::getGenes( Int_t index )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it = genePool->begin();
  for( Int_t i=0; i<index; i++) it++;
  return &(it->second);
}

Double_t TMVA_GeneticPopulation::getFitness( Int_t index )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it = genePool->begin();
  for( Int_t i=0; i<index; i++) it++;
  return it->first;
}

void TMVA_GeneticPopulation::clearResults()
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  for( it = genePool->begin(); it!=genePool->end(); it++ ){
    it->second.clearResults();
  }
}

TMVA_GeneticGenes* TMVA_GeneticPopulation::getGenes()
{
  TMVA_GeneticGenes *g;
  if( counter == genePool->end() ) {
    g = new TMVA_GeneticGenes();
    return g;
  }
  g = &(counter->second);
  counterFitness = counter->first;
  return g;
}

Double_t TMVA_GeneticPopulation::getFitness()
{
  if( counter == genePool->end() ) {
    reset();
    return -1.;
  }
  return counter->first;
}

void TMVA_GeneticPopulation::reset()
{
  counter = genePool->begin();
  newGenePool->clear();
}

Bool_t TMVA_GeneticPopulation::setFitness( TMVA_GeneticGenes *g, Double_t fitness, Bool_t add )
{
  if( add ) g->results.push_back( fitness );
  newGenePool->insert( entry( fitness, *g) );
  counter++;
  if( counter == genePool->end() ){
    genePool->swap( (*newGenePool) );
    counter = genePool->begin();
    reset();
    return kFALSE;
  }
  return kTRUE;
}

void TMVA_GeneticPopulation::giveHint( vector< Double_t > hint, Double_t fitness )
{
  TMVA_GeneticGenes g;
  g.factors.assign( hint.begin(), hint.end() );             

  genePool->insert( entry( fitness, g ) );
}

void TMVA_GeneticPopulation::print( Int_t untilIndex)
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  Int_t n;
  for( it = genePool->begin(); it != genePool->end(); it++ ){
    if( untilIndex >= -1 ) {
      if( untilIndex == -1 ) return;
      untilIndex--;
    }
    n = 0;
    for( vector< Double_t >::iterator vec = it->second.factors.begin(); 
	 vec < it->second.factors.end(); vec++ ) {
      cout << "f_" << n++ << ": " << (*vec) << "     ";
    }
    cout << endl;
  }
}

void TMVA_GeneticPopulation::print( ostream & out, Int_t untilIndex )
{
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  Int_t n;
  for( it = genePool->begin(); it != genePool->end(); it++ ){
    if( untilIndex > -1 ) {
      untilIndex--;
      if( untilIndex == -1 ) return;
    }
    n = 0;
    out << "fitness: " << it->first << "    ";
    for( vector< Double_t >::iterator vec = it->second.factors.begin(); 
	 vec < it->second.factors.end(); vec++ ){
      out << "f_" << n++ << ": " << (*vec) << "     ";
    }
    out << endl;
  }
}

TH1F* TMVA_GeneticPopulation::variableDistribution( Int_t varNumber, Int_t bins, 
						    Int_t min, Int_t max ) 
{
  std::stringstream histName;
  histName.clear();
  histName.str("v");
  histName << varNumber;
  TH1F *hist = new TH1F( histName.str().c_str(),histName.str().c_str(), bins,min,max );
  hist->SetBit(TH1::kCanRebin);

  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  for( it = genePool->begin(); it != genePool->end(); it++ ){
    hist->Fill( it->second.factors.at(varNumber));
  }
  return hist;
}

vector<Double_t> TMVA_GeneticPopulation::variableDistribution( Int_t varNumber ) 
{
  vector< Double_t > varDist;
  multimap<Double_t, TMVA_GeneticGenes >::iterator it;
  for( it = genePool->begin(); it != genePool->end(); it++ ){
    varDist.push_back( it->second.factors.at( varNumber ) );
  }
  return varDist;
}

TMVA_GeneticPopulation::~TMVA_GeneticPopulation()
{
  if( randomGenerator != NULL ) delete randomGenerator;
  if( genePool != NULL ) delete genePool;
  if( newGenePool != NULL ) delete newGenePool;
}



