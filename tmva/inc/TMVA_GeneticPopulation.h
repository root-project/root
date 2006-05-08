// @(#)root/tmva $Id: TMVA_GeneticPopulation.h,v 1.5 2006/05/02 12:01:35 andreas.hoecker Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GeneticPopulation                                                *
 *                                                                                *
 * Description:                                                                   *
 *    Population definition for genetic algorithm                                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_GeneticPopulation.h,v 1.5 2006/05/02 12:01:35 andreas.hoecker Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_GeneticPopulation
#define ROOT_TMVA_GeneticPopulation

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_GeneticPopulation                                               //
//                                                                      //
// Population definition for genetic algorithm                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <map>

#ifndef ROOT_TMVA_GeneticGenes
#include "TMVA_GeneticGenes.h"
#endif
#ifndef ROOT_TMVA_GeneticRange
#include "TMVA_GeneticRange.h"
#endif

class TH1F;

class TMVA_GeneticPopulation {

 public:

  TMVA_GeneticPopulation();
  virtual ~TMVA_GeneticPopulation();

  TRandom *randomGenerator;
  typedef std::pair< Double_t, TMVA_GeneticGenes > entry;

  std::multimap<Double_t, TMVA_GeneticGenes  >* genePool;
  std::multimap<Double_t, TMVA_GeneticGenes  >* newGenePool;

  std::vector< TMVA_GeneticRange* > ranges;

  std::multimap<Double_t, TMVA_GeneticGenes >::iterator counter;
  Double_t counterFitness;

  Int_t populationSize;

  void createPopulation( Int_t size );
  void addPopulation( TMVA_GeneticPopulation *genePool );
  void trimPopulation();
  void giveHint( std::vector< Double_t > hint, Double_t fitness = 0 );
  void makeChildren();
  TMVA_GeneticGenes makeSex( TMVA_GeneticGenes male, TMVA_GeneticGenes female );

  void makeMutants( Double_t probability = 30, Bool_t near = kFALSE, 
		    Double_t spread = 0.1, Bool_t mirror = kFALSE  );
  void mutate( Double_t probability = 20, Int_t startIndex = 0, Bool_t near = kFALSE, 
	       Double_t spread = 0.1, Bool_t mirror = kFALSE  );

  void addFactor( Double_t from, Double_t to );

  TMVA_GeneticGenes* getGenes();
  TMVA_GeneticGenes* getGenes( Int_t index );

  void clearResults( );
  void reset();
  Bool_t setFitness( TMVA_GeneticGenes *g, Double_t fitness, Bool_t add = kTRUE );
  Double_t getFitness( Int_t index );
  Double_t getFitness( );
  void print( Int_t untilIndex = -1 );
  void print( ostream & out, Int_t utilIndex = -1 );
  TH1F* variableDistribution( Int_t varNumber, Int_t bins, Int_t min, Int_t max  );
  std::vector< Double_t > variableDistribution( Int_t varNumber );
};



#endif
