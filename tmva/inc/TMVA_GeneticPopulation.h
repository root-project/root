// @(#)root/tmva $Id: TMVA_GeneticPopulation.h,v 1.4 2006/05/09 08:37:06 brun Exp $    
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

  TRandom *fRandomGenerator;
  typedef std::pair<const Double_t, TMVA_GeneticGenes > entry;

  std::multimap<Double_t, TMVA_GeneticGenes  >* fGenePool;
  std::multimap<Double_t, TMVA_GeneticGenes  >* fNewGenePool;

  std::vector< TMVA_GeneticRange* > fRanges;

  std::multimap<Double_t, TMVA_GeneticGenes >::iterator fCounter;
  Double_t fCounterFitness;

  Int_t fPopulationSize;

  void CreatePopulation( Int_t size );
  void AddPopulation( TMVA_GeneticPopulation *genePool );
  void TrimPopulation();
  void GiveHint( std::vector< Double_t > hint, Double_t fitness = 0 );
  void MakeChildren();
  TMVA_GeneticGenes MakeSex( TMVA_GeneticGenes male, TMVA_GeneticGenes female );

  void MakeMutants( Double_t probability = 30, Bool_t near = kFALSE, 
                    Double_t spread = 0.1, Bool_t mirror = kFALSE  );
  void Mutate( Double_t probability = 20, Int_t startIndex = 0, Bool_t near = kFALSE, 
               Double_t spread = 0.1, Bool_t mirror = kFALSE  );

  void AddFactor( Double_t from, Double_t to );

  TMVA_GeneticGenes* GetGenes();
  TMVA_GeneticGenes* GetGenes( Int_t index );

  void ClearResults( );
  void Reset();
  Bool_t SetFitness( TMVA_GeneticGenes *g, Double_t fitness, Bool_t add = kTRUE );
  Double_t GetFitness( Int_t index );
  Double_t GetFitness( );
  void Print( Int_t untilIndex = -1 );
  void Print( ostream & out, Int_t utilIndex = -1 );
  TH1F* VariableDistribution( Int_t varNumber, Int_t bins, Int_t min, Int_t max  );
  std::vector< Double_t > VariableDistribution( Int_t varNumber );
  
  ClassDef(TMVA_GeneticPopulation,0) //Population definition for genetic algorithm
};



#endif
