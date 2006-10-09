// @(#)root/tmva $Id: GeneticPopulation.h,v 1.8 2006/08/30 22:19:58 andreas.hoecker Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : GeneticPopulation                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_GeneticPopulation
#define ROOT_TMVA_GeneticPopulation

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GeneticPopulation                                                    //
//                                                                      //
// Population definition for genetic algorithm                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <map>

#ifndef ROOT_TMVA_GeneticGenes
#include "TMVA/GeneticGenes.h"
#endif
#ifndef ROOT_TMVA_GeneticRange
#include "TMVA/GeneticRange.h"
#endif

class TH1F;

namespace TMVA {

   class GeneticPopulation {

   public:

      GeneticPopulation();
      virtual ~GeneticPopulation();

      typedef std::pair<const Double_t, GeneticGenes > entry;

      void CreatePopulation( Int_t size );
      void AddPopulation( GeneticPopulation *genePool );
      void TrimPopulation();
      void GiveHint( std::vector< Double_t > hint, Double_t fitness = 0 );
      void MakeChildren();
      GeneticGenes MakeSex( GeneticGenes male, GeneticGenes female );

      void MakeMutants( Double_t probability = 30, Bool_t near = kFALSE, 
                        Double_t spread = 0.1, Bool_t mirror = kFALSE  );
      void Mutate( Double_t probability = 20, Int_t startIndex = 0, Bool_t near = kFALSE, 
                   Double_t spread = 0.1, Bool_t mirror = kFALSE  );

      void AddFactor( Double_t from, Double_t to );

      GeneticGenes* GetGenes();
      GeneticGenes* GetGenes( Int_t index );

      void     ClearResults( );
      void     Reset();
      Bool_t   SetFitness( GeneticGenes *g, Double_t fitness, Bool_t add = kTRUE );
      Double_t GetFitness( Int_t index );
      Double_t GetFitness( );

      void Print( Int_t untilIndex = -1 );
      void Print( ostream & out, Int_t utilIndex = -1 );

      TH1F* VariableDistribution( Int_t varNumber, Int_t bins, Int_t min, Int_t max  );
      std::vector< Double_t > VariableDistribution( Int_t varNumber );

      Double_t GetCounterFitness() const { return fCounterFitness; }
      Int_t    GetPopulationSize() const { return fPopulationSize; }

      std::multimap<Double_t, GeneticGenes  >* GetGenePool()    const { return fGenePool; }
      std::multimap<Double_t, GeneticGenes  >* GetNewGenePool() const { return fNewGenePool; }
      std::vector< TMVA::GeneticRange* >&      GetRanges()      { return fRanges; }
  
   private:

      // internal use
      Double_t fCounterFitness;
      // the size of the population
      Int_t    fPopulationSize;

      // the "genePool" where the individuals of the current generation are stored
      std::multimap<Double_t, GeneticGenes  >* fGenePool;
      // the genePool where the offspring individuals are stored
      std::multimap<Double_t, GeneticGenes  >* fNewGenePool;
      // contains the ranges inbetween the values of the coefficients have to be
      std::vector< GeneticRange* > fRanges;
      // an internal counter
      std::multimap<Double_t, GeneticGenes >::iterator fCounter;

      // random Generator for this population
      TRandom *fRandomGenerator;  

      ClassDef(GeneticPopulation,0) //Population definition for genetic algorithm
         };

} // namespace TMVA

#endif
