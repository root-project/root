// @(#)root/tmva $Id$    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::GeneticPopulation                                               *
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

#include <iostream>
#include <iomanip>

#include "Rstrstream.h"
#include "TSystem.h"
#include "TRandom3.h"
#include "TH1.h"
#include <algorithm>

#include "TMVA/GeneticPopulation.h"
#include "TMVA/GeneticGenes.h"
#include "TMVA/MsgLogger.h"

ClassImp(TMVA::GeneticPopulation)

using namespace std;
   
//_______________________________________________________________________
//                                                                      
// Population definition for genetic algorithm                          
//_______________________________________________________________________

//_______________________________________________________________________
TMVA::GeneticPopulation::GeneticPopulation(const std::vector<Interval*>& ranges, Int_t size, UInt_t seed) 
   : fGenePool(size),
     fRanges(ranges.size()),
     fLogger( new MsgLogger("GeneticPopulation") )
{
   // Constructor
   
   // create a randomGenerator for this population and set a seed
   // create the genePools
   //
   fRandomGenerator = new TRandom3( 100 ); //please check
   fRandomGenerator->Uniform(0.,1.);
   fRandomGenerator->SetSeed( seed );

   for ( unsigned int i = 0; i < ranges.size(); ++i )
      fRanges[i] = new TMVA::GeneticRange( fRandomGenerator, ranges[i] );

   vector<Double_t> newEntry( fRanges.size() );
   for ( int i = 0; i < size; ++i )
      {
         for ( unsigned int rIt = 0; rIt < fRanges.size(); ++rIt )
            newEntry[rIt] = fRanges[rIt]->Random();
         fGenePool[i] = TMVA::GeneticGenes( newEntry);
      }

   fPopulationSizeLimit = size;
}

//_______________________________________________________________________
TMVA::GeneticPopulation::~GeneticPopulation() 
{
   // destructor
   if (fRandomGenerator != NULL) delete fRandomGenerator;

   std::vector<GeneticRange*>::iterator it = fRanges.begin();
   for (;it!=fRanges.end(); it++) delete *it;

   delete fLogger;
}



//_______________________________________________________________________
void TMVA::GeneticPopulation::SetRandomSeed( UInt_t seed ) 
{
   // the random seed of the random generator
   fRandomGenerator->SetSeed( seed );
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::MakeCopies( int number )
{
   // produces offspring which is are copies of their parents
   // Parameters:
   //         int number : the number of the last individual to be copied
   //
   
   int i=0; 
   for (std::vector<TMVA::GeneticGenes>::iterator it = fGenePool.begin();
        it != fGenePool.end() && i < number; 
        ++it, ++i ) {
      GiveHint( it->GetFactors(), it->GetFitness() );
   }
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::MakeChildren()
{
   // does what the name says,... it creates children out of members of the
   // current generation
   // children have a combination of the coefficients of their parents
   //

#ifdef _GLIBCXX_PARALLEL
#pragma omp parallel
#pragma omp for
#endif
   for ( int it = 0; it < (int) (fGenePool.size() / 2); ++it )
      {
         Int_t pos = (Int_t)fRandomGenerator->Integer( fGenePool.size()/2 );
         fGenePool[(fGenePool.size() / 2) + it] = MakeSex( fGenePool[it], fGenePool[pos] );
      }
}

//_______________________________________________________________________
TMVA::GeneticGenes TMVA::GeneticPopulation::MakeSex( TMVA::GeneticGenes male, 
                                                     TMVA::GeneticGenes female ) 
{
   // this function takes two individuals and produces offspring by mixing (recombining) their
   // coefficients
   //
   vector< Double_t > child(fRanges.size());
   for (unsigned int i = 0; i < fRanges.size(); ++i) {
      if (fRandomGenerator->Integer( 2 ) == 0) {
         child[i] = male.GetFactors()[i];
      }else{
         child[i] = female.GetFactors()[i];
      }
   }
   return TMVA::GeneticGenes( child );
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
   //         Bool_t near : if true, the mutation will produce a new coefficient which is "near" the old one
   //                     (gaussian around the current value)
   //         double spread : if near==true, spread gives the sigma of the gaussian
   //         Bool_t mirror : if the new value obtained would be outside of the given constraints
   //                    the value is mapped between the constraints again. This can be done either
   //                    by a kind of periodic boundary conditions or mirrored at the boundary.
   //                    (mirror = true seems more "natural")
   //

   vector< Double_t>::iterator vec;
   vector< TMVA::GeneticRange* >::iterator vecRange;

   //#ifdef _GLIBCXX_PARALLEL
   // #pragma omp parallel
   // #pragma omp for
   //#endif
   // The range methods are not thread safe!
   for (int it = startIndex; it < (int) fGenePool.size(); ++it) {
      vecRange = fRanges.begin();
      for (vec = (fGenePool[it].GetFactors()).begin(); vec < (fGenePool[it].GetFactors()).end(); ++vec) {
         if (fRandomGenerator->Uniform( 100 ) <= probability) {
            (*vec) = (*vecRange)->Random( near, (*vec), spread, mirror );
         }
         ++vecRange;
      }
   }
}


//_______________________________________________________________________
TMVA::GeneticGenes* TMVA::GeneticPopulation::GetGenes( Int_t index )
{
   // gives back the "Genes" of the population with the given index.
   //
   return &(fGenePool[index]);
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::Print( Int_t untilIndex )
{
   // make a little printout of the individuals up to index "untilIndex"
   // this means, .. write out the best "untilIndex" individuals.
   //

   for ( unsigned int it = 0; it < fGenePool.size(); ++it )
      {
         Int_t n=0;
         if (untilIndex >= -1 ) {
            if (untilIndex == -1 ) return;
            untilIndex--;
         }
         Log() << "fitness: " << fGenePool[it].GetFitness() << "    ";
         for (vector< Double_t >::iterator vec = fGenePool[it].GetFactors().begin(); 
              vec < fGenePool[it].GetFactors().end(); vec++ ) {
            Log() << "f_" << n++ << ": " << (*vec) << "     ";
         }
         Log() << Endl;
      }
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::Print( ostream & out, Int_t untilIndex )
{
   // make a little printout to the stream "out" of the individuals up to index "untilIndex"
   // this means, .. write out the best "untilIndex" individuals.
   //

   for ( unsigned int it = 0; it < fGenePool.size(); ++it ) {
      Int_t n=0;
      if (untilIndex >= -1 ) {
         if (untilIndex == -1 ) return;
         untilIndex--;
      }
      out << "fitness: " << fGenePool[it].GetFitness() << "    ";
      for (vector< Double_t >::iterator vec = fGenePool[it].GetFactors().begin(); 
           vec < fGenePool[it].GetFactors().end(); vec++ ) {
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

   cout << "FAILED! TMVA::GeneticPopulation::VariableDistribution" << endl;

   std::stringstream histName;
   histName.clear();
   histName.str("v");
   histName << varNumber;
   TH1F *hist = new TH1F( histName.str().c_str(),histName.str().c_str(), bins,min,max );

   return hist;
}

//_______________________________________________________________________
vector<Double_t> TMVA::GeneticPopulation::VariableDistribution( Int_t /*varNumber*/ ) 
{
   // gives back all the values of coefficient "varNumber" of the current generation
   //

   cout << "FAILED! TMVA::GeneticPopulation::VariableDistribution" << endl;

   vector< Double_t > varDist;

   return varDist;
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::AddPopulation( GeneticPopulation *strangers )
{
   // add another population (strangers) to the one of this GeneticPopulation
   for (std::vector<TMVA::GeneticGenes>::iterator it = strangers->fGenePool.begin(); 
        it != strangers->fGenePool.end(); it++ ) {
      GiveHint( it->GetFactors(), it->GetFitness() );
   }
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::AddPopulation( GeneticPopulation &strangers )
{
   // add another population (strangers) to the one of this GeneticPopulation
   AddPopulation(&strangers);
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::TrimPopulation()
{
   // trim the population to the predefined size
   std::sort(fGenePool.begin(), fGenePool.end());
   while ( fGenePool.size() > (unsigned int) fPopulationSizeLimit )
      fGenePool.pop_back();
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::GiveHint( std::vector< Double_t >& hint, Double_t fitness )
{
   // add an individual (a set of variables) to the population
   // if there is a set of variables which is known to perform good, they can be given as a hint to the population
   TMVA::GeneticGenes g(hint);
   g.SetFitness(fitness);

   fGenePool.push_back( g );
}

//_______________________________________________________________________
void TMVA::GeneticPopulation::Sort()
{
   // sort the genepool according to the fitness of the individuals
   std::sort(fGenePool.begin(), fGenePool.end());
}

