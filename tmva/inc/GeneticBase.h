// @(#)root/tmva $Id: GeneticBase.h,v 1.2 2006/05/23 13:03:15 brun Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : GeneticBase                                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Base definition for genetic algorithm                                     *
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

#ifndef ROOT_TMVA_GeneticBase
#define ROOT_TMVA_GeneticBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GeneticBase                                                          //
//                                                                      //
// Base definition for genetic algorithm                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <deque>
#include <map>
#include <string>

#include <stdio.h>
#include "Riostream.h"

#ifndef ROOT_TMVA_GeneticPopulation
#include "TMVA/GeneticPopulation.h"
#endif

namespace TMVA {
   
   typedef std::pair<Double_t,Double_t>   LowHigh_t;
  
   class GeneticBase {

   public:
    
      GeneticBase() {}
      GeneticBase(Int_t populationSize, std::vector < TMVA::LowHigh_t * > ranges);
      virtual ~GeneticBase() {}
      void Init();
      virtual Bool_t HasConverged(Int_t steps = 10, Double_t ratio = 0.1);
      virtual Double_t SpreadControl(Int_t steps, Int_t ofSteps,
                                     Double_t factor);
      Double_t Calc();
      virtual Double_t FitnessFunction(const std::vector < Double_t > & factors);
      virtual Double_t NewFitness(Double_t oldValue, Double_t newValue);
      virtual Double_t CalculateFitness();
      Double_t DoRenewFitness();
      virtual Double_t RenewFitness(std::vector < Double_t > factors,
                                    std::vector < Double_t > results);
      virtual void Evolution();
      void Finalize();
      
      GeneticPopulation& GetGeneticPopulation() { return fPopulation; } 
      Double_t GetSpread() const { return fSpread; }

      void SetSpread(Double_t s) { fSpread = s; }

   protected:
    
      // contains and controls the "individual"
      GeneticPopulation fPopulation;

      // converging? ... keeps track of the number of improvements
      Int_t fConvCounter;
      // keeps track of the quantity of improvement
      Double_t fConvValue;

      // spread-control (stepsize)
      // successList keeps track of the improvements to be able
      // to adjust the stepSize
      std::deque < Int_t > fSuccessList;
      // remembers the last obtained result (for internal use)
      Double_t fLastResult;

      Double_t fSpread;         // regulates the spread of the value change at mutation (sigma)
      Bool_t fMirror;           // new values for mutation are mirror-mapped if outside of constraints
      Bool_t fSexual;           // allow sexual recombination of individual
      Bool_t fFirstTime;        // if true its the first time, so no evolution yet

      ClassDef(GeneticBase, 0)  // Base definition for genetic algorithm
         };

} // namespace TMVA

#endif
