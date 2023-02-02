// @(#)root/tmva $Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : GeneticAlgorithm                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Base definition for genetic algorithm                                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_GeneticAlgorithm
#define ROOT_TMVA_GeneticAlgorithm

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GeneticAlgorithm                                                     //
//                                                                      //
// Base definition for genetic algorithm                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <deque>
#include <iosfwd>

#include "TMVA/IFitterTarget.h"
#include "TMVA/GeneticPopulation.h"
#include "TMVA/Types.h"

namespace TMVA {

   class IFitterTarget;
   class Interval;
   class MsgLogger;

   class GeneticAlgorithm {

   public:

      GeneticAlgorithm( IFitterTarget& target, Int_t populationSize,
                        const std::vector<TMVA::Interval*>& ranges, UInt_t seed = 0 );
      virtual ~GeneticAlgorithm();

      void Init();

      virtual Bool_t   HasConverged(Int_t steps = 10, Double_t ratio = 0.1);
      virtual Double_t SpreadControl(Int_t steps, Int_t ofSteps,
                                     Double_t factor);
      virtual Double_t NewFitness(Double_t oldValue, Double_t newValue);
      virtual Double_t CalculateFitness();
      virtual void Evolution();

      GeneticPopulation& GetGeneticPopulation() { return fPopulation; }

      Double_t GetSpread() const { return fSpread; }
      void     SetSpread(Double_t s) { fSpread = s; }

      void   SetMakeCopies(Bool_t s) { fMakeCopies = s; }
      Bool_t GetMakeCopies() { return fMakeCopies; }

      Int_t    fConvCounter;              // converging? ... keeps track of the number of improvements

   protected:

      IFitterTarget&    fFitterTarget;    // the fitter target

      Double_t fConvValue;                // keeps track of the quantity of improvement

      // spread-control (stepsize)
      // successList keeps track of the improvements to be able

      std::deque<Int_t> fSuccessList;     // to adjust the stepSize
      Double_t          fLastResult;      // remembers the last obtained result (for internal use)

      Double_t          fSpread;          // regulates the spread of the value change at mutation (sigma)
      Bool_t            fMirror;          // new values for mutation are mirror-mapped if outside of constraints
      Bool_t            fFirstTime;       // if true its the first time, so no evolution yet
      Bool_t            fMakeCopies;      // if true, the population will make copies of the first individuals
                                          // avoid for speed performance.
      Int_t             fPopulationSize;  // the size of the population

      const std::vector<TMVA::Interval*>& fRanges; // parameter ranges

      GeneticPopulation fPopulation;      // contains and controls the "individual"
      Double_t fBestFitness;

      mutable MsgLogger* fLogger;         // message logger
      MsgLogger& Log() const { return *fLogger; }

      ClassDef(GeneticAlgorithm, 0);  // Genetic algorithm controller
   };

} // namespace TMVA

#endif
