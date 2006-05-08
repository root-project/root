// @(#)root/tmva $Id: TMVA_GeneticBase.h,v 1.1 2006/05/08 12:46:30 brun Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GeneticBase                                                      *
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
// TMVA_GeneticBase                                                     //
//                                                                      //
// Base definition for genetic algorithm                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <deque>
#include <map>
#include <string>

#include <stdio.h>
#include <iostream>

#ifndef ROOT_TMVA_GeneticPopulation
#include "TMVA_GeneticPopulation.h"
#endif

typedef std::pair<Double_t, Double_t> LowHigh;

class TMVA_GeneticBase {

 private:

  // converging?
  Int_t              fConvCounter;
  Double_t           fConvValue;

  // spread-control (stepsize)
  std::deque<Int_t>  fSuccessList;
  Double_t           fLastResult;

 public:

  TMVA_GeneticPopulation fPopulation;

  Double_t fSpread;  // regulates the spread of the value change at mutation (sigma)
  Bool_t fMirror;
  Bool_t fSexual;
  Bool_t fFirstTime;

  TMVA_GeneticBase() {}
  TMVA_GeneticBase( Int_t populationSize, std::vector<LowHigh*> ranges );
  virtual ~TMVA_GeneticBase() {}

  void Init();
  virtual Bool_t HasConverged( Int_t steps = 10, Double_t ratio = 0.1 );
  virtual Double_t SpreadControl( Int_t steps, Int_t ofSteps, Double_t factor );
  Double_t Calc();
  virtual Double_t FitnessFunction( std::vector< Double_t > factors );
  virtual Double_t NewFitness( Double_t oldValue, Double_t newValue );
  virtual Double_t CalculateFitness();
  Double_t DoRenewFitness();
  virtual Double_t RenewFitness( std::vector< Double_t > factors, 
				 std::vector< Double_t > results );
  virtual void Evolution();
  void Finalize();
  
  ClassDef(TMVA_GeneticBase,0) //Base definition for genetic algorithm
};



#endif
