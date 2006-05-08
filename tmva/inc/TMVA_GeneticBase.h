// @(#)root/tmva $Id: TMVA_GeneticBase.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $    
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
 * File and Version Information:                                                  *
 * $Id: TMVA_GeneticBase.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $    
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
  Int_t convCounter;
  Double_t convValue;

  // spread-control (stepsize)
  std::deque<Int_t> successList;
  Double_t lastResult;

 public:

  TMVA_GeneticPopulation population;

  Double_t spread;  // regulates the spread of the value change at mutation (sigma)
  Bool_t mirror;
  Bool_t sexual;
  Bool_t firstTime;

  TMVA_GeneticBase() {}
  TMVA_GeneticBase( Int_t populationSize, std::vector<LowHigh*> ranges );
  virtual ~TMVA_GeneticBase() {}

  void init();
  virtual Bool_t hasConverged( Int_t steps = 10, Double_t ratio = 0.1 );
  virtual Double_t spreadControl( Int_t steps, Int_t ofSteps, Double_t factor );
  Double_t calc();
  virtual Double_t fitnessFunction( std::vector< Double_t > factors );
  virtual Double_t newFitness( Double_t oldValue, Double_t newValue );
  virtual Double_t calculateFitness();
  Double_t doRenewFitness();
  virtual Double_t renewFitness( std::vector< Double_t > factors, 
				 std::vector< Double_t > results );
  virtual void evolution();
  void finalize();
};



#endif
