// @(#)root/tmva $Id: GeneticCuts.cxx,v 1.5 2006/05/22 01:34:14 stelzer Exp $ 
// Author: Andreas Hoecker, Peter Speckmayer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::GeneticCuts                                                     *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
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
//_______________________________________________________________________
//                                                                      
// User-defined class for genetics algorithm;
// defines interface to the fitness function
//_______________________________________________________________________

#include "TMVA/GeneticCuts.h"
#include "TMVA/MethodCuts.h"

ClassImp(TMVA::GeneticCuts)

//_______________________________________________________________________
TMVA::GeneticCuts::GeneticCuts( Int_t size, std::vector<LowHigh*> ranges ) 
  : TMVA::GeneticBase( size, ranges ) 
{
  // constructor
}

// //_______________________________________________________________________
// Double_t TMVA::GeneticCuts::FitnessFunction( const std::vector<Double_t> parameters )
// {
//   Int_t n = parameters.size();
//   const Int_t kNMAX=1000;
//   if (n >= kNMAX) {
//      printf("n>1000 in TMVA::GeneticCuts::fitnessFunction, aborting\n");
//      return 0;
//   }
//   Double_t p[kNMAX];
//   for (Int_t i=0; i<n; i++) p[i] = parameters[i];
  
//   return TMVA::MethodCuts::ThisCuts()->ComputeEstimator( p, n ); 
// }

//_______________________________________________________________________
Double_t TMVA::GeneticCuts::FitnessFunction( const std::vector<Double_t> & parameters )
{
  // fitness function interface for Genetics Algorithm application of cut 
  // optimisation method
  return TMVA::MethodCuts::ThisCuts()->ComputeEstimator( parameters ); 
}
