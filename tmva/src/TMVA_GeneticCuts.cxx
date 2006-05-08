// @(#)root/tmva $Id: TMVA_GeneticCuts.cxx,v 1.2 2006/05/08 15:39:03 brun Exp $ 
// Author: Andreas Hoecker, Peter Speckmayer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GeneticCuts                                                      *
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

#include "TMVA_GeneticCuts.h"
#include "TMVA_MethodCuts.h"

//_______________________________________________________________________
//                                                                      
// User class for genetics algorithm                                    
//                                                                      
//_______________________________________________________________________


TMVA_GeneticCuts::TMVA_GeneticCuts( Int_t size, std::vector<LowHigh*> ranges ) 
  : TMVA_GeneticBase( size, ranges ) 
{}

//_______________________________________________________________________
Double_t TMVA_GeneticCuts::fitnessFunction( std::vector<Double_t> parameters )
{
  Int_t n = parameters.size();
  const Int_t kNMAX=1000;
  if (n >= kNMAX) {
     printf("n>1000 in TMVA_GeneticCuts::fitnessFunction, aborting\n");
     return 0;
  }
  Double_t p[kNMAX];
  for (Int_t i=0; i<n; i++) p[i] = parameters[i];
  
  return TMVA_MethodCuts::ThisCuts()->ComputeEstimator( p, n ); 
}
