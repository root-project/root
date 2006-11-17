// @(#)root/tmva $Id: GeneticCuts.cxx,v 1.17 2006/11/16 22:51:58 helgevoss Exp $ 
// Author: Andreas Hoecker, Matt Jachowski, Peter Speckmayer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::GeneticCuts                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// User-defined class for genetics algorithm;
// defines interface to the fitness function
//_______________________________________________________________________

#include "TMVA/GeneticCuts.h"
#include "TMVA/MethodCuts.h"

ClassImp(TMVA::GeneticCuts)
   ;

//_______________________________________________________________________
TMVA::GeneticCuts::GeneticCuts( Int_t size, std::vector<LowHigh_t*> ranges, 
                                TMVA::MethodCuts* methodCuts ) 
   : TMVA::GeneticBase( size, ranges ) 
{
   // constructor
   fMethodCuts = methodCuts;
}            

//_______________________________________________________________________
Double_t TMVA::GeneticCuts::FitnessFunction( const std::vector<Double_t>& parameters )
{
   // fitness function interface for Genetics Algorithm application of cut 
   // optimisation method
   if (fMethodCuts == NULL) 
      return TMVA::MethodCuts::ThisCuts()->ComputeEstimator( parameters );
   else
      return fMethodCuts->ComputeEstimator( parameters );
}
