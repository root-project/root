// @(#)root/tmva $Id: GeneticANN.cxx,v 1.5 2006/11/16 22:51:58 helgevoss Exp $ 
// Author: Andreas Hoecker, Matt Jachowski, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : GeneticANN                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
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

#include "TMVA/GeneticANN.h"
#include "TMVA/MethodMLP.h"

ClassImp(TMVA::GeneticANN)
   ;

//_______________________________________________________________________
TMVA::GeneticANN::GeneticANN( Int_t size, std::vector<LowHigh_t*> ranges, TMVA::MethodMLP* methodMLP ) 
   : TMVA::GeneticBase( size, ranges ) 
{
   // constructor
   fMethodMLP = methodMLP;
}

//_______________________________________________________________________
Double_t TMVA::GeneticANN::FitnessFunction( const std::vector<Double_t>& parameters )
{
   // fitness function interface for GA
   return fMethodMLP->ComputeEstimator( parameters );
}
