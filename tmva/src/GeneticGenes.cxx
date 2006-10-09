// @(#)root/tmva $Id: GeneticGenes.cxx,v 1.11 2006/08/30 22:19:58 andreas.hoecker Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::GeneticGenes                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-KP Heidelberg, Germany,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/GeneticGenes.h"
#include "TMVA/GeneticPopulation.h"
#include "Riostream.h"

//_______________________________________________________________________
//                                                                      
// Cut optimisation interface class for genetic algorithm               //
//_______________________________________________________________________

ClassImp(TMVA::GeneticGenes)
   
//_______________________________________________________________________
TMVA::GeneticGenes::GeneticGenes( std::vector<Double_t> & f  ) 
{
   // Constructor:
   // set the factors of this individual
   fFactors = f;
}

//_______________________________________________________________________
void TMVA::GeneticGenes::Clear() 
{
   // clear the factors (coefficients) of this individual
   // clear the fitness-results obtained by this individual
   fFactors.clear();
   fResults.clear();
}
  
//_______________________________________________________________________
void TMVA::GeneticGenes::ClearResults() 
{
   // clear the fitness-results obtained by this individual
   fResults.clear();
}
