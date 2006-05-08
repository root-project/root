// @(#)root/tmva $Id: TMVA_GeneticGenes.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GeneticGenes                                                     *
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
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

#include "TMVA_GeneticGenes.h"
#include "TMVA_GeneticPopulation.h"
#include <iostream>

//_______________________________________________________________________
//                                                                      
// Genes definition for genetic algorithm                               
//                                                                      
//_______________________________________________________________________

ClassImp(TMVA_GeneticGenes)

TMVA_GeneticGenes::TMVA_GeneticGenes( std::vector<Double_t> f  ) 
{
  fFactors = f;
}
