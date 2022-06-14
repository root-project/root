// @(#)root/tmva $Id$
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
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/GeneticGenes.h"
#include "TMVA/GeneticPopulation.h"

#include "Rtypes.h"
#include "RtypesCore.h"

/*! \class TMVA::GeneticGenes
\ingroup TMVA

Cut optimisation interface class for genetic algorithm.

*/

ClassImp(TMVA::GeneticGenes);

////////////////////////////////////////////////////////////////////////////////
/// Constructor:
/// set the factors of this individual

TMVA::GeneticGenes::GeneticGenes( std::vector<Double_t> & f  )
{
   fFactors = f;
   fFitness = 0;
}

Bool_t TMVA::operator <(const TMVA::GeneticGenes& first, const TMVA::GeneticGenes& second)
{
   return first.fFitness < second.fFitness;
}
