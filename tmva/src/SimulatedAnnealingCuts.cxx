// @(#)root/tmva $Id: SimulatedAnnealingCuts.cxx,v 1.6 2006/11/20 15:35:28 brun Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SimulatedAnnealingCuts                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
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
// Implementation of Simulated Annealing fitter for cut optimisation
//_______________________________________________________________________

#include "Riostream.h"
#include "TMVA/SimulatedAnnealingCuts.h"
#include "TMVA/MethodCuts.h"

ClassImp(TMVA::SimulatedAnnealingCuts)

TMVA::SimulatedAnnealingCuts::SimulatedAnnealingCuts( std::vector<Interval*>& ranges )
   : SimulatedAnnealingBase( ranges )
{
   // constructor
}

TMVA::SimulatedAnnealingCuts::~SimulatedAnnealingCuts()
{
   // destructor
}

Double_t TMVA::SimulatedAnnealingCuts::MinimizeFunction( const std::vector<Double_t>& parameters )
{
   // minimize function interface for Simulated Annealing fitter for cut optimisation

   return TMVA::MethodCuts::ThisCuts()->ComputeEstimator( parameters ); 
}
