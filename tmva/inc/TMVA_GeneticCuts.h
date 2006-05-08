// @(#)root/tmva $Id: TMVA_GeneticCuts.h,v 1.3 2006/04/29 23:55:41 andreas.hoecker Exp $ 
// Author: Andreas Hoecker, Peter Speckmayer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GeneticCuts                                                      *
 *                                                                                *
 * Description:                                                                   *
 *      User class for genetics algorithm                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
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
 * File and Version Information:                                                  *
 * $Id: TMVA_GeneticCuts.h,v 1.3 2006/04/29 23:55:41 andreas.hoecker Exp $ 
 **********************************************************************************/

#ifndef ROOT_TMVA_GeneticCuts
#define ROOT_TMVA_GeneticCuts

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_GeneticCuts                                                     //
//                                                                      //
// User class for genetics algorithm                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_GeneticBase
#include "TMVA_GeneticBase.h"
#endif

class TMVA_GeneticCuts :  public TMVA_GeneticBase {

 public:

  TMVA_GeneticCuts( Int_t size, std::vector<LowHigh*> ranges );
  virtual ~TMVA_GeneticCuts() {}

  Double_t fitnessFunction( std::vector<Double_t> parameters );
		
  Double_t newFitness( Double_t oldF, Double_t newF ) { return oldF + newF; }
};

#endif
