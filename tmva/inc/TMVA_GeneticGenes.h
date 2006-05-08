// @(#)root/tmva $Id: TMVA_GeneticGenes.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GeneticGenes                                                     *
 *                                                                                *
 * Description:                                                                   *
 *      Genes definition for genetic algorithm                                    *
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
 * $Id: TMVA_GeneticGenes.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_GeneticGenes
#define ROOT_TMVA_GeneticGenes

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_GeneticGenes                                                    //
//                                                                      //
// Genes definition for genetic algorithm                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"
#include <vector>

class TMVA_GeneticGenes{

 public:
  
  TMVA_GeneticGenes() {}  
  TMVA_GeneticGenes( std::vector<Double_t> f );
  
  void clear() {
    factors.clear();
    results.clear();
  }
  
  void clearResults() {
    results.clear();
  }

  std::vector< Double_t > factors;
  std::vector< Double_t > results;
};


#endif
