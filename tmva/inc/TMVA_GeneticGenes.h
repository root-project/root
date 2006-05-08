// @(#)root/tmva $Id: TMVA_GeneticGenes.h,v 1.1 2006/05/08 12:46:30 brun Exp $    
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
  virtual ~TMVA_GeneticGenes() {}  
  
  void Clear() {
    fFactors.clear();
    fResults.clear();
  }
  
  void ClearResults() {
    fResults.clear();
  }

  std::vector< Double_t > fFactors;
  std::vector< Double_t > fResults;
  
  ClassDef(TMVA_GeneticGenes,0) //Genes definition for genetic algorithm
};


#endif
