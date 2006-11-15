// @(#)root/tmva $Id: GeneticGenes.h,v 1.8 2006/10/10 17:43:51 andreas.hoecker Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : GeneticGenes                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_GeneticGenes
#define ROOT_TMVA_GeneticGenes

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GeneticGenes                                                         //
//                                                                      //
// Genes definition for genetic algorithm                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"
#include <vector>

namespace TMVA {

   class GeneticGenes {

   public:
  
      GeneticGenes() {}  
      GeneticGenes( std::vector<Double_t> & f );
      virtual ~GeneticGenes() {}  
  
      void Clear();  
      void ClearResults();

      std::vector<Double_t>& GetFactors() { return fFactors; }
      std::vector<Double_t>& GetResults() { return fResults; }  

   private:

      std::vector<Double_t> fFactors; // stores the factors (coefficients) of one individual
      std::vector<Double_t> fResults; // stores the fitness-results of this individual
  
      ClassDef(GeneticGenes,0) // Genes definition for genetic algorithm
         ;
   };

} // namespace TMVA

#endif
