// @(#)root/tmva $Id$    
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
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#include <vector>

namespace TMVA {
   
   class GeneticGenes {
      
   public:
      
      GeneticGenes():fFitness(0) {}  
      GeneticGenes( std::vector<Double_t> & f );
      virtual ~GeneticGenes() {}  
      
      std::vector<Double_t>& GetFactors() { return fFactors; }
      
      void SetFitness(Double_t fitness) { fFitness = fitness; }
      Double_t GetFitness() const { return fFitness; }
      
      friend Bool_t operator <(const GeneticGenes&, const GeneticGenes&);
      
   private:
      
      std::vector<Double_t> fFactors; // stores the factors (coefficients) of one individual
      Double_t fFitness;
      
      ClassDef(GeneticGenes,0) // Genes definition for genetic algorithm
   };

   Bool_t operator <(const GeneticGenes&, const GeneticGenes&);

} // namespace TMVA

#endif
