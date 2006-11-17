// @(#)root/tmva $Id: GeneticANN.h,v 1.4 2006/11/16 22:51:58 helgevoss Exp $ 
// Author: Andreas Hoecker, Matt Jachowski, Peter Speckmayer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : GeneticANN                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      User class for genetics algorithm                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
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

#ifndef ROOT_TMVA_GeneticANN
#define ROOT_TMVA_GeneticANN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GeneticANN                                                          //
//                                                                      //
// Cut optimisation interface class for genetic algorithm               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_GeneticBase
#include "TMVA/GeneticBase.h"
#endif
#ifndef ROOT_TMVA_MethodMLP
#include "TMVA/MethodMLP.h"
#endif

namespace TMVA {

   class GeneticANN : public GeneticBase {
      
   public:
      
      GeneticANN( Int_t size, std::vector<LowHigh_t*> ranges, TMVA::MethodMLP* methodMLP);

      virtual ~GeneticANN() {}

      Double_t FitnessFunction( const std::vector<Double_t> & parameters );

      Double_t NewFitness( Double_t oldF, Double_t newF ) { return oldF + newF; }

   private:

      TMVA::MethodMLP* fMethodMLP; // pointer to method

      ClassDef(GeneticANN,0) // ANN interface class for genetic algorithm
      ;
   };
   
} // namespace TMVA

#endif
