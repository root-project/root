// @(#)root/tmva $Id: SimulatedAnnealingCuts.h,v 1.5 2006/10/10 17:43:52 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SimulatedAnnealingCuts                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Concrete Simulated Annealing fitter for cut optimisation                  *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_SimulatedAnnealingCuts
#define ROOT_TMVA_SimulatedAnnealingCuts

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// SimulatedAnnealingCuts                                               //
//                                                                      //
// Concrete Simulated Annealing fitter for cut optimisation             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_SimulatedAnnealingBase
#include "TMVA/SimulatedAnnealingBase.h"
#endif

namespace TMVA {

   class SimulatedAnnealingCuts : public SimulatedAnnealingBase {
      
   public:
      
      SimulatedAnnealingCuts( std::vector<LowHigh_t*>& ranges );
      ~SimulatedAnnealingCuts();
      
      Double_t MinimizeFunction( const std::vector<Double_t>& parameters );                      

      ClassDef(SimulatedAnnealingCuts,0)  // Concrete Simulated Annealing fitter for cut optimisation
         ;
   };
} // namespace TMVA

#endif

