// @(#)root/tmva $Id: SimulatedAnnealingFitter.h,v 1.5 2007/05/31 14:17:49 andreas.hoecker Exp $ 
// Author: Andreas Hoecker

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SimulatedAnnealingFitter                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *       Fitter using Simulated Annealing                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_SimulatedAnnealingFitter
#define ROOT_TMVA_SimulatedAnnealingFitter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// SimulatedAnnealingFitter                                             //
//                                                                      //
// Fitter using a Simulated Annealing Algorithm                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_FitterBase
#include "TMVA/FitterBase.h"
#endif

namespace TMVA {

   class IFitterTarget;
   class Interval;
   
   class SimulatedAnnealingFitter : public FitterBase {
      
   public:
      
      SimulatedAnnealingFitter( IFitterTarget& target, const TString& name, 
                                const std::vector<TMVA::Interval*>& ranges, const TString& theOption );

      virtual ~SimulatedAnnealingFitter() {}

      void SetParameters( Int_t    fMaxCalls,              
                          Int_t    fNFunLoops,             
                          Int_t    fNEps,                  
                          Bool_t   fUseAdaptiveTemperature,
                          Double_t fTemperatureGradient,   
                          Double_t fInitialTemperature,    
                          Double_t fMinTemperature,        
                          Double_t fEps );                  

      Double_t Run( std::vector<Double_t>& pars );

   private:

      void DeclareOptions();

      Int_t              fMaxCalls;                // max number of FCN calls
      Int_t              fNFunLoops;               // number of FCN loops
      Int_t              fNEps;                    // test parameter
      Bool_t             fUseAdaptiveTemperature;  // compute temperature steps on the fly
      Double_t           fTemperatureGradient;     // starting value for temperature gradient
      Double_t           fInitialTemperature;      // initial temperature (depends on FCN)
      Double_t           fMinTemperature;          // minimum temperature before SA quit
      Double_t           fEps;                     // relative required FCN accuracy at minimum
      
      ClassDef(SimulatedAnnealingFitter,0) // Fitter using a Genetic Algorithm
   };

} // namespace TMVA

#endif


