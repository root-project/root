// @(#)root/tmva $\Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SimulatedAnnealing                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Base implementation of simulated annealing fitting procedure              *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_SimulatedAnnealing
#define ROOT_TMVA_SimulatedAnnealing

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// SimulatedAnnealing                                                   //
//                                                                      //
// Base implementation of simulated annealing fitting procedure         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

class TRandom;

namespace TMVA {

   class IFitterTarget;
   class Interval;

   class SimulatedAnnealing {

   public:

      SimulatedAnnealing( IFitterTarget& target, const std::vector<Interval*>& ranges );
      virtual ~SimulatedAnnealing();

      // returns FCN value at minimum
      Double_t Minimize( std::vector<Double_t>& parameters );

      // accessors
      void SetMaxCalls    ( Int_t    mc    ) { fMaxCalls = mc; }
      void SetTempGrad    ( Double_t dt    ) { fTemperatureGradient = dt; }
      void SetUseAdaptTemp( Bool_t   yesno ) { fUseAdaptiveTemperature = yesno; }
      void SetInitTemp    ( Double_t it    ) { fInitialTemperature = it; }
      void SetMinTemp     ( Double_t min   ) { fMinTemperature = min; }
      void SetNumFunLoops ( Int_t    num   ) { fNFunLoops = num; }
      void SetAccuracy    ( Double_t eps   ) { fEps = eps; }
      void SetNEps        ( Int_t    neps  ) { fNEps = neps; }

   private:

      Double_t GetPerturbationProbability( Double_t energy, Double_t energyRef, 
                                           Double_t temperature );

      IFitterTarget&                fFitterTarget;           // the fitter target
      TRandom*                      fRandom;                 // random generator
      const std::vector<Interval*>& fRanges;                 // parameter ranges

      // fitter setup 
      Int_t                         fMaxCalls;               // maximum number of minimisation calls
      Double_t                      fTemperatureGradient;    // temperature gradient
      Bool_t                        fUseAdaptiveTemperature; // use adaptive termperature
      Double_t                      fInitialTemperature;     // initial temperature
      Double_t                      fMinTemperature;         // mimimum temperature
      Double_t                      fEps;                    // epsilon
      Int_t                         fNFunLoops;              // number of function loops
      Int_t                         fNEps;                   // number of epsilons                    

      mutable MsgLogger             fLogger;                 // message logger

      ClassDef(SimulatedAnnealing,0)  // Base class for Simulated Annealing fitting
   };   

} // namespace TMVA

#endif

