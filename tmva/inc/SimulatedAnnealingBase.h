// @(#)root/tmva $Id: SimulatedAnnealingBase.h,v 1.7 2007/01/16 14:28:35 brun Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SimulatedAnnealingBase                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Base class for simulated annealing fitting procedure                      *
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

#ifndef ROOT_TMVA_SimulatedAnnealingBase
#define ROOT_TMVA_SimulatedAnnealingBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// SimulatedAnnealingBase                                               //
//                                                                      //
// Base class for Simulated Annealing fitting procedure                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#include "TObject.h"
#include "TRandom.h"

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_Interval
#include "TMVA/Interval.h"
#endif

namespace TMVA {

   class SimulatedAnnealingBase : public TObject {

   public:

      SimulatedAnnealingBase( std::vector<Interval*>& ranges );
      virtual ~SimulatedAnnealingBase();

      virtual Double_t MinimizeFunction( const std::vector<Double_t>& parameters ) = 0;
   
      // returns FCN value at minimum
      Double_t Minimize( std::vector<Double_t>& parameters );
      Double_t GetPerturbationProbability( Double_t energy, Double_t energyRef, 
                                           Double_t temperature);
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

      TRandom*                fRandom;                 // random generator
      std::vector<Interval*> fRanges;                 // parameter ranges

      // fitter setup 
      Int_t                   fMaxCalls;               // maximum number of minimisation calls
      Double_t                fTemperatureGradient;    // temperature gradient
      Bool_t                  fUseAdaptiveTemperature; // use adaptive termperature
      Double_t                fInitialTemperature;     // initial temperature
      Double_t                fMinTemperature;         // mimimum temperature
      Double_t                fEps;                    // epsilon
      Int_t                   fNFunLoops;              // number of function loops
      Int_t                   fNEps;                   // number of epsilons                    

      ClassDef(SimulatedAnnealingBase,0)  // Base class for Simulated Annealing fitting
   };   

} // namespace TMVA

#endif

