// @(#)root/tmva $Id: SimulatedAnnealingBase.h,v 1.5 2006/08/30 22:19:59 andreas.hoecker Exp $   
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

#ifndef ROOT_TMVA_SimulatedAnnealingBase
#define ROOT_TMVA_SimulatedAnnealingBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// SimulatedAnnealingBase                                               //
//                                                                      //
// Base class for Simulated Annealing fitting procedure                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TRandom.h"

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif

namespace TMVA {

   class SimulatedAnnealingBase : public TObject {

   public:

      SimulatedAnnealingBase( std::vector<LowHigh_t*>& ranges );
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

      TRandom*                fRandom;
      std::vector<LowHigh_t*> fRanges;

      // fitter setup 
      Int_t                   fMaxCalls;
      Double_t                fTemperatureGradient;
      Bool_t                  fUseAdaptiveTemperature;
      Double_t                fInitialTemperature;
      Double_t                fMinTemperature;
      Double_t                fEps;
      Int_t                   fNFunLoops;
      Int_t                   fNEps;

      ClassDef(SimulatedAnnealingBase,0)  // Base class for Simulated Annealing fitting

   };   
} // namespace TMVA

#endif

