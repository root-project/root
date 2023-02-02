// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : FitterBase                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Base class for TMVA fitters                                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *      Joerg Stelzer    <Joerg.Stelzer@cern.ch>  - CERN, Switzerland             *
 *      Helge Voss       <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_FitterBase
#define ROOT_TMVA_FitterBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// FitterBase                                                           //
//                                                                      //
// Baseclass for TMVA fitters                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include<vector>
#include "TObject.h"
#include "TString.h"

#include "TMVA/Configurable.h"

namespace TMVA {

   class Interval;
   class IFitterTarget;
   class MsgLogger;

   class FitterBase : public Configurable {

   public:

      FitterBase( IFitterTarget& target, const TString& name, const std::vector<TMVA::Interval*> ranges,
                  const TString& theOption );

      virtual ~FitterBase() {}

      Double_t Run();
      virtual Double_t Run( std::vector<Double_t>& pars ) = 0;

      Double_t       EstimatorFunction( std::vector<Double_t>& parameters );
      IFitterTarget& GetFitterTarget() const { return fFitterTarget; }

      // accessor
      Int_t GetNpars() const { return fNpars; }

      // remove namespace in name
      const char* GetName() const { return fClassName; }

      // setting up variables for JsMVA interactive training
      void SetIPythonInteractive(bool* ExitFromTraining, UInt_t *fIPyMaxIter_, UInt_t *fIPyCurrentIter_){
        fExitFromTraining = ExitFromTraining;
        fIPyMaxIter = fIPyMaxIter_;
        fIPyCurrentIter = fIPyCurrentIter_;
      }

   protected:

      // need to implement option declaration
      virtual void DeclareOptions() = 0;

      IFitterTarget&                      fFitterTarget; // pointer to target of fitting procedure
      const std::vector<TMVA::Interval*>  fRanges;       // allowed intervals
      Int_t                               fNpars;        // number of parameters

      mutable MsgLogger*                  fLogger;       // message logger
      MsgLogger& Log() const { return *fLogger; }

      TString                             fClassName;    // remove TMVA:: from TObject name

      // variables needed by JsMVA
      UInt_t *fIPyCurrentIter = nullptr, *fIPyMaxIter = nullptr;
      bool* fExitFromTraining = nullptr;

      ClassDef(FitterBase,0); // Baseclass for fitters
   };

} // namespace TMVA

#endif
