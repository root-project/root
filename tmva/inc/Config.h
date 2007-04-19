// @(#)root/tmva $\Id$
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Config                                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      GLobal configuration settings (singleton)                                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker    <Andreas.Hocker@cern.ch>     - CERN, Switzerland       *
 *      Joerg Stelzer      <Joerg.Stelzer@cern.ch>      - CERN, Switzerland       *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch> - Iowa State U., USA      *
 *      Helge Voss         <Helge.Voss@cern.ch>         - MPI-K Heidelberg, GER   *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland,                                                        *
 *      Iowa State U., USA,                                                       *
 *      MPI-K Heidelberg, Germany,                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#ifndef ROOT_TMVA_Config
#define ROOT_TMVA_Config

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Config                                                               //
//                                                                      //
// Singleton class for global configuration settings used by TMVA       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "Riostream.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {

   class Config {
               
   public:

      static Config& Instance() { return fgConfigPtr ? *fgConfigPtr : *(fgConfigPtr = new Config()); }
      virtual ~Config();

      Bool_t UseColor() { return fUseColoredConsole; }
      void SetUseColor(Bool_t uc) { fUseColoredConsole = uc; }

   public:

      // publicly accessible global settings
      struct VariablePlotting {
         Float_t timesRMS;
         Int_t   nbins1D;
         Int_t   nbins2D;
         Int_t   maxNumOfAllowedVariablesForScatterPlots;
      } variablePlotting;

      // for file names and similar
      struct IONames {
         TString weightFileDir;
         TString weightFileExtension;
      } ioNames;
         
      
   private:

      // private constructor
      Config();
      static Config* fgConfigPtr;
                  
   private:

      Bool_t fUseColoredConsole;
         
      mutable MsgLogger fLogger;   // message logger
         
      ClassDef(Config,0) // Singleton class for global configuration settings
   };

   // global accessor
   Config& gConfig();
}

#endif
