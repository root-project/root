// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Types                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      GLobal types (singleton class)                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#ifndef ROOT_TMVA_Types
#define ROOT_TMVA_Types

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Types                                                                //
//                                                                      //
// Singleton class for Global types used by TMVA                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <map>

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

namespace TMVA {

   typedef UInt_t TMVAVersion_t;

   class MsgLogger;

   // message types for MsgLogger
   // define outside of Types class to facilite access
   enum EMsgType {
      kDEBUG   = 1,
      kVERBOSE = 2,
      kINFO    = 3,
      kWARNING = 4,
      kERROR   = 5,
      kFATAL   = 6,
      kSILENT  = 7
   };

   class Types {

   public:

      // available MVA methods
      enum EMVA {
         kVariable    = 0,
         kCuts           ,
         kLikelihood     ,
         kPDERS          ,
         kHMatrix        ,
         kFisher         ,
         kKNN            ,
         kCFMlpANN       ,
         kTMlpANN        ,
         kBDT            ,
         kDT             ,
         kRuleFit        ,
         kSVM            ,
         kMLP            ,
         kBayesClassifier,
         kFDA            ,
         kCommittee      ,
         kBoost          ,
         kPDEFoam        ,
         kLD             ,
         kPlugins        ,
         kCategory       ,
         kMaxMethod
      };

      // available variable transformations
      enum EVariableTransform {
         kIdentity = 0,
         kDecorrelated,
         kNormalized,
         kPCA,
         kRearranged,
         kGauss,
         kUniform,
         kMaxVariableTransform
      };

      // type of analysis
      enum EAnalysisType {
         kClassification = 0,
         kRegression,
         kMulticlass,
         kNoAnalysisType,
         kMaxAnalysisType
      };

      enum ESBType {
         kSignal = 0,  // Never change this number - it is elsewhere assumed to be zero !
         kBackground,
         kSBBoth,
         kMaxSBType,
         kTrueType
      };

      enum ETreeType {
         kTraining = 0,
         kTesting,
         kMaxTreeType,
         kValidation,
         kTrainingOriginal
      };

      enum EBoostStage {
         kBoostProcBegin=0,
         kBeforeTraining,
         kBeforeBoosting,
         kAfterBoosting,
         kBoostValidation,
         kBoostProcEnd
      };

   public:

      static Types& Instance();
      static void   DestroyInstance();
      ~Types();

      Types::EMVA   GetMethodType( const TString& method ) const;
      TString       GetMethodName( Types::EMVA    method ) const;

      Bool_t        AddTypeMapping(Types::EMVA method, const TString& methodname);

   private:

      Types();
      static Types* fgTypesPtr;

   private:

      std::map<TString, TMVA::Types::EMVA> fStr2type; // types-to-text map
      mutable MsgLogger* fLogger;   // message logger
      MsgLogger& Log() const { return *fLogger; }

   };
}

#endif
