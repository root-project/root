// @(#)root/tmva $Id: Types.h,v 1.11 2007/01/16 09:37:03 brun Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Types                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      GLobal types (singleton)                                                  *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
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
// Singleton class for GLobal types used by TMVA                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "Riostream.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {

   //typedef std::pair<Double_t,Double_t> LowHigh_t;

   class Types {
      
   public:
         
      // available MVA methods in TMVA
      enum EMVA {
         kVariable    = 0,
         kCuts           ,     
         kLikelihood     ,
         kPDERS          ,
         kHMatrix        ,
         kFisher         ,
         kCFMlpANN       ,
         kTMlpANN        , 
         kBDT            ,     
         kRuleFit        ,
         kRuleFitJF      ,
         kSVM            ,
         kMLP            ,
         kBayesClassifier,
         kCommittee      ,
         kMaxMethod
      };

      enum EVariableTransform {
         kNone = 0,
         kDecorrelated,
         kPCA,
         kMaxVariableTransform
      };

      enum ESBType { 
         kSignal = 0, 
         kBackground, 
         kSBBoth, 
         kMaxSBType,
         kTrueType
      };

      enum ETreeType { kTraining = 0, kTesting, kMaxTreeType };

   public:

      static Types& Instance() { return fgTypesPtr ? *fgTypesPtr : *(fgTypesPtr = new Types()); }
      ~Types() {}

      EMVA GetMethodType( const TString& method ) const;

   private:

      Types();
      
      static Types* fgTypesPtr;
                  
   private:
         
      std::map<TString, EMVA> fStr2type; // types-to-text map
      mutable MsgLogger       fLogger;   // message logger
         
   };
}

#endif
