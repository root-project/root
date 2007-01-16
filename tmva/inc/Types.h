// @(#)root/tmva $Id: Types.h,v 1.10 2006/11/23 17:43:39 rdm Exp $   
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

#include <map>
#include "Riostream.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {

   typedef std::pair<Double_t,Double_t> LowHigh_t;

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
         kSVM            ,
         kMLP            ,
         kBayesClassifier,
         kCommittee      ,
         kMaxMethod
      };

      enum EPreprocessingMethod {
         kNone = 0,
         kDecorrelated,
         kPCA,
         kMaxPreprocessingMethod
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

      EMVA GetMethodType( const TString& method ) const { 
         std::map<TString, EMVA>::const_iterator it = fStr2type.find( method );
         if (it == fStr2type.end()) {
            fLogger << kFATAL << "unknown method in map: " << method << Endl;
            return kVariable; // Inserted to get rid of GCC warning...
         }
         else return it->second;
      }

   private:

      Types();
      
      static Types* fgTypesPtr;
                  
   private:
         
      std::map<TString, EMVA> fStr2type; // types-to-text map
      mutable MsgLogger            fLogger;   // message logger
         
   };
}

#endif
