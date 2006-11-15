// @(#)root/tmva $Id: Types.h,v 1.25 2006/11/13 23:43:34 stelzer Exp $   
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
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
#include "TROOT.h"
#include "Riostream.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {

   typedef std::pair<Double_t,Double_t> LowHigh_t;

   class Types {
      
   public:
         
      // available MVA methods in TMVA
      enum MVA {
         Variable    = 0,
         Cuts           ,     
         Likelihood     ,
         PDERS          ,
         HMatrix        ,
         Fisher         ,
         CFMlpANN       ,
         TMlpANN        , 
         BDT            ,     
         RuleFit        ,
         SVM            ,
         MLP            ,
         BayesClassifier,
         Committee      ,
         kMaxMethod
      };

      enum PreprocessingMethod {
         kNone = 0,
         kDecorrelated,
         kPCA,
         kMaxPreprocessingMethod
      };

      enum SBType { 
         kSignal = 0, 
         kBackground, 
         kSBBoth, 
         kMaxSBType,
         kTrueType
      };

      enum TreeType { kTrain = 1, kTest };
         
   public:

      static Types& Instance() { return fgTypesPtr ? *fgTypesPtr : *(fgTypesPtr = new Types()); }
      ~Types() {}

      const MVA GetMethodType( const TString& method ) const { 
         std::map<TString, MVA>::const_iterator it = fStr2type.find( method );
         if (it == fStr2type.end()) {
            fLogger << kFATAL << "unknown method in map: " << method << Endl;
            return Variable; // Inserted to get rid of GCC warning...
         }
         else return it->second;
      }

   private:

      Types();
      
      static Types* fgTypesPtr;
                  
   private:
         
      std::map<TString, MVA> fStr2type; // types-to-text map
      mutable MsgLogger            fLogger;   // message logger
         
   };
}

#endif
