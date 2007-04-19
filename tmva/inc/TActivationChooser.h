// @(#)root/tmva $Id: TActivationChooser.h,v 1.7 2006/11/23 17:43:39 rdm Exp $
// Author: Matt Jachowski 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TActivationChooser                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Class for easily choosing activation functions.                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/
 

#ifndef ROOT_TMVA_TActivationChooser
#define ROOT_TMVA_TActivationChooser

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TActivationChooser                                                   //
//                                                                      //
// Class for easily choosing activation functions                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include "TString.h"

#ifndef ROOT_TMVA_TActivation
#include "TActivation.h"
#endif
#ifndef ROOT_TMVA_TActivationIdentity
#include "TActivationIdentity.h"
#endif
#ifndef ROOT_TMVA_TActivationSigmoid
#include "TActivationSigmoid.h"
#endif
#ifndef ROOT_TMVA_TActivationTanh
#include "TActivationTanh.h"
#endif
#ifndef ROOT_TMVA_TActivationRadial
#include "TActivationRadial.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {
  
   class TActivationChooser {
    
   public:

      TActivationChooser()
         : fLogger( "TActivationChooser" )
      {
        // defaut constructor 

         fLINEAR  = "linear";
         fSIGMOID = "sigmoid";
         fTANH    = "tanh";
         fRADIAL  = "radial";
      }
      virtual ~TActivationChooser() {}

      enum EActivationType { kLinear = 0,
                             kSigmoid,
                             kTanh,
                             kRadial
      };

      TActivation* CreateActivation(EActivationType type) const
      {
        // instantiate the correct activation object according to the
        // type choosen (given as the enumeration type)

         switch (type) {
         case kLinear:  return new TActivationIdentity();
         case kSigmoid: return new TActivationSigmoid(); 
         case kTanh:    return new TActivationTanh();    
         case kRadial:  return new TActivationRadial();  
         default:
            fLogger << kFATAL << "no Activation function of type " << type << " found" << Endl;
            return 0; 
         }
         return NULL;
      }
      
      TActivation* CreateActivation(const TString type) const
      {
        // instantiate the correct activation object according to the
        // type choosen (given by a TString)

         if      (type == fLINEAR)  return CreateActivation(kLinear);
         else if (type == fSIGMOID) return CreateActivation(kSigmoid);
         else if (type == fTANH)    return CreateActivation(kTanh);
         else if (type == fRADIAL)  return CreateActivation(kRadial);
         else {
            fLogger << kFATAL << "no Activation function of type " << type << " found" << Endl;
            return 0;
         }
      }
      
      vector<TString>* GetAllActivationNames() const
      {
        // retuns the names of all know activation functions

         vector<TString>* names = new vector<TString>();
         names->push_back(fLINEAR);
         names->push_back(fSIGMOID);
         names->push_back(fTANH);
         names->push_back(fRADIAL);
         return names;
      }

   private:

      TString fLINEAR;  // activation function name
      TString fSIGMOID; // activation function name
      TString fTANH;    // activation function name
      TString fRADIAL;  // activation function name

      mutable MsgLogger fLogger; // message logger

      ClassDef(TActivationChooser,0) // Class for choosing activation functions
   };

} // namespace TMVA

#endif
