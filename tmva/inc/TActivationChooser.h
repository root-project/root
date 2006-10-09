// @(#)root/tmva $Id: TActivationChooser.h,v 1.3 2006/08/30 22:19:59 andreas.hoecker Exp $
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

namespace TMVA {
  
   class TActivationChooser {
    
   public:

      TActivationChooser()
      {
         LINEAR  = "linear";
         SIGMOID = "sigmoid";
         TANH    = "tanh";
         RADIAL  = "radial";
      }
      virtual ~TActivationChooser() {}

      enum ActivationType { kLinear = 0,
                            kSigmoid,
                            kTanh,
                            kRadial
      };

      TActivation* CreateActivation(const ActivationType type) const
      {
         switch (type) {
         case kLinear:  return new TActivationIdentity();
         case kSigmoid: return new TActivationSigmoid();
         case kTanh:    return new TActivationTanh();
         case kRadial:  return new TActivationRadial();
         default:
            cout << "ERROR: no Activation function of type " << type << " found" << endl;
            exit(1);
         }
         return NULL;
      }
      
      TActivation* CreateActivation(const TString type) const
      {
         if      (type == LINEAR)  return CreateActivation(kLinear);
         else if (type == SIGMOID) return CreateActivation(kSigmoid);
         else if (type == TANH)    return CreateActivation(kTanh);
         else if (type == RADIAL)  return CreateActivation(kRadial);
         else {
            cout << "ERROR: no Activation function of type " << type << " found" << endl;
            exit(1);
         }
      }
      
      vector<TString>* GetAllActivationNames() const
      {
         vector<TString>* names = new vector<TString>();
         names->push_back(LINEAR);
         names->push_back(SIGMOID);
         names->push_back(TANH);
         names->push_back(RADIAL);
         return names;
      }

   private:

      TString LINEAR; 
      TString SIGMOID;
      TString TANH;   
      TString RADIAL; 

      ClassDef(TActivationChooser,0) // Class for choosing activation functions
         };

} // namespace TMVA

#endif
