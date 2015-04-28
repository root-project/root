// @(#)root/tmva $Id$
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
 
#include "TMVA/TActivationChooser.h"


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
#include "TMVA/TActivation.h"
#endif
#ifndef ROOT_TMVA_TActivationIdentity
#include "TMVA/TActivationIdentity.h"
#endif
#ifndef ROOT_TMVA_TActivationSigmoid
#include "TMVA/TActivationSigmoid.h"
#endif
#ifndef ROOT_TMVA_TActivationTanh
#include "TMVA/TActivationTanh.h"
#endif
#ifndef ROOT_TMVA_TActivationReLU
#include "TMVA/TActivationReLU.h"
#endif
#ifndef ROOT_TMVA_TActivationRadial
#include "TMVA/TActivationRadial.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif


TMVA::TActivationChooser::TActivationChooser() :
   fLINEAR("linear"),
   fSIGMOID("sigmoid"),
   fTANH("tanh"),
   fRELU("ReLU"),
   fRADIAL("radial"),
   fLogger( new MsgLogger("TActivationChooser") )
{
   // defaut constructor   
}

TMVA::TActivationChooser::~TActivationChooser()
{
   // destructor
   delete fLogger;
}

TMVA::TActivation*
TMVA::TActivationChooser::CreateActivation(EActivationType type) const
{
   // instantiate the correct activation object according to the
   // type choosen (given as the enumeration type)
   
   switch (type) {
   case kLinear:  return new TActivationIdentity();
   case kSigmoid: return new TActivationSigmoid(); 
   case kTanh:    return new TActivationTanh();    
   case kReLU:    return new TActivationReLU();    
   case kRadial:  return new TActivationRadial();  
   default:
      Log() << kFATAL << "no Activation function of type " << type << " found" << Endl;
      return 0; 
   }
   return NULL;
}
      
TMVA::TActivation*
TMVA::TActivationChooser::CreateActivation(const TString& type) const
{
   // instantiate the correct activation object according to the
   // type choosen (given by a TString)

   if      (type == fLINEAR)  return CreateActivation(kLinear);
   else if (type == fSIGMOID) return CreateActivation(kSigmoid);
   else if (type == fTANH)    return CreateActivation(kTanh);
   else if (type == fRELU)    return CreateActivation(kReLU);
   else if (type == fRADIAL)  return CreateActivation(kRadial);
   else {
      Log() << kFATAL << "no Activation function of type " << type << " found" << Endl;
      return 0;
   }
}
      
std::vector<TString>*
TMVA::TActivationChooser::GetAllActivationNames() const
{
   // retuns the names of all know activation functions

   std::vector<TString>* names = new std::vector<TString>();
   names->push_back(fLINEAR);
   names->push_back(fSIGMOID);
   names->push_back(fTANH);
   names->push_back(fRELU);
   names->push_back(fRADIAL);
   return names;
}


