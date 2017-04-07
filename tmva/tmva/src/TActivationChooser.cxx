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

/*! \class TMVA::TActivationChooser
\ingroup TMVA
Class for easily choosing activation functions
*/

#include "TMVA/TActivationChooser.h"

#include <vector>
#include "TString.h"

#include "TMVA/TActivation.h"
#include "TMVA/TActivationIdentity.h"
#include "TMVA/TActivationSigmoid.h"
#include "TMVA/TActivationTanh.h"
#include "TMVA/TActivationReLU.h"
#include "TMVA/TActivationRadial.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Types.h"


////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TMVA::TActivationChooser::TActivationChooser() :
   fLINEAR("linear"),
   fSIGMOID("sigmoid"),
   fTANH("tanh"),
   fRELU("ReLU"),
   fRADIAL("radial"),
   fLogger( new MsgLogger("TActivationChooser") )
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TMVA::TActivationChooser::~TActivationChooser()
{
   delete fLogger;
}

////////////////////////////////////////////////////////////////////////////////
/// instantiate the correct activation object according to the
/// type chosen (given as the enumeration type)

TMVA::TActivation*
TMVA::TActivationChooser::CreateActivation(EActivationType type) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// instantiate the correct activation object according to the
/// type chosen (given by a TString)

TMVA::TActivation*
TMVA::TActivationChooser::CreateActivation(const TString& type) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// returns the names of all know activation functions

std::vector<TString>*
TMVA::TActivationChooser::GetAllActivationNames() const
{
   std::vector<TString>* names = new std::vector<TString>();
   names->push_back(fLINEAR);
   names->push_back(fSIGMOID);
   names->push_back(fTANH);
   names->push_back(fRELU);
   names->push_back(fRADIAL);
   return names;
}
