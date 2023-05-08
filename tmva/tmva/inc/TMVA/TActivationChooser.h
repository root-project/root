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

namespace TMVA {

   class TActivation;
   class MsgLogger;

   class TActivationChooser {
   public:

      enum EActivationType { kLinear = 0,
                             kSigmoid,
                             kTanh,
                             kReLU,
                             kRadial
      };

      TActivationChooser();
      virtual ~TActivationChooser();

      TActivation* CreateActivation(EActivationType type) const;
      TActivation* CreateActivation(const TString& type) const;
      std::vector<TString>* GetAllActivationNames() const;

   private:

      TString fLINEAR;  ///< activation function name
      TString fSIGMOID; ///< activation function name
      TString fTANH;    ///< activation function name
      TString fRELU;    ///< activation function name
      TString fRADIAL;  ///< activation function name

      mutable MsgLogger* fLogger;                     ///<! message logger
      MsgLogger& Log() const { return *fLogger; }

      ClassDef(TActivationChooser,0); // Class for choosing activation functions
   };

} // namespace TMVA

#endif
