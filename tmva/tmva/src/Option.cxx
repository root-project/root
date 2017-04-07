// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Option                                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

/*! \class TMVA::OptionBase
\ingroup TMVA
Class for TMVA-option handling
*/

#include "TMVA/Option.h"

#include "TMVA/Types.h"

#include "ThreadLocalStorage.h"
#include "TObject.h"
#include "TString.h"


////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::OptionBase::OptionBase( const TString& name, const TString& desc )
   : TObject(),
     fName        ( name ),
     fNameAllLower( name ),
     fDescription ( desc ),
     fIsSet       ( kFALSE )
{
   fNameAllLower.ToLower();
}

////////////////////////////////////////////////////////////////////////////////
/// set value for option

Bool_t TMVA::OptionBase::SetValue( const TString& vs, Int_t )
{
   fIsSet = kTRUE;
   SetValueLocal(vs);
   return kTRUE;
}

TMVA::MsgLogger& TMVA::OptionBase::Log()
{
   TTHREAD_TLS_DECL_ARG2(MsgLogger,logger,"Option",kDEBUG);  // message logger
   return logger;
}
