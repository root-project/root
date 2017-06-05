// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableIdentityTransform                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::VariableIdentityTransform
\ingroup TMVA
Linear interpolation class
*/

#include "TMVA/VariableIdentityTransform.h"

#include "TMVA/Event.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/SVEvent.h"
#include "TMVA/Types.h"
#include "TMVA/VariableTransformBase.h"

#include "Rtypes.h"
#include "TString.h"

#include <iostream>

namespace TMVA {
   class DataSetInfo;
}

ClassImp(TMVA::VariableIdentityTransform);

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::VariableIdentityTransform::VariableIdentityTransform( DataSetInfo& dsi )
: VariableTransformBase( dsi, Types::kIdentity, "Id" )
{
}

////////////////////////////////////////////////////////////////////////////////
/// nothing to initialize

void TMVA::VariableIdentityTransform::Initialize()
{
}

////////////////////////////////////////////////////////////////////////////////
/// the identity does not need to be prepared, only calculate the normalization

Bool_t TMVA::VariableIdentityTransform::PrepareTransformation (const std::vector<Event*>& events)
{
   Initialize();

   if (!IsEnabled() || IsCreated()) return kTRUE;

   Log() << kDEBUG << "Preparing the Identity transformation..." << Endl;

   if( fGet.size() < events[0]->GetNVariables() )
      Log() << kFATAL << "Identity transform does not allow for a selection of input variables. Please remove the variable selection option and put only 'I'." << Endl;

   SetNVariables(events[0]->GetNVariables());

   SetCreated( kTRUE );

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// identity transformation to write to XML
///Log() << kFATAL << "Please implement writing of transformation as XML" << Endl;

void TMVA::VariableIdentityTransform::AttachXMLTo( void* )
{
}

////////////////////////////////////////////////////////////////////////////////
/// reding the identity transformation from XML

void TMVA::VariableIdentityTransform::ReadFromXML( void* )
{
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// identity transform returns same event

const TMVA::Event* TMVA::VariableIdentityTransform::Transform (const TMVA::Event* const ev, Int_t ) const
{
   return ev;
}

////////////////////////////////////////////////////////////////////////////////
/// creates C++ code fragment of the identity transform for inclusion in standalone C++ class

void TMVA::VariableIdentityTransform::MakeFunction( std::ostream& fout, const TString& fncName,
                                                    Int_t , UInt_t trCounter, Int_t )
{
   fout << "inline void " << fncName << "::InitTransform_Identity_" << trCounter << "() {}" << std::endl;
   fout << std::endl;
   fout << "inline void " << fncName << "::Transform_Identity_" << trCounter << "(const std::vector<double> &, int) const {}" << std::endl;
}
