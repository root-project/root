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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Identity transformation of input variables                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/VariableIdentityTransform.h"
#include <iostream>

ClassImp(TMVA::VariableIdentityTransform)

//_______________________________________________________________________
TMVA::VariableIdentityTransform::VariableIdentityTransform( std::vector<VariableInfo>& varinfo )
   : VariableTransformBase( varinfo, Types::kNone )
{
   // constructor
   SetName("NoTransform");
}

//_______________________________________________________________________
Bool_t TMVA::VariableIdentityTransform::PrepareTransformation( TTree* inputTree)
{
   // the identity does not need to be prepared, only calculate the normalization
   if (!IsEnabled() || IsCreated()) return kTRUE;

   SetCreated( kTRUE );

   CalcNorm( inputTree );

   return kTRUE;
}

//_______________________________________________________________________
std::vector<TString>* TMVA::VariableIdentityTransform::GetTransformationStrings( Types::ESBType ) const
{
   // creates string with variable transformations applied (here, just untransformed variables)

   std::vector<TString>* strVec = new std::vector<TString>;

   // fill vector
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) {
      TString str( Variable(ivar).GetExpression() );
      strVec->push_back( TString("[") + str + "]" );
   }      

   return strVec;
}

//_______________________________________________________________________
void TMVA::VariableIdentityTransform::ApplyTransformation( Types::ESBType ) const
{
   // this transformation doesn't do anything
   if      (fEvent == fEventRaw) return;
   else if (fEvent != 0)         delete fEvent;
   else    fEvent = fEventRaw;
}

//_______________________________________________________________________
void TMVA::VariableIdentityTransform::MakeFunction(std::ostream& fout, const TString& fncName, Int_t /*part*/) {
   fout << "inline void " << fncName << "::InitTransform() {}" << std::endl;
   fout << std::endl;
   fout << "inline void " << fncName << "::Transform(const std::vector<double> &, int) const {}" << std::endl;
}
