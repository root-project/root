// @(#)root/tmva $\Id$
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
 *      CERN, Switzerland,                                                        *
 *      MPI-K Heidelberg, Germany ,                                               *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/


#include "TMVA/VariableIdentityTransform.h"

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
void TMVA::VariableIdentityTransform::ApplyTransformation( Types::ESBType ) const
{
   cout << "Test" << endl;
   // this transformation doesn't do anything
   if      (fEvent == fEventRaw) return;
   else if (fEvent != 0)         delete fEvent;
   else    fEvent = fEventRaw;
}
