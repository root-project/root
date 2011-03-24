// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableRearrangeTransform                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch> - CERN, Switzerland           *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iostream>
#include <iomanip>
#include <stdexcept>

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_VariableRearrangeTransform
#include "TMVA/VariableRearrangeTransform.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif

ClassImp(TMVA::VariableRearrangeTransform)

//_______________________________________________________________________
TMVA::VariableRearrangeTransform::VariableRearrangeTransform( DataSetInfo& dsi )
:  VariableTransformBase( dsi, Types::kRearranged, "Rearrange" )
{ 
   // constructor
}

//_______________________________________________________________________
TMVA::VariableRearrangeTransform::~VariableRearrangeTransform() {
}

//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::Initialize()
{
   // initialization of the rearrangement transformation
   // (nothing to do)
}

//_______________________________________________________________________
Bool_t TMVA::VariableRearrangeTransform::PrepareTransformation( const std::vector<Event*>& /*events*/ )
{
   // prepare transformation --> (nothing to do)
   if (!IsEnabled() || IsCreated()) return kTRUE;

   UInt_t nvars = 0, ntgts = 0, nspcts = 0;
   CountVariableTypes( nvars, ntgts, nspcts );
   if (ntgts>0) Log() << kFATAL << "Targets used in Rearrange-transformation." << Endl;

   SetCreated( kTRUE );
   return kTRUE;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableRearrangeTransform::Transform( const TMVA::Event* const ev, Int_t /*cls*/ ) const
{
   if (!IsEnabled()) return ev;

   // apply the normalization transformation
   if (!IsCreated()) Log() << kFATAL << "Transformation not yet created" << Endl;

   if (fTransformedEvent==0) fTransformedEvent = new Event();

   FloatVector input; // will be filled with the selected variables, (targets)
   std::vector<Char_t> mask; // masked variables
   GetInput( ev, input, mask );
   SetOutput( fTransformedEvent, input, mask, ev );

   return fTransformedEvent;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableRearrangeTransform::InverseTransform( const TMVA::Event* const ev, Int_t /*cls*/ ) const
{
   if (!IsEnabled()) return ev;

   // apply the inverse transformation
   if (!IsCreated()) Log() << kFATAL << "Transformation not yet created" << Endl;

   if (fBackTransformedEvent==0) fBackTransformedEvent = new Event( *ev );

   FloatVector input;  // will be filled with the selected variables, targets, (spectators)
   std::vector<Char_t> mask; // masked variables
   GetInput( ev, input, mask, kTRUE );
   SetOutput( fBackTransformedEvent, input, mask, ev, kTRUE );

   return fBackTransformedEvent;
}


//_______________________________________________________________________
std::vector<TString>* TMVA::VariableRearrangeTransform::GetTransformationStrings( Int_t /*cls*/ ) const
{
   // creates string with variable transformations applied

   const UInt_t size = fGet.size();
   std::vector<TString>* strVec = new std::vector<TString>(size);

   return strVec;
}

//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::AttachXMLTo(void* parent) 
{
//    // create XML description of Rearrange transformation
   void* trfxml = gTools().AddChild(parent, "Transform");
   gTools().AddAttr(trfxml, "Name", "Rearrange");

   VariableTransformBase::AttachXMLTo( trfxml );
}

//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::ReadFromXML( void* trfnode ) 
{
//    // Read the transformation matrices from the xml node


   Bool_t newFormat = kFALSE;

   void* inpnode = NULL;

   inpnode = gTools().GetChild(trfnode, "Selection"); // new xml format
   if(inpnode == NULL)
      Log() << kFATAL << "Unknown weight file format for transformations. (tried to read in 'rearrange' transform)" << Endl;
   newFormat = kTRUE;
   
   VariableTransformBase::ReadFromXML( inpnode );
   
   SetCreated();
}

//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::PrintTransformation( ostream& ) 
{
   // prints the transformation ranges
}

//_______________________________________________________________________
void TMVA::VariableRearrangeTransform::MakeFunction( std::ostream& /*fout*/, const TString& /*fcncName*/, 
                                                     Int_t /*part*/, UInt_t /*trCounter*/, Int_t ) 
{
   // creates a normalizing function
}
