// @(#)root/tree:$Id$
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TLeaf for the general case when using the branches created via     //
// a TStreamerInfo (i.e. using TBranchElement).                         //
//////////////////////////////////////////////////////////////////////////

#include "TLeafElement.h"
//#include "TMethodCall.h"


ClassImp(TLeafElement)

//______________________________________________________________________________
TLeafElement::TLeafElement(): TLeaf()
{
   // Default constructor for LeafObject.

   fAbsAddress = 0;
   fID   = -1;
   fType = -1;
}

//______________________________________________________________________________
TLeafElement::TLeafElement(TBranch *parent, const char *name, Int_t id, Int_t type)
   :TLeaf(parent, name,name)
{
   // Create a LeafObject.
   
   fAbsAddress = 0;
   fID         = id;
   fType       = type;
}

//______________________________________________________________________________
TLeafElement::~TLeafElement()
{
   // Default destructor for a LeafObject.

}

//______________________________________________________________________________
TMethodCall *TLeafElement::GetMethodCall(const char * /*name*/)
{
   // Returns pointer to method corresponding to name name is a string
   // with the general form "method(list of params)" If list of params is
   // omitted, () is assumed;

   return 0;
}


//______________________________________________________________________________
Bool_t TLeafElement::IsOnTerminalBranch() const
{
   // Return true if this leaf is does not have any sub-branch/leaf.

   if (fBranch->GetListOfBranches()->GetEntriesFast()) return kFALSE;
   return kTRUE;
}
