// @(#)root/tree:$Id$
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TLeafElement
\ingroup tree

A TLeaf for the general case when using the branches created via
a TStreamerInfo (i.e. using TBranchElement).
*/

#include "TLeafElement.h"
//#include "TMethodCall.h"

#include "TVirtualStreamerInfo.h"

ClassImp(TLeafElement)

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for LeafObject.

TLeafElement::TLeafElement(): TLeaf()
{
   fAbsAddress = 0;
   fID   = -1;
   fType = -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a LeafObject.

TLeafElement::TLeafElement(TBranch *parent, const char *name, Int_t id, Int_t type)
   :TLeaf(parent, name,name)
{
   fAbsAddress = 0;
   fID         = id;
   fType       = type;
   if (type < TVirtualStreamerInfo::kObject) {
      Int_t bareType = type;
      if (bareType > TVirtualStreamerInfo::kOffsetP)
         bareType -= TVirtualStreamerInfo::kOffsetP;
      else if (bareType > TVirtualStreamerInfo::kOffsetL)
         bareType -= TVirtualStreamerInfo::kOffsetL;

      if ((bareType >= TVirtualStreamerInfo::kUChar && bareType <= TVirtualStreamerInfo::kULong)
          || bareType == TVirtualStreamerInfo::kULong64)
      SetUnsigned();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Default destructor for a LeafObject.

TLeafElement::~TLeafElement()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Returns pointer to method corresponding to name name is a string
/// with the general form "method(list of params)" If list of params is
/// omitted, () is assumed;

TMethodCall *TLeafElement::GetMethodCall(const char * /*name*/)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if this leaf is does not have any sub-branch/leaf.

Bool_t TLeafElement::IsOnTerminalBranch() const
{
   if (fBranch->GetListOfBranches()->GetEntriesFast()) return kFALSE;
   return kTRUE;
}
