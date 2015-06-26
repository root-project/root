// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TObjectSet.h"
#include "TBrowser.h"

ClassImp(TObjectSet)

//////////////////////////////////////////////////////////////////////////////////////
//                                                                                  //
//  TObjectSet  - is a container TDataSet                                           //
//                  This means this object has an extra pointer to an embedded      //
//                  TObject.                                                        //
//  Terminology:    This TObjectSet may be an OWNER of the embeded TObject          //
//                  If the container is the owner it can delete the embeded object  //
//                  otherwsie it leaves that object "as is"                         //
//                                                                                  //
//////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///to be documented

TObjectSet::TObjectSet(const Char_t *name, TObject *obj, Bool_t makeOwner):TDataSet(name)
{
   SetTitle("TObjectSet");
   SetObject(obj,makeOwner);
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

TObjectSet::TObjectSet(TObject *obj,Bool_t makeOwner) : TDataSet("unknown","TObjectSet")
{
   SetObject(obj,makeOwner);
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

TObjectSet::~TObjectSet()
{
   if (fObj && IsOwner() && (TObject::TestBit(kNotDeleted))  ) delete fObj;
   fObj = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Aliase for SetObject method

TObject *TObjectSet::AddObject(TObject *obj,Bool_t makeOwner)
{
   return SetObject(obj,makeOwner);
}

////////////////////////////////////////////////////////////////////////////////
/// Browse this dataset (called by TBrowser).

void TObjectSet::Browse(TBrowser *b)
{
   if (b && fObj) b->Add(fObj);
   TDataSet::Browse(b);
}

////////////////////////////////////////////////////////////////////////////////
///to be documented

void TObjectSet::Delete(Option_t *opt)
{
   if (opt) {/* no used */}
   if (fObj && IsOwner()) delete fObj;
   fObj = 0;
   TDataSet::Delete();
}
////////////////////////////////////////////////////////////////////////////////
/// Set / Reset the ownerships and returns the previous
/// status of the ownerships.

Bool_t TObjectSet::DoOwner(Bool_t done)
{
   Bool_t own = IsOwner();
   if (own != done) {
      if (done) SetBit(kIsOwner);
      else ResetBit(kIsOwner);
   }
   return own;
}
////////////////////////////////////////////////////////////////////////////////
/// apply the class default ctor to instantiate a new object of the same kind.
/// This is a base method to be overriden by the classes
/// derived from TDataSet (to support TDataSetIter::Mkdir for example)

TDataSet *TObjectSet::Instance() const
{
   return instance();
}
////////////////////////////////////////////////////////////////////////////////
/// - Replace the embedded object with a new supplied
/// - Destroy the preivous embedded object if this is its owner
/// - Return the previous embedded object if any

TObject *TObjectSet::SetObject(TObject *obj,Bool_t makeOwner)
{
   TObject *oldObject = fObj;
   if (IsOwner()) { delete oldObject; oldObject = 0;} // the object has been killed
   fObj = obj;
   DoOwner(makeOwner);
   return oldObject;
}
