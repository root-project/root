// @(#)root/star:$Name:  $:$Id: TObjectSet.cxx,v 1.2 2001/01/14 01:27:02 fine Exp $
// Author: Valery Fine(fine@bnl.gov)   25/12/98
// $Id: TObjectSet.cxx,v 1.2 2001/01/14 01:27:02 fine Exp $
// $Log: TObjectSet.cxx,v $
// Revision 1.2  2001/01/14 01:27:02  fine
// New implementation TTable::SavePrimitive and AsString
//
// Revision 1.1.1.1  2000/11/27 22:57:14  fisyak
//
//
// Revision 1.1.1.1  2000/05/16 17:00:48  rdm
// Initial import of ROOT into CVS
//
// Revision 1.7  1999/05/07 21:35:32  fine
// Fix some first implemenation bugs
//
// Revision 1.6  1999/05/07 17:53:18  fine
// owner bit has been introduced to deal with the embedded objects
//
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

//_____________________________________________________________________________
TObjectSet::TObjectSet(const Char_t *name, TObject *obj, Bool_t makeOwner):TDataSet(name)
{
  SetTitle("TObjectSet");
  SetObject(obj,makeOwner);
}

//_____________________________________________________________________________
TObjectSet::TObjectSet(TObject *obj,Bool_t makeOwner) : TDataSet("unknown","TObjectSet")
{
  SetObject(obj,makeOwner);
}

//_____________________________________________________________________________
TObjectSet::~TObjectSet()
{
// Attn.: virtual  TObject::Delete will be called via virtual dtor of TDataSet
}

//_____________________________________________________________________________
void TObjectSet::Delete(Option_t *opt)
{
   if (opt) {/* no used */}
   if (fObj && IsOwner()) delete fObj;
   fObj = 0;
   TDataSet::Delete();
}

//______________________________________________________________________________
void TObjectSet::Browse(TBrowser *b)
{
  // Browse this dataset (called by TBrowser).
   if (b && fObj) b->Add(fObj);
  TDataSet::Browse(b);
}

//______________________________________________________________________________
TObject *TObjectSet::SetObject(TObject *obj,Bool_t makeOwner)
{
  //
  // - Replace the embedded object with a new supplied
  // - Destroy the preivous embedded object if this is its owner
  // - Return the previous embedded object if any
  //
   TObject *oldObject = fObj;
   if (IsOwner()) { delete oldObject; oldObject = 0;} // the object has been killed
   fObj = obj;
   DoOwner(makeOwner);
   return oldObject;
}

//______________________________________________________________________________
TObject *TObjectSet::AddObject(TObject *obj,Bool_t makeOwner)
{
  // Aliase for SetObject method
 return SetObject(obj,makeOwner);
}

//______________________________________________________________________________
Bool_t TObjectSet::DoOwner(Bool_t done)
{
 // Set / Reset the ownerships and returns the previous
 // status of the ownerships.

  Bool_t own = IsOwner();
  if (own != done) {
    if (done) SetBit(kIsOwner);
    else ResetBit(kIsOwner);
  }
  return own;
}
