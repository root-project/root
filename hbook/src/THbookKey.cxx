// @(#)root/hbook:$Name:  $:$Id: THbookKey.cxx,v 1.1 2002/02/18 18:02:57 rdm Exp $
// Author: Rene Brun   20/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THbookKey.h"
#include "THbookTree.h"
#include "TBrowser.h"

ClassImp(THbookKey)


//______________________________________________________________________________
THbookKey::THbookKey(Int_t id, THbookFile *file)
{
   fDirectory = file;
   fID = id;
   char name[10];
   sprintf(name,"h%d",id);
   SetName(name);
}


//______________________________________________________________________________
THbookKey::~THbookKey()
{
}

//______________________________________________________________________________
void THbookKey::Browse(TBrowser *b)
{
   // Read object from disk and call its Browse() method.
   // If object with same name already exist in memory delete it (like
   // TDirectory::Get() is doing), except when the key references a
   // folder in which case we don't want to re-read the folder object
   // since it might contain new objects not yet saved.

   fDirectory->cd();
   
   TObject *obj = fDirectory->GetList()->FindObject(GetName());
   if (obj && !obj->IsFolder()) {
      if (obj->InheritsFrom(TCollection::Class()))
         obj->Delete();   // delete also collection elements
      delete obj;
      obj = 0;
   }

   if (!obj)
      obj = fDirectory->Get(fID);

   if (b && obj) {
      obj->Browse(b);
      b->SetRefreshFlag(kTRUE);
   }
}

//______________________________________________________________________________
Bool_t THbookKey::IsFolder() const
{
    Bool_t ret = kFALSE;


    return( ret );
}
