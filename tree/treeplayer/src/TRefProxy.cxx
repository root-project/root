// @(#)root/meta:$Id$
// Author: Markus Frank  10/02/2006

/*************************************************************************
* Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "TRef.h"
#include "TTree.h"
#include "TError.h"
#include "TBranch.h"
#include "TRefProxy.h"
#include "TBranchRef.h"
#include "TRefTable.h"
#include "TTreeFormula.h"
#include "TFormLeafInfoReference.h"
#include <iostream>

//______________________________________________________________________________
//
// TRefProxy is a reference proxy, which allows to access ROOT references (TRef)
// stored contained in other objects from TTree::Draw


//______________________________________________________________________________
Bool_t TRefProxy::Update()
{
   // TVirtualRefProxy overload: Update (and propagate) cached information

   return kTRUE;
}

//______________________________________________________________________________
TClass* TRefProxy::GetValueClass(void* data) const
{
   // TVirtualRefProxy overload: Access to value class

   TObject* obj = (TObject*)data;
   return ( obj ) ? obj->IsA() : 0;
}

//______________________________________________________________________________
void* TRefProxy::GetObject(TFormLeafInfoReference* info, void* data, int)
{
   // Access referenced object(-data)

   if ( data )  {
      TRef*      ref    = (TRef*)((char*)data + info->GetOffset());
      void* obj = ref->GetObject();
      // std::cout << "UID=" << ref->GetUniqueID() << std::endl;
      if ( obj )  {
         return obj;
      }
      else  {
         TBranch*   branch = info->GetBranch();
         if ( branch )  {
            UInt_t     uid    = ref->GetUniqueID();
            Long64_t   ent    = branch->GetReadEntry();
            TRefTable *table  = TRefTable::GetRefTable();
            table->SetUID(uid, ref->GetPID());
            ((TBranch*)table->GetOwner())->GetEntry(ent);
            TBranch *b = (TBranch*)table->GetParent(uid, ref->GetPID());
            if ( 0 == b ) {
               ((TBranch*)table->GetOwner())->GetEntry(ent);
               b = (TBranch*)table->GetParent(uid, ref->GetPID());
            }
            if ( 0 != b )  {
               TBranch* br = b->GetMother();
               if ( br ) br->GetEntry(ent);
            }
            obj = ref->GetObject();
            if ( obj )   {
               (*ref) = 0;
               return obj;
            }
         }
      }
   }
   return 0;
}
