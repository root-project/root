// @(#)root/tree:$Name:  $:$Id: TBranchObject.cxx,v 1.11 2001/02/20 08:15:10 brun Exp $
// Author: Rene Brun   11/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchObject                                                        //
//                                                                      //
// A Branch for the case of an object                                   //                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TFile.h"
#include "TBranchObject.h"
#include "TBranchClones.h"
#include "TTree.h"
#include "TBasket.h"
#include "TLeafObject.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TRealData.h"
#include "TDataType.h"
#include "TDataMember.h"
#include "TBrowser.h"

R__EXTERN  TTree *gTree;

ClassImp(TBranchObject)

//______________________________________________________________________________
TBranchObject::TBranchObject(): TBranch()
{
//*-*-*-*-*-*Default constructor for BranchObject*-*-*-*-*-*-*-*-*-*
//*-*        ====================================

   fNleaves   = 1;
   fOldObject = 0;
}


//______________________________________________________________________________
TBranchObject::TBranchObject(const char *name, const char *classname, void *addobj, Int_t basketsize, Int_t splitlevel, Int_t compress)
    :TBranch()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a BranchObject*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =====================
//
   TClass *cl      = gROOT->GetClass(classname);
   if (!cl) {
      Error("TBranchObject","Cannot find class:%s",classname);
      return;
   }
   char **apointer = (char**)(addobj);
   TObject *obj = (TObject*)(*apointer);
   Bool_t delobj = kFALSE;
   if (!obj) {
      obj = (TObject*)cl->New();
      delobj = kTRUE;
   }
   gTree->BuildStreamerInfo(cl,obj);
   if (delobj) delete obj;
   
   SetName(name);
   SetTitle(name);
   fCompress = compress;
   if (compress == -1) {
      TFile *bfile = gTree->GetDirectory()->GetFile();
      if (bfile) fCompress = bfile->GetCompressionLevel();
   }
   if (basketsize < 100) basketsize = 100;
   fBasketSize     = basketsize;
   fAddress        = (char*)addobj;
   fClassName      = classname;
   fBasketEntry    = new Int_t[fMaxBaskets];
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketSeek     = new Seek_t[fMaxBaskets];
   fOldObject      = 0;

   fBasketEntry[0] = fEntryNumber;
   fBasketBytes[0] = 0;

   TLeaf *leaf     = new TLeafObject(name,classname);
   leaf->SetBranch(this);
   leaf->SetAddress(addobj);
   fNleaves = 1;
   fLeaves.Add(leaf);
   gTree->GetListOfLeaves()->Add(leaf);

// Set the bit kAutoDelete to specify that when reading
// in TLeafObject::ReadBasket, the object should be deleted
// before calling Streamer.
// It is foreseen to not set this bit in a future version.
   SetAutoDelete(kTRUE);

   fTree       = gTree;
   fDirectory  = fTree->GetDirectory();
   fFileName   = "";

//*-*-  Create the first basket
   if (splitlevel) return;
   TBasket *basket = new TBasket(name,fTree->GetName(),this);
   fBaskets.Add(basket);
}


//______________________________________________________________________________
TBranchObject::~TBranchObject()
{
//*-*-*-*-*-*Default destructor for a BranchObject*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =====================================

   fBranches.Delete();
}


//______________________________________________________________________________
void TBranchObject::Browse(TBrowser *b)
{

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches > 1) {
      fBranches.Browse( b );
   }
}

//______________________________________________________________________________
Int_t TBranchObject::Fill()
{
//*-*-*-*-*-*-*-*Loop on all leaves of this branch to fill Basket buffer*-*-*
//*-*            =======================================================

   Int_t nbytes = 0;
   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches) {
      fEntries++;
      UpdateAddress();
      for (i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches[i];
         if (!branch->TestBit(kDoNotProcess)) nbytes += branch->Fill();
      }
   } else {
      if (!TestBit(kDoNotProcess)) nbytes += TBranch::Fill();
   }
   return nbytes;
}

//______________________________________________________________________________
Int_t TBranchObject::GetEntry(Int_t entry, Int_t getall)
{
//*-*-*-*-*Read all branches of a BranchObject and return total number of bytes
//*-*      ====================================================================
//   If entry = 0 take current entry number + 1
//   If entry < 0 reset entry number to 0
//
//  The function returns the number of bytes read from the input buffer.
//  If entry does not exist or an I/O error occurs, the function returns 0.
//  if entry is the same as the previous call, the function returns 1.

   if (TestBit(kDoNotProcess) && !getall) return 0;
   Int_t nbytes;
   Int_t nbranches = fBranches.GetEntriesFast();

   if (nbranches) {
      if (fAddress == 0) { // try to create object
         if (!TestBit(kWarn)) {
            TClass *cl = gROOT->GetClass(fClassName);
            if (cl) {
               TObject** voidobj = (TObject**) new Long_t[1];
               *voidobj = (TObject*)cl->New();
               SetAddress(voidobj);
            } else {
               Warning("GetEntry","Cannot get class: %s",fClassName.Data());
               SetBit(kWarn);
            }
         }
      }
      nbytes = 0;
      for (Int_t i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches[i];
            nbytes += branch->GetEntry(entry);
      }
   } else {
      nbytes = TBranch::GetEntry(entry);
   }
   return nbytes;
}


//______________________________________________________________________________
Bool_t TBranchObject::IsFolder() const
{
//*-*-*-*-*Return TRUE if more than one leaf, FALSE otherwise*-*
//*-*      ==================================================

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches >= 1) return kTRUE;
   else                return kFALSE;
}

//______________________________________________________________________________
void TBranchObject::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print TBranch parameters*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches) {
      Printf("*Branch  :%-9s : %-54s *",GetName(),GetTitle());
      Printf("*Entries : %8d : BranchObject (see below)                               *",Int_t(fEntries));
      Printf("*............................................................................*");
      for (i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches.At(i);
         branch->Print(option);
      }
   } else {
      TBranch::Print(option);
   }
}

//______________________________________________________________________________
void TBranchObject::Reset(Option_t *option)
{
//*-*-*-*-*-*-*-*Reset a Branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//
//    Existing buffers are deleted
//    Entries, max and min are reset
//

   TBranch::Reset(option);

   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->Reset(option);
   }
}

//______________________________________________________________________________
void TBranchObject::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*Set address of this branch*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ====================
//

   //special case when called from code generated by TTree::MakeClass
   if (Long_t(add) == -1) {
      SetBit(kWarn);
      return;
   }
   fReadEntry = -1; 
   Int_t nbranches = fBranches.GetEntriesFast();
   TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(0);
   if (leaf) leaf->SetAddress(add);
   TBranch *branch;
   fAddress = (char*)add;
   char *pointer   = fAddress;
   void **ppointer = (void**)add;
   TObject *obj = 0;
   if (add) obj = (TObject*)(*ppointer);
   TClass *cl = gROOT->GetClass(fClassName.Data());
   if (!obj && cl) {
      obj = (TObject*)cl->New();
      *ppointer = (void*)obj;
   }
   //fOldObject = obj;
   Int_t i, offset;
   if (!cl) {
      for (i=0;i<nbranches;i++)  {
         branch  = (TBranch*)fBranches[i];
         pointer = (char*)obj;
         branch->SetAddress(pointer);
      }
      return;
   }
   if (!cl->GetListOfRealData())  cl->BuildRealData(obj);
   char *fullname = new char[200];
   const char *bname = GetName();
   Int_t lenName = strlen(bname);
   Int_t isDot = 0;
   if (bname[lenName-1] == '.') isDot = 1;
   const char *rdname;
   TRealData *rd;
   TIter      next(cl->GetListOfRealData());
   while ((rd = (TRealData *) next())) {
      TDataMember *dm = rd->GetDataMember();
      if (!dm->IsPersistent()) continue;
      rdname = rd->GetName();
      TDataType *dtype = dm->GetDataType();
      Int_t code = 0;
      if (dtype) code = dm->GetDataType()->GetType();
      offset  = rd->GetThisOffset();
      pointer = (char*)obj + offset;
      branch  = 0;
      if (dm->IsaPointer()) {
         TClass *clobj = 0;
         if (!dm->IsBasic()) clobj = gROOT->GetClass(dm->GetTypeName());
         if (clobj && clobj->InheritsFrom("TClonesArray")) {
            if (isDot) sprintf(fullname,"%s%s",bname,&rdname[1]);
            else       sprintf(fullname,"%s",&rdname[1]);
            branch = (TBranch*)fBranches.FindObject(fullname);
         } else {
            if (!clobj) {
               // this is a basic type we can handle only if
               // he has a dimension:
               const char * index = dm->GetArrayIndex();
               if (strlen(index)==0) {
                  if (code==1) {
                     // Case of a string ... we do not need the size
                     if (isDot) sprintf(fullname,"%s%s",bname,&rdname[0]);
                     else       sprintf(fullname,"%s",&rdname[0]);
                  } else {
                     continue;
                  }
               }
               if (isDot) sprintf(fullname,"%s%s",bname,&rdname[0]);
               else       sprintf(fullname,"%s",&rdname[0]);
               // let's remove the stars!
               UInt_t cursor,pos;
               for( cursor = 0, pos = 0;
                    cursor < strlen(fullname);
                    cursor ++ ) {
                  if (fullname[cursor]!='*') {
                     fullname[pos++] = fullname[cursor];
                  };
               };
               fullname[pos] = '\0';
               branch = (TBranch*)fBranches.FindObject(fullname);		 
            } else {
               if (!clobj->InheritsFrom(TObject::Class())) continue;
               if (isDot) sprintf(fullname,"%s%s",bname,&rdname[1]);
               else       sprintf(fullname,"%s",&rdname[1]);
               branch = (TBranch*)fBranches.FindObject(fullname);
            }
         }
      } else {
         if (dm->IsBasic()) {
            if (isDot) sprintf(fullname,"%s%s",bname,&rdname[0]);
            else       sprintf(fullname,"%s",&rdname[0]);
            branch = (TBranch*)fBranches.FindObject(fullname);
         }
      }
      if(branch) branch->SetAddress(pointer);
   }
   delete [] fullname;
}

//______________________________________________________________________________
void TBranchObject::SetAutoDelete(Bool_t autodel)
{
//*-*-*-*-*-*-*-*Set the AutoDelete bit
//*-*            ====================
//  This function can be used to instruct Root in TBranchObject::ReadBasket
//  to not delete the object referenced by a branchobject before reading a
//  new entry. By default, the object is deleted.
//  If autodel is kTRUE, this existing object will be deleted, a new object
//    created by the default constructor, then object->Streamer called.
//  If autodel is kFALSE, the existing object is not deleted. Root assumes
//    that the user is taking care of deleting any internal object or array
//    This can be done in Streamer itself.
//  If this branch has sub-branches, the function sets autodel for these
//  branches as well.
//  We STRONGLY suggest to activate this option by default when you create
//  the top level branch. This will make the read phase more efficient
//  because it minimizes the numbers of new/delete operations.
//  Once this option has been set and the Tree is written to a file, it is
//  not necessary to specify the option again when reading, unless you
//  want to set the opposite mode.
//

   TBranch::SetAutoDelete(autodel);

   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->SetAutoDelete(autodel);
   }
}

//______________________________________________________________________________
void TBranchObject::SetBasketSize(Int_t buffsize)
{
//*-*-*-*-*-*-*-*Reset basket size for all subbranches of this branchobject
//*-*            ==========================================================
//

   fBasketSize = buffsize;
   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->SetBasketSize(buffsize);
   }
}

//______________________________________________________________________________
void TBranchObject::UpdateAddress()
{
//*-*-*-*-*-*-*-*Update branch addresses if a new object was created*-*-*
//*-*            ===================================================
//

   void **ppointer = (void**)fAddress;
   if (ppointer == 0) return;
   TObject *obj = (TObject*)(*ppointer);
   if (obj != fOldObject) {
      fOldObject = obj;
      SetAddress(fAddress);
   }
}
