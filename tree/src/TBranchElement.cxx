// @(#)root/tree:$Name:  $:$Id: TBranchElement.cxx,v 1.5 2001/01/18 09:45:31 brun Exp $
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
// TBranchElement                                                       //
//                                                                      //
// A Branch for the case of an object                                   //                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TFile.h"
#include "TBranchElement.h"
#include "TClonesArray.h"
#include "TTree.h"
#include "TBasket.h"
#include "TLeafElement.h"
#include "TVirtualPad.h"
#include "TClass.h"
#include "TRealData.h"
#include "TDataType.h"
#include "TDataMember.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TBrowser.h"

R__EXTERN  TTree *gTree;

ClassImp(TBranchElement)

//______________________________________________________________________________
TBranchElement::TBranchElement(): TBranch()
{
//*-*-*-*-*-*Default constructor for BranchElement*-*-*-*-*-*-*-*-*-*
//*-*        ====================================

   fNleaves   = 1;
   fInfo = 0;
}


//______________________________________________________________________________
TBranchElement::TBranchElement(const char *name, TStreamerInfo *sinfo, Int_t id, char *pointer, Int_t basketsize, Int_t splitlevel, Int_t compress)
    :TBranch()
{
// Create a BranchElement
//
// If splitlevel > 0 this branch in turn is split into sub branches
   
   TClass *cl    = sinfo->GetClass();
   fInfo         = sinfo;
   fID           = id;
   fType         = -1;
   fCounter      = 0;
   fClassVersion = cl->GetClassVersion();
   if (id >= 0) {
     Int_t *types = sinfo->GetTypes();
     fType = types[fID];
   }
           
   SetName(name);
   SetTitle(name);
   fCompress = compress;
   if (compress == -1) {
      TFile *bfile = gTree->GetDirectory()->GetFile();
      if (bfile) fCompress = bfile->GetCompressionLevel();
   }
   if (basketsize < 100) basketsize = 100;
   fBasketSize     = basketsize;
   //fAddress        = (char*)addobj;
   fClassName      = cl->GetName();
   fBasketEntry    = new Int_t[fMaxBaskets];
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketSeek     = new Seek_t[fMaxBaskets];

   fBasketEntry[0] = fEntryNumber;
   fBasketBytes[0] = 0;

   TLeaf *leaf     = new TLeafElement(name,fID, fType);
   leaf->SetBranch(this);
   fNleaves = 1;
   fLeaves.Add(leaf);
   gTree->GetListOfLeaves()->Add(leaf);

   // Set the bit kAutoDelete to specify that when reading
   // in TLeafElement::ReadBasket, the object should be deleted
   // before calling Streamer.
   // It is foreseen to not set this bit in a future version.
   SetAutoDelete(kTRUE);

   fTree       = gTree;
   fDirectory  = fTree->GetDirectory();
   fFileName   = "";
   
   // create sub branches if requested by splitlevel
   TClass *clm;
   Int_t unroll = 0;
   if (splitlevel > 0) {
      TObjArray *elements = sinfo->GetElements();
      TStreamerElement *element = (TStreamerElement *)elements->At(id);
      if (element->IsA() == TStreamerBase::Class()) {
         // ===> develop the base class
         fCounter = 1;
         clm = gROOT->GetClass(element->GetName());
         if (!strcmp(name,clm->GetName())) unroll = Unroll("",cl,clm,basketsize,splitlevel);
         else                              unroll = Unroll(name,clm,clm,basketsize,splitlevel);
         
      } else if (!strchr(element->GetTypeName(),'*') && (fType == TStreamerInfo::kObject || fType == TStreamerInfo::kAny)) {
         // ===> create sub branches for members that are classes
         fCounter = 2;
         clm = gROOT->GetClass(element->GetTypeName());
         Unroll(name,clm,clm,basketsize,splitlevel);
         
      } else if (!strcmp(element->GetTypeName(),"TClonesArray*")) {
         // ===> create sub branches for each data member of a TClonesArray
         fCounter = 3;
         char **ppointer = (char**)(pointer);
         TClonesArray *clones = (TClonesArray*)(*ppointer);
         if (!clones) return; // TClonesArray must exist
         clm = clones->GetClass();
         if (!clm) return;
         printf("will split this clonesarray: %s of %s\n",name,clm->GetName());
         char branchname[128];
         sprintf(branchname,"%s_",name);
         SetTitle(branchname);
         fCounter = 1;
         Unroll(name,clm,clm,basketsize,splitlevel);
         
      } else if (strstr(element->GetTypeName(),"vector<")) {
         // ===> create sub branches for each data member of a STL vector
         //      if it is a vector of class objects.
         //      STL vectors like vector<float> are not split
         fCounter = 4;
         char classname[128];
         strcpy(classname,element->GetTypeName()+7);
         char *star = strchr(classname,'>');
         *star = 0;
         clm = gROOT->GetClass(classname);
         if (!clm) return; // may be a basic type
         
         fCounter = 1;
         TStreamerSTL *stl = (TStreamerSTL*)element;
         if (stl->GetSTLtype() == 1 && stl->GetCtype() == 61) {
            printf("will split this STL vector: %s of %s\n",name,classname);
         }
         if (stl->GetSTLtype() == 41 && stl->GetCtype() == 61) {
            printf("will split this pointer to STL vector: %s of %s\n",name,classname);
         }
      //return;
      }
   }

   // Create a basket for the terminal branch
   TBasket *basket = new TBasket(name,fTree->GetName(),this);
   fBaskets.Add(basket);
}

//______________________________________________________________________________
TBranchElement::~TBranchElement()
{
//*-*-*-*-*-*Default destructor for a BranchElement*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =====================================

   fBranches.Delete();
}


//______________________________________________________________________________
void TBranchElement::Browse(TBrowser *b)
{

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches > 1) {
      fBranches.Browse( b );
   }
}

//______________________________________________________________________________
Int_t TBranchElement::Fill()
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
Int_t TBranchElement::GetEntry(Int_t entry, Int_t getall)
{
//*-*-*-*-*Read all branches of a BranchElement and return total number of bytes
//*-*      ====================================================================
//   If entry = 0 take current entry number + 1
//   If entry < 0 reset entry number to 0

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
Bool_t TBranchElement::IsFolder() const
{
//*-*-*-*-*Return TRUE if more than one leaf, FALSE otherwise*-*
//*-*      ==================================================

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches >= 1) return kTRUE;
   else                return kFALSE;
}

//______________________________________________________________________________
void TBranchElement::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print TBranch parameters*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches) {
      if (fID == -1) {
         Printf("*Branch  :%-9s : %-54s *",GetName(),GetTitle());
         Printf("*Entries : %8d : BranchElement (see below)                               *",Int_t(fEntries));
         Printf("*............................................................................*");
      } 
      if (fCounter) {
         TBranch::Print(option);
      }
      for (i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches.At(i);
         branch->Print(option);
      }
   } else {
      TBranch::Print(option);
   }
}

//______________________________________________________________________________
void TBranchElement::Reset(Option_t *option)
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
void TBranchElement::SetAddress(void *add)
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
   
   TClass *cl = gROOT->GetClass(fClassName.Data());
   fAddress = (char*)add;
   char *objadd = (char*)add;
   if (fID < 0) {
      char **ppointer = (char**)add;
      objadd = *ppointer;
      if (!objadd && cl) {
         objadd = (char*)cl->New();
         *ppointer = objadd;
      }
   }
   Int_t *leafOffsets;
   if (!fInfo) {
      TStreamerInfo::Optimize(kFALSE);
      cl->BuildRealData(objadd);
      fInfo = cl->GetStreamerInfo(fClassVersion);
      leafOffsets = fInfo->GetOffsets();
      if (!leafOffsets || fInfo->IsOptimized()) fInfo->BuildOld();
   }
   leafOffsets = fInfo->GetOffsets();
      
   TLeaf *leaf = (TLeaf*)fLeaves.UncheckedAt(0);
   if (leaf) {
      if (fID >= 0) {
         leafOffsets = fInfo->GetOffsets();
         leaf->SetAddress(objadd + leafOffsets[fID]);
      }
   }

   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i=0;i<nbranches;i++)  {
      TBranchElement *branch = (TBranchElement*)fBranches[i];
      Int_t nb2 = branch->GetListOfBranches()->GetEntries();
      if (nb2 > 0) {
         Int_t id = branch->GetID();
         branch->SetAddress(objadd + leafOffsets[id]);
      } else {
         branch->SetAddress(objadd);
      }
   }
}

//______________________________________________________________________________
void TBranchElement::SetAutoDelete(Bool_t autodel)
{
//*-*-*-*-*-*-*-*Set the AutoDelete bit
//*-*            ====================
//  This function can be used to instruct Root in TBranchElement::ReadBasket
//  to not delete the object referenced by a branchelement before reading a
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
}

//______________________________________________________________________________
void TBranchElement::SetBasketSize(Int_t buffsize)
{
// Reset basket size for all subbranches of this branchelement

   fBasketSize = buffsize;
   Int_t i;
   Int_t nbranches = fBranches.GetEntriesFast();
   for (i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->SetBasketSize(buffsize);
   }
}

//______________________________________________________________________________
Int_t TBranchElement::Unroll(const char *name, TClass *cltop, TClass *cl,Int_t basketsize, Int_t splitlevel)
{
// unroll base classes and loop on all elements of class cl

   if (cl == TObject::Class() && cltop->CanIgnoreTObjectStreamer()) return 0;
   TStreamerInfo *info = cl->GetStreamerInfo();
   TIter next(info->GetElements());
   TStreamerElement *elem;
   Int_t jd = 0;
   while ((elem = (TStreamerElement*)next())) {
      if (elem->IsA() == TStreamerBase::Class()) {
         TClass *clbase = gROOT->GetClass(elem->GetName());
         if (clbase->Property() & kIsAbstract) {
            jd = -1;
            break;
         }
         Unroll(name,cltop,clbase,basketsize,splitlevel-1);
      } else {
         char branchname[128];
         if (strlen(name)) sprintf(branchname,"%s.%s",name,elem->GetName());
         else              sprintf(branchname,"%s",elem->GetName());
         fBranches.Add(new TBranchElement(branchname,info,jd,0,basketsize,splitlevel-1));
      }
      jd++;
   }
   return 1;
}
