// @(#)root/tree:$Name:  $:$Id: TBranchElement.cxx,v 1.7 2001/02/06 11:02:00 brun Exp $
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
TBranchElement::TBranchElement(const char *bname, TStreamerInfo *sinfo, Int_t id, char *pointer, Int_t basketsize, Int_t splitlevel, Int_t compress)
    :TBranch()
{
// Create a BranchElement
//
// If splitlevel > 0 this branch in turn is split into sub branches
   
   char name[128];
   strcpy(name,bname);
//printf("Creating branch:%s, sinfo:%s\n",name,sinfo->GetName());
   TClass *cl    = sinfo->GetClass();
   fInfo         = sinfo;
   fID           = id;
   fStreamerType = -1;
   fType         = 0;
   fClassVersion = cl->GetClassVersion();
   if (id >= 0) {
     Int_t *types = sinfo->GetTypes();
     fStreamerType = types[fID];
   }
           
   SetName(name);
   SetTitle(name);
   fClassName = sinfo->GetName();
   fCompress = compress;
   if (compress == -1) {
      TFile *bfile = gTree->GetDirectory()->GetFile();
      if (bfile) fCompress = bfile->GetCompressionLevel();
   }
   if (basketsize < 100) basketsize = 100;
   fBasketSize     = basketsize;
   fBasketEntry    = new Int_t[fMaxBaskets];
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketSeek     = new Seek_t[fMaxBaskets];

   fBasketEntry[0] = fEntryNumber;
   fBasketBytes[0] = 0;

   // Set the bit kAutoDelete to specify that when reading
   // the object should be deleted before calling Streamer.
   // It is foreseen to not set this bit in a future version.
   SetAutoDelete(kTRUE);

   fTree       = gTree;
   fDirectory  = fTree->GetDirectory();
   fFileName   = "";
   
   // create sub branches if requested by splitlevel
   if (splitlevel > 0) {
      TClass *clm;
      TObjArray *elements = sinfo->GetElements();
      TStreamerElement *element = (TStreamerElement *)elements->At(id);
      if (element->IsA() == TStreamerBase::Class()) {
         // ===> develop the base class
         fType = 1;
         clm = gROOT->GetClass(element->GetName());
         if (!strcmp(name,clm->GetName())) Unroll("",cl,clm,basketsize,splitlevel,0);
         else                              Unroll(name,clm,clm,basketsize,splitlevel,0);
         return;
                  
      } else if (!strchr(element->GetTypeName(),'*') && (fStreamerType == TStreamerInfo::kObject || fStreamerType == TStreamerInfo::kAny)) {
         // ===> create sub branches for members that are classes
         fType = 2;
         clm = gROOT->GetClass(element->GetTypeName());
         Unroll(name,clm,clm,basketsize,splitlevel,0);
         fClassName = element->GetTypeName();
         return;
         
      } else if (!strcmp(element->GetTypeName(),"TClonesArray*")) {
         // ===> Create a leafcount
         TLeaf *leaf     = new TLeafElement(name,fID, fStreamerType);
         leaf->SetBranch(this);
         fNleaves = 1;
         fLeaves.Add(leaf);
         fTree->GetListOfLeaves()->Add(leaf);
         // Create a basket for the leafcount
         TBasket *basket = new TBasket(name,fTree->GetName(),this);
         fBaskets.Add(basket);
         // ===> create sub branches for each data member of a TClonesArray
         fType = 3;
         char **ppointer = (char**)(pointer);
         TClonesArray *clones = (TClonesArray*)(*ppointer);
         if (!clones) return; // TClonesArray must exist
         clm = clones->GetClass();
         if (!clm) return;
         char branchname[128];
         sprintf(branchname,"%s_",name);
         SetTitle(branchname);
         Unroll(name,clm,clm,basketsize,splitlevel,31);
         Int_t nbranches = fBranches.GetEntries();
         for (Int_t i=0;i<nbranches;i++) {
            TBranchElement *bre = (TBranchElement*)fBranches.At(i);
            const char *fin = strrchr(bre->GetTitle(),'.');
            if (fin == 0) continue;
            sprintf(branchname,"%s[%s_]",fin+1,name);
            bre->SetTitle(branchname);
         }
         return;
         
      } else if (strstr(element->GetTypeName(),"vector<")) {
         // ===> create sub branches for each data member of a STL vector
         //      if it is a vector of class objects.
         //      STL vectors like vector<float> are not split
         fType = 0;
         char classname[128];
         strcpy(classname,element->GetTypeName()+7);
         char *star = strchr(classname,'>');
         *star = 0;
         clm = gROOT->GetClass(classname);
         if (clm) {         
            TLeaf *leaf     = new TLeafElement(name,fID, fStreamerType);
            leaf->SetBranch(this);
            fNleaves = 1;
            fLeaves.Add(leaf);
            fTree->GetListOfLeaves()->Add(leaf);
            // Create a basket for the leafcount
            TBasket *basket = new TBasket(name,fTree->GetName(),this);
            fBaskets.Add(basket);
            // ===> create sub branches for each data member of the class
            fType = 4;
            char branchname[128];
            sprintf(branchname,"%s_",name);
            SetTitle(branchname);
            Unroll(name,clm,clm,basketsize,splitlevel,41);
            TStreamerSTL *stl = (TStreamerSTL*)element;
            if (stl->GetSTLtype() == 1 && stl->GetCtype() == 61) {
               printf("will split this STL vector: %s of %s\n",name,classname);
            }
            if (stl->GetSTLtype() == 41 && stl->GetCtype() == 61) {
               printf("will split this pointer to STL vector: %s of %s\n",name,classname);
            }
            return;
         }
      }
   }

   TLeaf *leaf     = new TLeafElement(name,fID, fStreamerType);
   leaf->SetBranch(this);
   fNleaves = 1;
   fLeaves.Add(leaf);
   gTree->GetListOfLeaves()->Add(leaf);

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
   if (!TestBit(kDoNotProcess)) nbytes += TBranch::Fill();
   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches) {
      for (Int_t i=0;i<nbranches;i++)  {
         TBranchElement *branch = (TBranchElement*)fBranches[i];
         if (!branch->TestBit(kDoNotProcess)) nbytes += branch->Fill();
      }
   }
   return nbytes;
}

//______________________________________________________________________________
void TBranchElement::FillLeaves(TBuffer &b)
{
//  Fill buffers of this branch
         
  if (fType == 4) {           // STL vector/list of objects
     //printf ("STL split mode not yet implemented\n");
  } else if (fType == 41) {   // sub branch of an STL class
    //char **ppointer = (char**)fAddress;
  } else if (fType == 3) {   //top level branch of a TClonesArray
    char **ppointer = (char**)fAddress;
    TClonesArray *clones = (TClonesArray*)(*ppointer);
    if (!clones) return; 
    Int_t n = clones->GetEntriesFast();
    b << n;
  } else if (fType == 31) {   // sub branch of a TClonesArray
    char **ppointer = (char**)fAddress;
    TClonesArray *clones = (TClonesArray*)(*ppointer);
    if (!clones) return; 
    Int_t n = clones->GetEntriesFast();
    fInfo->WriteBufferClones(b,clones,n,fID);
  } else if (fType == 0) {
     if (fID >= 0) {           // branch in split mode
        fInfo->WriteBuffer(b,fAddress,fID);
     } else if (fID == -1) {   // top level branch in non split mode
        char **ppointer = (char**)fAddress;
        fInfo->WriteBuffer(b,*ppointer,fID);
     }
  }   
}

//______________________________________________________________________________
Int_t TBranchElement::GetEntry(Int_t entry, Int_t getall)
{
//*-*-*-*-*Read all branches of a BranchElement and return total number of bytes
//*-*      ====================================================================
//   If entry = 0 take current entry number + 1
//   If entry < 0 reset entry number to 0

   if (TestBit(kDoNotProcess) && !getall) return 0;
   Int_t nbranches = fBranches.GetEntriesFast();
   
   Int_t nbytes = TBranch::GetEntry(entry);

   if (nbranches) {
      if (fAddress == 0) { // try to create object
         if (!TestBit(kWarn)) {
            TClass *cl = gROOT->GetClass(fClassName.Data());
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
      for (Int_t i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches[i];
         nbytes += branch->GetEntry(entry);
      }
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

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches) {
      if (fID == -2) {
         Printf("*Branch  :%-9s : %-54s *",GetName(),GetTitle());
         Printf("*Entries : %8d : BranchElement (see below)                               *",Int_t(fEntries));
         Printf("*............................................................................*");
      } 
      if (fType > 2) {
         TBranch::Print(option);
      }
      for (Int_t i=0;i<nbranches;i++)  {
         TBranch *branch = (TBranch*)fBranches.At(i);
         branch->Print(option);
      }
   } else {
      TBranch::Print(option);
   }
}

//______________________________________________________________________________
void TBranchElement::ReadLeaves(TBuffer &b)
{
// Read buffers for this branch
         
  if (fType == 4) {           // STL vector/list of objects
     //printf ("STL split mode not yet implemented\n");
  } else if (fType == 41) {    // sub branch of an STL class
    //char **ppointer = (char**)fAddress;
  } else if (fType == 3) {    //top level branch of a TClonesArray
    char **ppointer = (char**)fAddress;
    TClonesArray *clones = (TClonesArray*)(*ppointer);
    if (!clones) return; 
    Int_t n;
    b >> n;
    clones->Clear();
    clones->ExpandCreateFast(n);
  } else if (fType == 31) {    // sub branch of a TClonesArray
    char **ppointer = (char**)fAddress;
    TClonesArray *clones = (TClonesArray*)(*ppointer);
    if (!clones) return; 
    Int_t n = clones->GetEntriesFast();
    fInfo->ReadBufferClones(b,clones,n,fID);
  } else if (fType == 0) {     // branch in split mode
     if (fID >= 0) {
        fInfo->ReadBuffer(b,fAddress,fID);
     } else if (fID == -1) {   // top level branch in non split mode
        char **ppointer = (char**)fAddress;
        fInfo->ReadBuffer(b,*ppointer,fID);
     }
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

   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i=0;i<nbranches;i++)  {
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
   
   Int_t nbranches = fBranches.GetEntriesFast();
//printf("SetAddress, branch:%s, classname=%s, fID=%d, fType=%d, nbranches=%d, add=%x, fInfo=%x\n",GetName(),fClassName.Data(),fID,fType,nbranches,add,fInfo);
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
   
   //build the StreamerInfo if first time for the class
   TStreamerInfo::Optimize(kFALSE);
   if (!fInfo ) fInfo = fTree->BuildStreamerInfo(cl);
   Int_t *leafOffsets = fInfo->GetOffsets();
      
   for (Int_t i=0;i<nbranches;i++)  {
      TBranchElement *branch = (TBranchElement*)fBranches[i];
      Int_t nb2 = branch->GetListOfBranches()->GetEntries();
      Int_t id = branch->GetID();
      Int_t baseOffset = 0;
      TClass *clm = gROOT->GetClass(branch->GetClassName());
      if ((clm != cl) && (branch->GetType() == 0)) {
         if (cl->GetBaseClass(clm)) {
            baseOffset = cl->GetBaseClassOffset(clm);
            if (baseOffset < 0) baseOffset = 0;
         }
      }
      if (nb2 > 0) {
         branch->SetAddress(objadd + leafOffsets[id] + baseOffset);
      } else {
         branch->SetAddress(objadd + baseOffset);
      }
   }
      
   TStreamerInfo::Optimize(kTRUE);
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
   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i=0;i<nbranches;i++)  {
      TBranch *branch = (TBranch*)fBranches[i];
      branch->SetBasketSize(buffsize);
   }
}

//______________________________________________________________________________
Int_t TBranchElement::Unroll(const char *name, TClass *cltop, TClass *cl,Int_t basketsize, Int_t splitlevel, Int_t btype)
{
// unroll base classes and loop on all elements of class cl

   if (cl == TObject::Class() && cltop->CanIgnoreTObjectStreamer()) return 0;
   TStreamerInfo *info = cl->GetStreamerInfo();
   Int_t ndata = info->GetNdata();
   ULong_t *elems = info->GetElems();
   TIter next(info->GetElements());
   TStreamerElement *elem;
   Int_t jd = 0;
//printf("unroll, name=%s, ndata=%d\n",name,ndata);
   for (Int_t i=0;i<ndata;i++) {
      elem = (TStreamerElement*)elems[i];
      if (elem->IsA() == TStreamerBase::Class()) {
         TClass *clbase = gROOT->GetClass(elem->GetName());
         if (clbase->Property() & kIsAbstract) {
            jd = -1;
            break;
         }
         Unroll(name,cltop,clbase,basketsize,splitlevel-1,btype);
      } else {
//printf("elemgetname=%s\n",elem->GetName());
         char branchname[128];
         if (strlen(name)) sprintf(branchname,"%s.%s",name,elem->GetFullName());
         else              sprintf(branchname,"%s",elem->GetFullName());
         TBranchElement *branch = new TBranchElement(branchname,info,jd,0,basketsize,0);
         branch->SetType(btype);
         fBranches.Add(branch);
      }
      jd++;
   }
   return 1;
}
