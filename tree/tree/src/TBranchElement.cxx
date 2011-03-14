// @(#)root/tree:$Id$
// Authors Rene Brun , Philippe Canal, Markus Frank  14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchElement                                                       //
//                                                                      //
// A Branch for the case of an object                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBranchElement.h"

#include "TBasket.h"
#include "TBranchObject.h"
#include "TBranchRef.h"
#include "TBrowser.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClonesArray.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TError.h"
#include "TMath.h"
#include "TFile.h"
#include "TFolder.h"
#include "TLeafElement.h"
#include "TRealData.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TTree.h"
#include "TVirtualCollectionProxy.h"
#include "TVirtualCollectionIterators.h"
#include "TVirtualPad.h"
#include "TBranchSTL.h"
#include "TVirtualArray.h"
#include "TBufferFile.h"
#include "TInterpreter.h"
#include "TROOT.h"

#include "TStreamerInfoActions.h"

ClassImp(TBranchElement)

//______________________________________________________________________________
namespace {
   void RemovePrefix(TString& str, const char* prefix) {
      // -- Remove a prefix from a string.
      if (str.Length() && prefix && strlen(prefix)) {
         if (!str.Index(prefix)) {
            str.Remove(0, strlen(prefix));
         }
      }
   }
   struct R__PushCache {
      TBufferFile &fBuffer;
      TVirtualArray *fOnfileObject;

      R__PushCache(TBufferFile &b, TVirtualArray *in) : fBuffer(b), fOnfileObject(in) {
         if (fOnfileObject) fBuffer.PushDataCache( fOnfileObject );
      }
      ~R__PushCache() {
         if (fOnfileObject) fBuffer.PopDataCache();
      }
   };
}

//______________________________________________________________________________
void TBranchElement::SwitchContainer(TObjArray* branches) {
   // -- Modify the container type of the branches

   const Int_t nbranches = branches->GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i) {
      TBranchElement* br = (TBranchElement*) branches->At(i);
      switch (br->GetType()) {
         case 31: br->SetType(41); break;
         case 41: {
            br->SetType(31);
            br->fCollProxy = 0;
            break;
         }
      }
      br->SetReadLeavesPtr();
      // Note: This is a tail recursion.
      SwitchContainer(br->GetListOfBranches());
   }
}

//______________________________________________________________________________
namespace {
   Bool_t CanSelfReference(TClass *cl) {
      if (cl) {
         if (cl->GetCollectionProxy()) {
            TClass *inside = cl->GetCollectionProxy()->GetValueClass();
            if (inside) {
               return CanSelfReference(inside);
            } else {
               return kFALSE;
            }
         }
         static TClassRef stringClass("std::string");
         if (cl == stringClass || cl == TString::Class()) {
            return kFALSE;
         }
         // Here we could scan through the TStreamerInfo to see if there
         // is any pointer anywhere and know whether this is a possibility
         // of selfreference (but watch out for very indirect cases).
         return kTRUE;
      }
      return kFALSE;
   }
}

//______________________________________________________________________________
TBranchElement::TBranchElement()
: TBranch()
, fClassName()
, fParentName()
, fClonesName()
, fCollProxy(0)
, fCheckSum(0)
, fClassVersion(0)
, fID(0)
, fType(0)
, fStreamerType(-1)
, fMaximum(0)
, fSTLtype(TClassEdit::kNotSTL)
, fNdata(1)
, fBranchCount(0)
, fBranchCount2(0)
, fInfo(0)
, fObject(0)
, fOnfileObject(0)
, fInit(kFALSE)
, fInitOffsets(kFALSE)
, fTargetClass()
, fCurrentClass()
, fParentClass()
, fBranchClass()
, fBranchOffset(0)
, fBranchID(-1)
, fReadActionSequence(0)
, fIterators(0)
, fPtrIterators(0)
{
   // -- Default and I/O constructor.
   fNleaves = 0;
   fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesImpl;
}

//______________________________________________________________________________
TBranchElement::TBranchElement(TTree *tree, const char* bname, TStreamerInfo* sinfo, Int_t id, char* pointer, Int_t basketsize, Int_t splitlevel, Int_t btype)
: TBranch()
, fClassName(sinfo->GetName())
, fParentName()
, fClonesName()
, fCollProxy(0)
, fCheckSum(sinfo->GetCheckSum())
, fClassVersion(sinfo->GetClass()->GetClassVersion())
, fID(id)
, fType(0)
, fStreamerType(-1)
, fMaximum(0)
, fSTLtype(TClassEdit::kNotSTL)
, fNdata(1)
, fBranchCount(0)
, fBranchCount2(0)
, fInfo(sinfo)
, fObject(0)
, fOnfileObject(0)
, fInit(kTRUE)
, fInitOffsets(kFALSE)
, fTargetClass(fClassName)
, fCurrentClass()
, fParentClass()
, fBranchClass(sinfo->GetClass())
, fBranchOffset(0)
, fBranchID(-1)
, fReadActionSequence(0)
, fIterators(0)
, fPtrIterators(0)
{
   // -- Constructor when the branch object is not a TClonesArray nor an STL container.
   //
   // If splitlevel > 0 this branch in turn is split into sub-branches.

   Init(tree, 0, bname,sinfo,id,pointer,basketsize,splitlevel,btype);
}

//______________________________________________________________________________
TBranchElement::TBranchElement(TBranch *parent, const char* bname, TStreamerInfo* sinfo, Int_t id, char* pointer, Int_t basketsize, Int_t splitlevel, Int_t btype)
: TBranch()
, fClassName(sinfo->GetName())
, fParentName()
, fClonesName()
, fCollProxy(0)
, fCheckSum(sinfo->GetCheckSum())
, fClassVersion(sinfo->GetClass()->GetClassVersion())
, fID(id)
, fType(0)
, fStreamerType(-1)
, fMaximum(0)
, fSTLtype(TClassEdit::kNotSTL)
, fNdata(1)
, fBranchCount(0)
, fBranchCount2(0)
, fInfo(sinfo)
, fObject(0)
, fOnfileObject(0)
, fInit(kTRUE)
, fInitOffsets(kFALSE)
, fTargetClass( fClassName )
, fCurrentClass()
, fParentClass()
, fBranchClass(sinfo->GetClass())
, fBranchOffset(0)
, fBranchID(-1)
, fReadActionSequence(0)
, fIterators(0)
, fPtrIterators(0)
{
   // -- Constructor when the branch object is not a TClonesArray nor an STL container.
   //
   // If splitlevel > 0 this branch in turn is split into sub-branches.

   Init(parent ? parent->GetTree() : 0, parent, bname,sinfo,id,pointer,basketsize,splitlevel,btype);
}

//______________________________________________________________________________
void TBranchElement::Init(TTree *tree, TBranch *parent,const char* bname, TStreamerInfo* sinfo, Int_t id, char* pointer, Int_t basketsize, Int_t splitlevel, Int_t btype)
{
   // -- Init when the branch object is not a TClonesArray nor an STL container.
   //
   // If splitlevel > 0 this branch in turn is split into sub-branches.

   TString name(bname);

   // Set our TNamed attributes.
   SetName(name);
   SetTitle(name);

   // Set our TBranch attributes.
   fSplitLevel = splitlevel;
   fTree   = tree;
   if (fTree == 0) return;
   fMother = parent ? parent->GetMother() : this;
   fParent = parent;
   fDirectory = fTree->GetDirectory();
   fFileName = "";

   // Clear the bit kAutoDelete to specify that when reading
   // the object should not be deleted before calling Streamer.

   SetAutoDelete(kFALSE);

   fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesImpl;

   //---------------------------------------------------------------------------
   // Handling the splitting of the STL collections of pointers
   //---------------------------------------------------------------------------
   Int_t splitSTLP = splitlevel - (splitlevel%TTree::kSplitCollectionOfPointers);
   splitlevel %= TTree::kSplitCollectionOfPointers;

   fCompress = -1;
   if (fTree->GetDirectory()) {
      TFile* bfile = fTree->GetDirectory()->GetFile();
      if (bfile) {
         fCompress = bfile->GetCompressionLevel();
      }
   }

   //
   // Initialize streamer type and element.
   //

   if (id > -1) {
      // We are *not* a top-level branch.
      ULong_t* elems = sinfo->GetElems();
      TStreamerElement* element = (TStreamerElement*) elems[id];
      fStreamerType = element->GetType();
   }

   //
   // Handle varying-length datatypes by allocating an offsets array.
   //
   // The fBits part of a TObject is of varying length because the pidf
   // is streamed only when the TObject is referenced by a TRef.
   //

   fEntryOffsetLen = 0;
   if (btype || (fStreamerType <= TVirtualStreamerInfo::kBase) || (fStreamerType == TVirtualStreamerInfo::kCharStar) || (fStreamerType == TVirtualStreamerInfo::kBits) || (fStreamerType > TVirtualStreamerInfo::kFloat16)) {
      fEntryOffsetLen = fTree->GetDefaultEntryOffsetLen();
   }

   //
   // Make sure the basket is big enough to contain the
   // entry offset array plus 100 bytes of data.
   //

   if (basketsize < (100 + fEntryOffsetLen)) {
      basketsize = 100 + fEntryOffsetLen;
   }
   fBasketSize = basketsize;


   //
   // Allocate and initialize the basket control arrays.
   //

   fBasketBytes = new Int_t[fMaxBaskets];
   fBasketEntry = new Long64_t[fMaxBaskets];
   fBasketSeek = new Long64_t[fMaxBaskets];

   for (Int_t i = 0; i < fMaxBaskets; ++i) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i] = 0;
   }

   // We need to keep track of the counter branch if we have
   // one, since we cannot set it until we have created our
   // leaf, which we do last.
   TBranchElement* brOfCounter = 0;

   if (id < 0) {
      // -- We are a top-level branch.  Don't split a top-level branch, TTree::Bronch will do that work.
      if (fBranchClass.GetClass()) {
         Bool_t hasCustomStreamer = kFALSE;
         Bool_t canSelfReference = CanSelfReference(fBranchClass);
         if (fBranchClass.GetClass()->InheritsFrom(TObject::Class())) {
            if (canSelfReference) SetBit(kBranchObject);
            hasCustomStreamer = (!fBranchClass.GetClass()->GetCollectionProxy() && (gCint->ClassInfo_RootFlag(fBranchClass.GetClass()->GetClassInfo()) & 1));
         } else {
            if (canSelfReference) SetBit(kBranchAny);
            hasCustomStreamer = !fBranchClass.GetClass()->GetCollectionProxy() && (fBranchClass.GetClass()->GetStreamer() != 0 || (gCint->ClassInfo_RootFlag(fBranchClass.GetClass()->GetClassInfo()) & 1));
         }
         if (hasCustomStreamer) {
            fType = -1;
         }
      }
   } else {
      // -- We are a sub-branch of a split object.
      ULong_t* elems = sinfo->GetElems();
      TStreamerElement* element = (TStreamerElement*) elems[id];
      if ((fStreamerType == TVirtualStreamerInfo::kObject) || (fStreamerType == TVirtualStreamerInfo::kBase) || (fStreamerType == TVirtualStreamerInfo::kTNamed) || (fStreamerType == TVirtualStreamerInfo::kTObject) || (fStreamerType == TVirtualStreamerInfo::kObjectp) || (fStreamerType == TVirtualStreamerInfo::kObjectP)) {
         // -- If we are an object data member which inherits from TObject,
         // flag it so that later during i/o we will register the object
         // with the buffer so that pointers are handled correctly.
         if (CanSelfReference(fBranchClass)) {
            if (fBranchClass.GetClass()->InheritsFrom(TObject::Class())) {
               SetBit(kBranchObject);
            } else {
               SetBit(kBranchAny);
            }
         }
      }
      if (element->IsA() == TStreamerBasicPointer::Class()) {
         // -- Fixup title with counter if we are a varying length array data member.
         TStreamerBasicPointer *bp = (TStreamerBasicPointer *)element;
         TString countname;
         countname = bname;
         Ssiz_t dot = countname.Last('.');
         if (dot>=0) {
            countname.Remove(dot+1);
         } else {
            countname = "";
         }
         countname += bp->GetCountName();
         brOfCounter = (TBranchElement *)fTree->GetBranch(countname);
         countname.Form("%s[%s]",name.Data(),bp->GetCountName());
         SetTitle(countname);

      } else if (element->IsA() == TStreamerLoop::Class()) {
         // -- Fixup title with counter if we are a varying length array data member.
         TStreamerLoop *bp = (TStreamerLoop *)element;
         TString countname;
         countname = bname;
         Ssiz_t dot = countname.Last('.');
         if (dot>=0) {
            countname.Remove(dot+1);
         } else {
            countname = "";
         }
         countname += bp->GetCountName();
         brOfCounter = (TBranchElement *)fTree->GetBranch(countname);
         countname.Form("%s[%s]",name.Data(),bp->GetCountName());
         SetTitle(countname);

      }

      if (splitlevel > 0) {
         // -- Create sub branches if requested by splitlevel.
         const char* elem_type = element->GetTypeName();
         fSTLtype = TMath::Abs(TClassEdit::IsSTLCont(elem_type));
         if (element->CannotSplit()) {
            fSplitLevel = 0;
         } else if (element->IsA() == TStreamerBase::Class()) {
            // -- We are a base class element.
            // Note: This does not include an STL container class which is
            //        being used as a base class because the streamer element
            //        in that case is not the base streamer element it is the
            //        STL streamer element.
            fType = 1;
            TClass* clOfElement = TClass::GetClass(element->GetName());
            Int_t nbranches = fBranches.GetEntriesFast();
            // Note: The following code results in base class branches
            //       having two different cases for what their parent
            //       class will be, this is very annoying.  It is also
            //       very annoying that the naming conventions for the
            //       sub-branch names are different as well.
            if (!strcmp(name, clOfElement->GetName())) {
               // -- If the branch's name is the same as the base class name,
               // which happens when we are a child branch of a top-level
               // branch whose name does not end in a dot and also has no
               // internal dots, elide the branch name, and keep the branch
               // heirarchy rooted at the ultimate parent, this keeps the base
               // class part of the branch name from propagating downwards.
               // FIXME: We are eliding the base class here, creating a break in the branch hierarchy.
               // Note: We can use parent class (cltop) != branch class (elemClass) to detection elision.
               Unroll("", fBranchClass.GetClass(), clOfElement, pointer, basketsize, splitlevel+splitSTLP, 0);
               SetReadLeavesPtr();
               return;
            }
            // If the branch's name is not the same as the base class name,
            // keep the branch name as a prefix (i.e., continue the branch
            // heirarchy), but start a new class heirarchy at the base class.
            //
            // Note: If the parent branch was created by the branch constructor
            //       which takes a folder as a parameter, then this case will
            //       be used, because the branch name will be the same as the
            //       parent branch name.
            // Note: This means that the sub-branches of a base class branch
            //       created by TTree::Bronch() have the base class name as
            //       as part of the branch name, while those created by
            //       Unroll() do not, ouch!!!
            //
            Unroll(name, clOfElement, clOfElement, pointer, basketsize, splitlevel+splitSTLP, 0);
            if (strchr(bname, '.')) {
               // Note: How can this happen?
               // Answer: This is the case when using the new branch
               //        naming convention where the top-level branch ends in dot.
               // Note: Well actually not entirely, we could also be a sub-branch
               //        of a split class, even when the top-level branch does not
               //        end in a dot.
               // Note: Or the top-level branch could have been created by the
               //        branch constructor which takes a folder as input, in which
               //        case the top-level branch name will have internal dots
               //        representing the folder hierarchy.
               SetReadLeavesPtr();
               return;
            }
            if (nbranches == fBranches.GetEntriesFast()) {
               // -- We did not add any branches in the Unroll, finalize our name to be the base class name, because Unroll did not do it for us.
               if (strlen(bname)) {
                  name.Form("%s.%s", bname, clOfElement->GetName());
               } else {
                  name.Form("%s", clOfElement->GetName());
               }
               SetName(name);
               SetTitle(name);
            }
            SetReadLeavesPtr();
            return;
         } else if (!strcmp(elem_type, "TClonesArray") || !strcmp(elem_type, "TClonesArray*")) {
            // -- We are a TClonesArray element.
            Bool_t ispointer = !strcmp(elem_type,"TClonesArray*");
            TClonesArray *clones;
            if (ispointer) {
               char **ppointer = (char**)(pointer);
               clones = (TClonesArray*)(*ppointer);
            } else {
               clones = (TClonesArray*)pointer;
            }
//             basket->DeleteEntryOffset(); //entryoffset not required for the clonesarray counter
            fEntryOffsetLen = 0;
            // ===> Create a leafcount
            TLeaf* leaf = new TLeafElement(this, name, fID, fStreamerType);
            fNleaves = 1;
            fLeaves.Add(leaf);
            fTree->GetListOfLeaves()->Add(leaf);
            if (!clones) return;
            TClass* clOfClones = clones->GetClass();
            if (!clOfClones) {
               SetReadLeavesPtr();
               return;
            }
            fType = 3;
            // ===> create sub branches for each data member of a TClonesArray
            //check that the contained objects class name is part of the element title
            //This name is mandatory when reading the Tree later on and
            //the parent class with the pointer to the TClonesArray is not available.
            fClonesName = clOfClones->GetName();
            TString aname;
            aname.Form(" (%s)", clOfClones->GetName());
            TString atitle = element->GetTitle();
            if (!atitle.Contains(aname)) {
               atitle += aname;
               element->SetTitle(atitle.Data());
            }
            TString branchname( name );
            branchname += "_";
            SetTitle(branchname);
            leaf->SetName(branchname);
            leaf->SetTitle(branchname);
            Unroll(name, clOfClones, clOfClones, pointer, basketsize, splitlevel+splitSTLP, 31);
            BuildTitle(name);
            SetReadLeavesPtr();
            return;
         } else if (((fSTLtype >= TClassEdit::kVector) && (fSTLtype < TClassEdit::kEnd)) || ((fSTLtype > -TClassEdit::kEnd) && (fSTLtype <= -TClassEdit::kVector))) {
            // -- We are an STL container element.
            TClass* contCl = TClass::GetClass(elem_type);
            fCollProxy = contCl->GetCollectionProxy()->Generate();
            TClass* valueClass = GetCollectionProxy()->GetValueClass();
            // Check to see if we can split the container.
            Bool_t cansplit = kTRUE;
            if (!valueClass) {
               cansplit = kFALSE;
            } else if ((valueClass == TString::Class()) || (valueClass == TClass::GetClass("string"))) {
               cansplit = kFALSE;
            } else if (GetCollectionProxy()->HasPointers() && !splitSTLP ) {
               cansplit = kFALSE;
            } else if (!valueClass->CanSplit() && !(GetCollectionProxy()->HasPointers() && splitSTLP)) {
               cansplit = kFALSE;
            } else if (valueClass->GetCollectionProxy()) {
               // -- A collection was stored in a collection, we choose not to split it.
               // Note: Splitting it would require extending TTreeFormula
               //       to understand how to access it.
               cansplit = kFALSE;
            }
            if (cansplit) {
               // -- Do the splitting work if we are allowed to.
               fType = 4;
               // Create a leaf for the master branch (the counter).
               TLeaf *leaf = new TLeafElement(this, name, fID, fStreamerType);
               fNleaves = 1;
               fLeaves.Add(leaf);
               fTree->GetListOfLeaves()->Add(leaf);
               // Check that the contained objects class name is part of the element title.
               // This name is mandatory when reading the tree later on and
               // the parent class with the pointer to the STL container is not available.
               fClonesName = valueClass->GetName();
               TString aname;
               aname.Form(" (%s)", valueClass->GetName());
               TString atitle = element->GetTitle();
               if (!atitle.Contains(aname)) {
                  atitle += aname;
                  element->SetTitle(atitle.Data());
               }
               TString branchname (name);
               branchname += "_";
               SetTitle(branchname);
               leaf->SetName(branchname);
               leaf->SetTitle(branchname);
               // Create sub branches for each data member of an STL container.
               Unroll(name, valueClass, valueClass, pointer, basketsize, splitlevel+splitSTLP, 41);
               BuildTitle(name);
               SetReadLeavesPtr();
               return;
            }
         } else if (!strchr(elem_type, '*') && ((fStreamerType == TVirtualStreamerInfo::kObject) || (fStreamerType == TVirtualStreamerInfo::kAny))) {
            // -- Create sub-branches for members that are classes.
            //
            // Note: This can only happen if we were called directly
            //       (usually by TClass::Bronch) because Unroll never
            //       calls us for an element of this type.
            fType = 2;
            TClass* clm = TClass::GetClass(elem_type);
            Int_t err = Unroll(name, clm, clm, pointer, basketsize, splitlevel+splitSTLP, 0);
            if (err >= 0) {
               // Return on success.
               // FIXME: Why not on error too?
               SetReadLeavesPtr();
              return;
            }
         }
      }
   }

   //
   // Create a leaf to represent this branch.
   //

   TLeaf* leaf = new TLeafElement(this, GetTitle(), fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   fNleaves = 1;
   fLeaves.Add(leaf);
   fTree->GetListOfLeaves()->Add(leaf);

   //
   // If we have a counter branch set it now that we have
   // created our leaf, we cannot do it before then.
   //

   if (brOfCounter) {
      SetBranchCount(brOfCounter);
   }

   SetReadLeavesPtr();
}

//______________________________________________________________________________
TBranchElement::TBranchElement(TTree *tree, const char* bname, TClonesArray* clones, Int_t basketsize, Int_t splitlevel, Int_t compress)
: TBranch()
, fClassName("TClonesArray")
, fParentName()
, fInfo((TStreamerInfo*)TClonesArray::Class()->GetStreamerInfo())
, fTargetClass( fClassName )
, fCurrentClass()
, fParentClass()
, fBranchClass(TClonesArray::Class())
, fBranchID(-1)
, fReadActionSequence(0)
, fIterators(0)
, fPtrIterators(0)
{
   // -- Constructor when the branch object is a TClonesArray.
   //
   // If splitlevel > 0 this branch in turn is split into sub branches.

   Init(tree, 0, bname, clones, basketsize, splitlevel, compress);
}

//______________________________________________________________________________
TBranchElement::TBranchElement(TBranch *parent, const char* bname, TClonesArray* clones, Int_t basketsize, Int_t splitlevel, Int_t compress)
: TBranch()
, fClassName("TClonesArray")
, fParentName()
, fInfo((TStreamerInfo*)TClonesArray::Class()->GetStreamerInfo())
, fTargetClass( fClassName )
, fCurrentClass()
, fParentClass()
, fBranchClass(TClonesArray::Class())
, fBranchID(-1)
, fReadActionSequence(0)
, fIterators(0)
, fPtrIterators(0)
{
   // -- Constructor when the branch object is a TClonesArray.
   //
   // If splitlevel > 0 this branch in turn is split into sub branches.

   Init(parent ? parent->GetTree() : 0, parent, bname, clones, basketsize, splitlevel, compress);
}

//______________________________________________________________________________
void TBranchElement::Init(TTree *tree, TBranch *parent, const char* bname, TClonesArray* clones, Int_t basketsize, Int_t splitlevel, Int_t compress)
{
   // -- Init when the branch object is a TClonesArray.
   //
   // If splitlevel > 0 this branch in turn is split into sub branches.

   fCollProxy = 0;
   fSplitLevel    = splitlevel;
   fID            = 0;
   fInit          = kTRUE;
   fStreamerType  = -1;
   fType          = 0;
   fClassVersion  = TClonesArray::Class()->GetClassVersion();
   fCheckSum      = fInfo->GetCheckSum();
   fBranchCount   = 0;
   fBranchCount2  = 0;
   fObject        = 0;
   fOnfileObject  = 0;
   fMaximum       = 0;
   fBranchOffset  = 0;
   fSTLtype       = TClassEdit::kNotSTL;
   fInitOffsets   = kFALSE;

   fTree          = tree;
   fMother        = parent ? parent->GetMother() : this;
   fParent        = parent;
   fDirectory     = fTree->GetDirectory();
   fFileName      = "";

   SetName(bname);
   const char* name = GetName();
   SetTitle(name);
   //fClassName = fInfo->GetName();
   fCompress = compress;
   if (compress == -1 && fTree->GetDirectory()) {
      TFile *bfile = fTree->GetDirectory()->GetFile();
      if (bfile) fCompress = bfile->GetCompressionLevel();
   }

   if (basketsize < 100) basketsize = 100;
   fBasketSize     = basketsize;
   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketEntry    = new Long64_t[fMaxBaskets];
   fBasketSeek     = new Long64_t[fMaxBaskets];

   for (Int_t i=0;i<fMaxBaskets;i++) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i]  = 0;
   }

   // Reset the bit kAutoDelete to specify that when reading
   // the object should not be deleted before calling the streamer.
   SetAutoDelete(kFALSE);

   // create sub branches if requested by splitlevel
   if (splitlevel%TTree::kSplitCollectionOfPointers > 0) {
      TClass* clonesClass = clones->GetClass();
      if (!clonesClass) {
         Error("Init","Missing class object of the TClonesArray %s\n",clones->GetName());
         return;
      }
      fType = 3;
      // ===> Create a leafcount
      TLeaf* leaf = new TLeafElement(this, name, fID, fStreamerType);
      fNleaves = 1;
      fLeaves.Add(leaf);
      fTree->GetListOfLeaves()->Add(leaf);
      // ===> create sub branches for each data member of a TClonesArray
      fClonesName = clonesClass->GetName();
      std::string branchname = name + std::string("_");
      SetTitle(branchname.c_str());
      leaf->SetName(branchname.c_str());
      leaf->SetTitle(branchname.c_str());
      Unroll(name, clonesClass, clonesClass, 0, basketsize, splitlevel, 31);
      BuildTitle(name);
      SetReadLeavesPtr();
      return;
   }

   if (!clones->GetClass() || CanSelfReference(clones->GetClass())) {
      SetBit(kBranchObject);
   }
   TLeaf *leaf = new TLeafElement(this, GetTitle(), fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   fNleaves = 1;
   fLeaves.Add(leaf);
   fTree->GetListOfLeaves()->Add(leaf);

   SetReadLeavesPtr();
}

//______________________________________________________________________________
TBranchElement::TBranchElement(TTree *tree, const char* bname, TVirtualCollectionProxy* cont, Int_t basketsize, Int_t splitlevel, Int_t compress)
: TBranch()
, fClassName(cont->GetCollectionClass()->GetName())
, fParentName()
, fTargetClass( fClassName )
, fCurrentClass()
, fParentClass()
, fBranchClass(cont->GetCollectionClass())
, fBranchID(-1)
, fReadActionSequence(0)
, fIterators(0)
, fPtrIterators(0)
{
   // -- Constructor when the branch object is an STL collection.
   //
   // If splitlevel > 0 this branch in turn is split into sub branches.

   Init(tree, 0, bname, cont, basketsize, splitlevel, compress);
}

//______________________________________________________________________________
TBranchElement::TBranchElement(TBranch *parent, const char* bname, TVirtualCollectionProxy* cont, Int_t basketsize, Int_t splitlevel, Int_t compress)
: TBranch()
, fClassName(cont->GetCollectionClass()->GetName())
, fParentName()
, fTargetClass( fClassName )
, fCurrentClass()
, fParentClass()
, fBranchClass(cont->GetCollectionClass())
, fBranchID(-1)
, fReadActionSequence(0)
, fIterators(0)
, fPtrIterators(0)
{
   // -- Constructor when the branch object is an STL collection.
   //
   // If splitlevel > 0 this branch in turn is split into sub branches.

   Init(parent ? parent->GetTree() : 0, parent, bname, cont, basketsize, splitlevel, compress);
}

//______________________________________________________________________________
void TBranchElement::Init(TTree *tree, TBranch *parent, const char* bname, TVirtualCollectionProxy* cont, Int_t basketsize, Int_t splitlevel, Int_t compress)
{
   // -- Init when the branch object is an STL collection.
   //
   // If splitlevel > 0 this branch in turn is split into sub branches.

   fCollProxy = cont->Generate();
   TString name( bname );
   if (name[name.Length()-1]=='.') {
      name.Remove(name.Length()-1);
   }
   fInitOffsets   = kFALSE;
   fSplitLevel    = splitlevel;
   fInfo          = 0;
   fID            = -1;
   fInit          = kTRUE;
   fStreamerType  = -1; // TVirtualStreamerInfo::kSTLp;
   fType          = 0;
   fClassVersion  = cont->GetCollectionClass()->GetClassVersion();
   fCheckSum      = cont->GetCollectionClass()->GetCheckSum();
   fBranchCount   = 0;
   fBranchCount2  = 0;
   fObject        = 0;
   fOnfileObject  = 0;
   fMaximum       = 0;
   fBranchOffset  = 0;
   fSTLtype       = TClassEdit::kNotSTL;

   fTree          = tree;
   fMother        = parent ? parent->GetMother() : this;
   fParent        = parent;
   fDirectory     = fTree->GetDirectory();
   fFileName      = "";

   SetName(name);
   SetTitle(name);
   //fClassName = fBranchClass.GetClass()->GetName();
   fCompress = compress;
   if ((compress == -1) && fTree->GetDirectory()) {
      TFile* bfile = fTree->GetDirectory()->GetFile();
      if (bfile) {
         fCompress = bfile->GetCompressionLevel();
      }
   }

   if (basketsize < 100) {
      basketsize = 100;
   }
   fBasketSize     = basketsize;

   fBasketBytes    = new Int_t[fMaxBaskets];
   fBasketEntry    = new Long64_t[fMaxBaskets];
   fBasketSeek     = new Long64_t[fMaxBaskets];

   for (Int_t i = 0; i < fMaxBaskets; ++i) {
      fBasketBytes[i] = 0;
      fBasketEntry[i] = 0;
      fBasketSeek[i] = 0;
   }

   // Reset the bit kAutoDelete to specify that, when reading,
   // the object should not be deleted before calling the streamer.
   SetAutoDelete(kFALSE);

   // create sub branches if requested by splitlevel
   if ( (splitlevel%TTree::kSplitCollectionOfPointers > 0 && fBranchClass.GetClass() && fBranchClass.GetClass()->CanSplit()) ||
        (cont->HasPointers() && splitlevel > TTree::kSplitCollectionOfPointers && cont->GetValueClass() && cont->GetValueClass()->CanSplit() ) )
   {
      fType = 4;
      // ===> Create a leafcount
      TLeaf* leaf = new TLeafElement(this, name, fID, fStreamerType);
      fNleaves = 1;
      fLeaves.Add(leaf);
      fTree->GetListOfLeaves()->Add(leaf);
      // ===> create sub branches for each data member of an STL container value class
      TClass* valueClass = cont->GetValueClass();
      if (!valueClass) {
         return;
      }
      fClonesName = valueClass->GetName();
      TString branchname( name );
      branchname += "_";
      SetTitle(branchname);
      leaf->SetName(branchname);
      leaf->SetTitle(branchname);
      Unroll(name, valueClass, valueClass, 0, basketsize, splitlevel, 41);
      BuildTitle(name);
      SetReadLeavesPtr();
      return;
   }

   TLeaf *leaf = new TLeafElement(this, GetTitle(), fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   fNleaves = 1;
   fLeaves.Add(leaf);
   fTree->GetListOfLeaves()->Add(leaf);
   SetReadLeavesPtr();
}

//______________________________________________________________________________
TBranchElement::~TBranchElement()
{
   // -- Destructor.

   // Release any allocated I/O buffers.
   if (fOnfileObject && TestBit(kOwnOnfileObj)) {
      delete fOnfileObject;
      fOnfileObject = 0;
   }
   ResetAddress();

   delete[] fBranchOffset;
   fBranchOffset = 0;

   fInfo = 0;
   fBranchCount2 = 0;
   fBranchCount = 0;

   if (fType == 4) {
      // Only the top level TBranchElement containing an STL container,
      // owns the collectionproxy.
      delete fCollProxy;
   }
   fCollProxy = 0;

   delete fReadActionSequence;
   delete fIterators;
   delete fPtrIterators;
}

//
// This function is located here to allow inlining by the optimizer.
//
//______________________________________________________________________________
inline TStreamerInfo* TBranchElement::GetInfoImp() const
{
   // -- Get streamer info for the branch class.

   if (!fInfo || (fInfo && (!fInit || !fInfo->IsCompiled()))) {
      const_cast<TBranchElement*>(this)->InitInfo();
   }
   return fInfo;
}

//______________________________________________________________________________
TStreamerInfo* TBranchElement::GetInfo() const
{
   // -- Get streamer info for the branch class.

   return GetInfoImp();
}

//______________________________________________________________________________
void TBranchElement::Browse(TBrowser* b)
{
   // -- Browse the branch content.

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches > 0) {
      TList persistentBranches;
      TBranch* branch=0;
      TIter iB(&fBranches);
      while((branch=(TBranch*)iB())) {
         if (branch->IsFolder()) persistentBranches.Add(branch);
         else {
            // only show branches corresponding to persistent members
            TClass* cl=0;
            if (strlen(GetClonesName()))
               // this works both for top level branches and for sub-branches,
               // as GetClonesName() is properly updated for sub-branches
               cl=TClass::GetClass(GetClonesName());
            else {
               cl=TClass::GetClass(GetClassName());

               // check if we're in a sub-branch of this class
               // we can only find out asking the streamer given our ID
               ULong_t *elems=0;
               TStreamerElement *element=0;
               TClass* clsub=0;
               if (fID>=0 && GetInfoImp()
                   && ((elems=GetInfoImp()->GetElems()))
                   && ((element=(TStreamerElement *)elems[fID]))
                   && ((clsub=element->GetClassPointer())))
                  cl=clsub;
            }
            if (cl) {
               TString strMember=branch->GetName();
               Size_t mempos=strMember.Last('.');
               if (mempos!=kNPOS)
                  strMember.Remove(0, (Int_t)mempos+1);
               mempos=strMember.First('[');
               if (mempos!=kNPOS)
                  strMember.Remove((Int_t)mempos);
               TDataMember* m=cl->GetDataMember(strMember);
               if (!m || m->IsPersistent()) persistentBranches.Add(branch);
            } else persistentBranches.Add(branch);
         } // branch if not a folder
      }
      persistentBranches.Browse(b);
      // add all public const methods without params
      if (GetBrowsables() && GetBrowsables()->GetSize())
         GetBrowsables()->Browse(b);
   } else {
      if (GetBrowsables() && GetBrowsables()->GetSize()) {
         GetBrowsables()->Browse(b);
         return;
      }
      // Get the name and strip any extra brackets
      // in order to get the full arrays.
      TString slash("/");
      TString escapedSlash("\\/");
      TString name = GetName();
      Int_t pos = name.First('[');
      if (pos != kNPOS) {
         name.Remove(pos);
      }
      TString mothername;
      if (GetMother()) {
         mothername = GetMother()->GetName();
         pos = mothername.First('[');
         if (pos != kNPOS) {
            mothername.Remove(pos);
         }
         Int_t len = mothername.Length();
         if (len) {
            if (mothername(len-1) != '.') {
               // We do not know for sure whether the mother's name is
               // already preprended.  So we need to check:
               //    a) it is prepended
               //    b) it is NOT the name of a daugher (i.e. mothername.mothername exist)
               TString doublename = mothername;
               doublename.Append(".");
               Int_t isthere = (name.Index(doublename) == 0);
               if (!isthere) {
                  name.Prepend(doublename);
               } else {
                  if (GetMother()->FindBranch(mothername)) {
                     doublename.Append(mothername);
                     isthere = (name.Index(doublename) == 0);
                     if (!isthere) {
                        mothername.Append(".");
                        name.Prepend(mothername);
                     }
                  } else {
                     // Nothing to do because the mother's name is
                     // already in the name.
                  }
               }
            } else {
               // If the mother's name end with a dot then
               // the daughter probably already contains the mother's name
               if (name.Index(mothername) == kNPOS) {
                  name.Prepend(mothername);
               }
            }
         }
      }
      name.ReplaceAll(slash, escapedSlash);
      GetTree()->Draw(name, "", b ? b->GetDrawOption() : "");
      if (gPad) {
         gPad->Update();
      }
   }
}

//______________________________________________________________________________
void TBranchElement::BuildTitle(const char* name)
{
   // -- Set branch and leaf name and title in the case of a container sub-branch.

   TString branchname;

   Int_t nbranches = fBranches.GetEntries();

   for (Int_t i = 0; i < nbranches; ++i) {
      TBranchElement* bre = (TBranchElement*) fBranches.At(i);
      if (fType == 3) {
         bre->SetType(31);
      } else if (fType == 4) {
         bre->SetType(41);
      } else {
         Error("BuildTitle", "This cannot happen, fType of parent is not 3 or 4!");
      }
      bre->fCollProxy = GetCollectionProxy();
      bre->BuildTitle(name);
      const char* fin = strrchr(bre->GetTitle(), '.');
      if (fin == 0) {
         continue;
      }
      // The branch counter for a sub-branch of a container is the container master branch.
      bre->SetBranchCount(this);
      TLeafElement* lf = (TLeafElement*) bre->GetListOfLeaves()->At(0);
      // If branch name is of the form fTracks.fCovar[3][4], then
      // set the title to fCovar[fTracks_].
      branchname = fin+1;
      Ssiz_t dim = branchname.First('[');
      if (dim>=0) {
         branchname.Remove(dim);
      }
      branchname += Form("[%s_]",name);
      bre->SetTitle(branchname);
      if (lf) {
         lf->SetTitle(branchname);
      }
      // Is there a secondary branchcount?
      //
      // fBranchCount2 points to the secondary branchcount
      // in case a TClonesArray element itself has a branchcount.
      //
      // Example: In Event class with TClonesArray fTracks of Track objects.
      // if the Track object has two members
      //  Int_t    fNpoint;
      //  Float_t *fPoints;  //[fNpoint]
      // In this case the TBranchElement fTracks.fPoints has
      //  -its primary branchcount pointing to the branch fTracks
      //  -its secondary branchcount pointing to fTracks.fNpoint
      Int_t stype = bre->GetStreamerType();
      // FIXME: Should 60 be included here?
      if ((stype > 40) && (stype < 61)) {
         TString name2 (bre->GetName());
         Ssiz_t bn = name2.Last('.');
         if (bn<0) {
            continue;
         }
         TStreamerBasicPointer *el = (TStreamerBasicPointer*)bre->GetInfoImp()->GetElements()->FindObject(name2.Data()+bn+1);
         name2.Remove(bn+1);
         if (el) name2 += el->GetCountName();
         TBranchElement *bc2 = (TBranchElement*)fBranches.FindObject(name2);
         bre->SetBranchCount2(bc2);
      }
      bre->SetReadLeavesPtr();
   }
}

//______________________________________________________________________________
Int_t TBranchElement::Fill()
{
   // -- Loop on all leaves of this branch to fill the basket buffer.
   //
   // The function returns the number of bytes committed to the
   // individual branches.  If a write error occurs, the number of
   // bytes returned is -1.  If no data are written, because, e.g.,
   // the branch is disabled, the number of bytes returned is 0.
   //
   // Note: We not not use any member functions from TLeafElement!

   Int_t nbytes = 0;
   Int_t nwrite = 0;
   Int_t nerror = 0;
   Int_t nbranches = fBranches.GetEntriesFast();

   ValidateAddress();

   //
   // If we are a top-level branch, update addresses.
   //

   if (fID < 0) {
      if (!fObject) {
         Error("Fill", "attempt to fill branch %s while addresss is not set", GetName());
         return 0;
      }
   }

   //
   // If the tree has a TRefTable, set the current branch if
   // branch is not a basic type.
   //

   // FIXME: This test probably needs to be extended past 10.
   if ((fType >= -1) && (fType < 10)) {
      TBranchRef* bref = fTree->GetBranchRef();
      if (bref) {
         fBranchID = bref->SetParent(this, fBranchID);
      }
   }

   if (!nbranches) {
      // No sub-branches.
      if (!TestBit(kDoNotProcess)) {
         nwrite = TBranch::Fill();
         if (nwrite < 0) {
            Error("Fill", "Failed filling branch:%s, nbytes=%d", GetName(), nwrite);
            ++nerror;
         } else {
            nbytes += nwrite;
         }
      }
   } else {
      // We have sub-branches.
      if (fType == 3 || fType == 4) {
         // TClonesArray or STL container counter
         nwrite = TBranch::Fill();
         if (nwrite < 0) {
            Error("Fill", "Failed filling branch:%s, nbytes=%d", GetName(), nwrite);
            ++nerror;
         } else {
            nbytes += nwrite;
         }
      } else {
         ++fEntries;
      }
      for (Int_t i = 0; i < nbranches; ++i) {
         TBranchElement* branch = (TBranchElement*) fBranches[i];
         if (!branch->TestBit(kDoNotProcess)) {
            nwrite = branch->Fill();
            if (nwrite < 0) {
               Error("Fill", "Failed filling branch:%s.%s, nbytes=%d", GetName(), branch->GetName(), nwrite);
               nerror++;
            } else {
               nbytes += nwrite;
            }
         }
      }
   }

   if (fTree->Debug() > 0) {
      // Debugging.
      Long64_t entry = fEntries;
      if ((entry >= fTree->GetDebugMin()) && (entry <= fTree->GetDebugMax())) {
         printf("Fill: %lld, branch=%s, nbytes=%d\n", entry, GetName(), nbytes);
      }
   }

   if (nerror != 0) {
      return -1;
   }

   return nbytes;
}

//______________________________________________________________________________
void TBranchElement::FillLeaves(TBuffer& b)
{
   // -- Fill a buffer from the leaves of this branch.
   //
   // Note:  We do not use any member functions from TLeafElement!
   //        Except in the MakeClass case of a TClonesArray sub-branch.

   ValidateAddress();

   //
   // Silently do nothing if we have no user i/o buffer.
   //

   if (!fObject) {
      return;
   }

   //
   // Remember tobjects written to the buffer so that
   // pointers are handled correctly later.

   if (fType <=2) {
      if (TestBit(kBranchObject)) {
         b.MapObject((TObject*) fObject);
      } else if (TestBit(kBranchAny)) {
         b.MapObject(fObject, fBranchClass);
      }
   } else {
      // Case 3 and 4 nothing to do
      // Case 31 and 41, we should register the mother's fObject
   }

   //
   // Do the actual buffer filling now.
   //

   if (fType < 0) {
      // Non TObject, Non collection classes with a custom streamer.
      // if (fObject)
      fBranchClass->Streamer(fObject,b);
      return;

   } else if (fType <= 2) {
      // -- Top-level, data member, base class, or split class branch.
      // A non-split top-level branch (0, and fID == -1)), a non-split object (0, and fID > -1), or a base class (1), or a split (non-TClonesArray, non-STL container) object (2).  Write out the object.
      // Note: A split top-level branch (0, and fID == -2) should not happen here, see Fill().
      // FIXME: What happens with a split base class branch,
      //        or a split class branch???
      TStreamerInfo* si = GetInfoImp();
      if (!si) {
         Error("FillLeaves", "Cannot get streamer info for branch '%s'", GetName());
         return;
      }
      Int_t n = si->WriteBufferAux(b, &fObject, fID, 1, 0, 0);
      if ((fStreamerType == TVirtualStreamerInfo::kCounter) && (n > fMaximum)) {
         fMaximum = n;
      }
   } else if (fType == 3) {
      // -- TClonesArray top-level branch.  Write out number of entries, sub-branch writes the entries themselves.
      if (fTree->GetMakeClass()) {
         // Note: What if GetClonesName() is the zero pointer or an empty string?
         // Note: What if cl is a zero pointer here?
         // Answer: then fType would not have been set to 3, see TBranchElement::Init
         TClass* cl = TClass::GetClass(GetClonesName());
         TVirtualStreamerInfo* si = cl->GetStreamerInfo();
         if (!si) {
            Error("FillLeaves", "Cannot get streamer info for branch '%s' class '%s'", GetName(), cl->GetName());
            return;
         }
         b.ForceWriteInfo(si,kFALSE);
         Int_t* nptr = (Int_t*) fAddress;
         b << *nptr;
      } else {
         TClonesArray* clones = (TClonesArray*) fObject;
         Int_t n = clones->GetEntriesFast();
         if (n > fMaximum) {
            fMaximum = n;
         }
         b << n;
      }
   } else if (fType == 4) {
      // -- STL container top-level branch.  Write out number of entries, sub-branch writes the entries themselves.
      Int_t n = 0;
      {
         // We are in a block so the helper pops as soon as possible.
         TVirtualCollectionProxy::TPushPop helper(GetCollectionProxy(), fObject);
         n = GetCollectionProxy()->Size();
      }
      if (n > fMaximum) {
         fMaximum = n;
      }
      b << n;
   } else if (fType == 31) {
      // -- TClonesArray sub-branch.  Write out the entries in the TClonesArray.
      if (fTree->GetMakeClass()) {
         // -- A MakeClass() tree, we must use fAddress instead of fObject.
         if (!fAddress) {
            // FIXME: Enable this message.
            //Error("FillLeaves", "Branch address not set for branch '%s'!", GetName());
            return;
         }
         Int_t atype = fStreamerType;
         if (atype > 54) {
            // Note: We are not supporting kObjectp, kAny, kObjectp,
            //       kObjectP, kTString, kTObject, kTNamed, kAnyp,
            //       kAnyP, kSTLp, kSTL, kSTLstring, kStreamer,
            //       kStreamLoop here, nor pointers to varying length
            //       arrays of them either.
            //       Nor do we support pointers to varying length
            //       arrays of kBits, kLong64, kULong64, nor kBool.
            return;
         }
         Int_t* nn = (Int_t*) fBranchCount->GetAddress();
         if (!nn) {
            Error("FillLeaves", "The branch counter address was zero!");
            return;
         }
         Int_t n = *nn;
         if (atype > 40) {
            // Note: We are not supporting pointer to varying length array.
            Error("FillLeaves", "Clonesa: %s, n=%d, sorry not supported yet", GetName(), n);
            return;
         }
         if (atype > 20) {
            atype -= 20;
            TLeafElement* leaf = (TLeafElement*) fLeaves.UncheckedAt(0);
            n = n * leaf->GetLenStatic();
         }
         switch (atype) {
            // Note: Type 0 is a base class and cannot happen here, see Unroll().
            case TVirtualStreamerInfo::kChar     /*  1 */: { b.WriteFastArray((Char_t*)    fAddress, n); break; }
            case TVirtualStreamerInfo::kShort    /*  2 */: { b.WriteFastArray((Short_t*)   fAddress, n); break; }
            case TVirtualStreamerInfo::kInt      /*  3 */: { b.WriteFastArray((Int_t*)     fAddress, n); break; }
            case TVirtualStreamerInfo::kLong     /*  4 */: { b.WriteFastArray((Long_t*)    fAddress, n); break; }
            case TVirtualStreamerInfo::kFloat    /*  5 */: { b.WriteFastArray((Float_t*)   fAddress, n); break; }
            case TVirtualStreamerInfo::kCounter  /*  6 */: { b.WriteFastArray((Int_t*)     fAddress, n); break; }
            // FIXME: We do nothing with type 7 (TVirtualStreamerInfo::kCharStar, char*) here!
            case TVirtualStreamerInfo::kDouble   /*  8 */: { b.WriteFastArray((Double_t*)  fAddress, n); break; }
            case TVirtualStreamerInfo::kDouble32 /*  9 */: {
               TVirtualStreamerInfo* si = GetInfoImp();
               TStreamerElement* se = (TStreamerElement*) si->GetElems()[fID];
               Double_t* xx = (Double_t*) fAddress;
               for (Int_t ii = 0; ii < n; ++ii) {
                  b.WriteDouble32(&(xx[ii]),se);
               }
               break;
            }
            case TVirtualStreamerInfo::kFloat16 /*  19 */: {
               TVirtualStreamerInfo* si = GetInfoImp();
               TStreamerElement* se = (TStreamerElement*) si->GetElems()[fID];
               Float_t* xx = (Float_t*) fAddress;
               for (Int_t ii = 0; ii < n; ++ii) {
                  b.WriteFloat16(&(xx[ii]),se);
               }
               break;
            }
            // Note: Type 10 is unused for now.
            case TVirtualStreamerInfo::kUChar    /* 11 */: { b.WriteFastArray((UChar_t*)   fAddress, n); break; }
            case TVirtualStreamerInfo::kUShort   /* 12 */: { b.WriteFastArray((UShort_t*)  fAddress, n); break; }
            case TVirtualStreamerInfo::kUInt     /* 13 */: { b.WriteFastArray((UInt_t*)    fAddress, n); break; }
            case TVirtualStreamerInfo::kULong    /* 14 */: { b.WriteFastArray((ULong_t*)   fAddress, n); break; }
            // FIXME: This is wrong!!! TVirtualStreamerInfo::kBits is a variable length type.
            case TVirtualStreamerInfo::kBits     /* 15 */: { b.WriteFastArray((UInt_t*)    fAddress, n); break; }
            case TVirtualStreamerInfo::kLong64   /* 16 */: { b.WriteFastArray((Long64_t*)  fAddress, n); break; }
            case TVirtualStreamerInfo::kULong64  /* 17 */: { b.WriteFastArray((ULong64_t*) fAddress, n); break; }
            case TVirtualStreamerInfo::kBool     /* 18 */: { b.WriteFastArray((Bool_t*)    fAddress, n); break; }
         }
      } else {
         TClonesArray* clones = (TClonesArray*) fObject;
         Int_t n = clones->GetEntriesFast();
         TStreamerInfo* si = (TStreamerInfo*)GetInfoImp();
         if (!si) {
            Error("FillLeaves", "Cannot get streamer info for branch '%s'", GetName());
            return;
         }
         si->WriteBufferClones(b, clones, n, fID, fOffset);
      }
   } else if (fType == 41) {
      // -- STL container sub-branch.  Write out the entries in the STL container.

      // FIXME: This wont work if a pointer to vector is split!
      Int_t n = 0;
      TVirtualCollectionProxy::TPushPop helper(GetCollectionProxy(), fObject);
      n = GetCollectionProxy()->Size();
      // Note: We cannot pop the proxy here because we need it for the i/o.
      TStreamerInfo* si = (TStreamerInfo*)GetInfoImp();
      if (!si) {
         Error("FillLeaves", "Cannot get streamer info for branch '%s'", GetName());
         return;
      }
      if( fSplitLevel >= TTree::kSplitCollectionOfPointers )
         si->WriteBufferSTLPtrs(b, GetCollectionProxy(), n, fID, fOffset );
      else
         si->WriteBufferSTL(b, GetCollectionProxy(), n, fID, fOffset );
   }
}

//______________________________________________________________________________
static void R__CleanName(std::string &name)
{
   // Remove trailing dimensions and make sure
   // there is a trailing dot.

   if (name[name.length()-1]==']') {
      std::size_t dim = name.find_first_of("[");
      if (dim != std::string::npos) {
         name.erase(dim);
      }
   }
   if (name[name.size()-1] != '.') {
      name += '.';
   }
}

//______________________________________________________________________________
TBranch* TBranchElement::FindBranch(const char *name)
{
   // Find the immediate sub-branch with passed name.

   // The default behavior of TBranch::FindBranch is sometimes
   // incorrect if this branch represent a base class, since
   // the base class name might or might not be in the name
   // of the sub-branches and might or might not be in the
   // name being passed.

   if (fID >= 0) {
      TVirtualStreamerInfo* si = GetInfoImp();
      TStreamerElement* se = (TStreamerElement*) si->GetElems()[fID];
      if (se && se->IsBase()) {
         // We allow the user to pass only the last dotted component of the name.
         UInt_t len = strlen(name);
         std::string longnm;
         longnm.reserve(fName.Length()+len+3); // Enough space of fName + name + dots
         longnm = fName.Data();
         R__CleanName(longnm);
         longnm += name;
         std::string longnm_parent;
         longnm_parent.reserve(fName.Length()+len+3);
         longnm_parent = (GetMother()->GetSubBranch(this)->GetName());
         R__CleanName(longnm_parent);
         longnm_parent += name;  // Name without the base class name

         UInt_t namelen = strlen(name);

         TBranch* branch = 0;
         Int_t nbranches = fBranches.GetEntries();
         for(Int_t i = 0; i < nbranches; ++i) {
            branch = (TBranch*) fBranches.UncheckedAt(i);

            const char *brname = branch->GetName();
            UInt_t brlen = strlen(brname);
            if (brname[brlen-1]==']') {
               const char *dim = strchr(brname,'[');
               if (dim) {
                  brlen = dim - brname;
               }
            }
            if (namelen == brlen /* same effective size */
                && strncmp(name,brname,brlen) == 0) {
               return branch;
            }
            if (brlen == longnm.length()
                && strncmp(longnm.c_str(),brname,brlen) == 0) {
                return branch;
            }
            // This check is specific to base class
            if (brlen == longnm_parent.length()
                && strncmp(longnm_parent.c_str(),brname,brlen) == 0) {
               return branch;
            }

            if (namelen>brlen && name[brlen]=='.' && strncmp(name,brname,brlen)==0) {
               // The prefix subbranch name match the branch name.
               return branch->FindBranch(name+brlen+1);
            }
         }
      }
   }
   TBranch *result = TBranch::FindBranch(name);
   if (!result) {
      // Look in base classes if any
      Int_t nbranches = fBranches.GetEntries();
      for(Int_t i = 0; i < nbranches; ++i) {
         TObject *obj = fBranches.UncheckedAt(i);
         if(obj->IsA() != TBranchElement :: Class() )
            continue;
         TBranchElement *br = (TBranchElement*)obj;
         TVirtualStreamerInfo* si = br->GetInfoImp();
         if (si && br->GetID() >= 0) {
            TStreamerElement* se = (TStreamerElement*) si->GetElems()[br->GetID()];
            if (se && se->IsBase()) {
               result = br->FindBranch(name);
            }
         }
      }
   }
   return result;
}

//______________________________________________________________________________
TLeaf* TBranchElement::FindLeaf(const char *name)
{
   // -- Find the leaf corresponding to the name 'searchname'.

   TLeaf *leaf = TBranch::FindLeaf(name);

   if (leaf==0 && GetListOfLeaves()->GetEntries()==1) {
      TBranch *br = GetMother()->GetSubBranch( this );
      if( br->IsA() != TBranchElement::Class() )
         return 0;

      TBranchElement *parent = (TBranchElement*)br;
      if (parent==this || parent->GetID()<0 ) return 0;

      TVirtualStreamerInfo* si = parent->GetInfoImp();
      TStreamerElement* se = (TStreamerElement*) si->GetElems()[parent->GetID()];

      if (! se->IsBase() ) return 0;

      br = GetMother()->GetSubBranch( parent );
      if( br->IsA() != TBranchElement::Class() )
         return 0;

      TBranchElement *grand_parent = (TBranchElement*)br;

      std::string longname( grand_parent->GetName() );
      R__CleanName(longname);
      longname += name;

      std::string leafname( GetListOfLeaves()->At(0)->GetName() );

      if ( longname == leafname ) {
         return (TLeaf*)GetListOfLeaves()->At(0);
      }
   }
   return leaf;
}

//______________________________________________________________________________
char* TBranchElement::GetAddress() const
{
   // -- Get the branch address.
   //
   // If we are *not* owned by a MakeClass() tree:
   //
   //      If we are a top-level branch, return a pointer
   //      to the pointer to our object.
   //
   //      If we are *not* a top-level branch, return a pointer
   //      to our object.
   //
   // If we are owned by a MakeClass() tree:
   //
   //      Return a pointer to our object.
   //

   ValidateAddress();
   return fAddress;
}

//______________________________________________________________________________
void TBranchElement::InitInfo()
{
   // -- Init the streamer info for the branch class, try to compensate for class code unload/reload and schema evolution.

   if (!fInfo) {
      // We did not already have streamer info, so now we must find it.
      TClass* cl = fBranchClass.GetClass();

      //------------------------------------------------------------------------
      // Check if we're dealing with the name change
      //------------------------------------------------------------------------
      TClass* targetClass = 0;
      if( fTargetClass.GetClassName()[0]) {
         targetClass = fTargetClass;
         if( !targetClass ) {
            Error( "InitInfo", "The target class dictionary is not present!" );
            return;
         }
      } else {
         targetClass = cl;
      }

      if (cl) {
         //---------------------------------------------------------------------
         // Get the streamer info for given version
         //---------------------------------------------------------------------
         {
            if( targetClass != cl ) {
               fInfo = (TStreamerInfo*)targetClass->GetConversionStreamerInfo( cl, fClassVersion );
            } else {
               fInfo = (TStreamerInfo*)cl->GetStreamerInfo(fClassVersion);
            }
         }

         // FIXME: Check that the found streamer info checksum matches our branch class checksum here.
         // Check to see if the class code was unloaded/reloaded
         // since we were created.
         if (fCheckSum && (cl->IsForeign() || (!cl->IsLoaded() && (fClassVersion == 1) && cl->GetStreamerInfos()->At(1) && (fCheckSum != ((TVirtualStreamerInfo*) cl->GetStreamerInfos()->At(1))->GetCheckSum())))) {
            // Try to compensate for a class that got unloaded on us.
            // Search through the streamer infos by checksum
            // and take the first match.

            TStreamerInfo* info;
            if( targetClass != cl )
               info = (TStreamerInfo*)targetClass->GetConversionStreamerInfo( cl, fCheckSum );
            else {
               info = (TStreamerInfo*)cl->FindStreamerInfo( fCheckSum );
               if (info) {
                  // Now that we found it, we need to make sure it is initialize (Find does not initialize the StreamerInfo).
                  info = (TStreamerInfo*)cl->GetStreamerInfo(info->GetClassVersion());
               }
            }
            if( info ) {
               fInfo = info;
               // We no longer reset the class version so that in case the user is passing us later
               // the address of a class that require (another) Conversion we can find the proper
               // StreamerInfo.
               //    fClassVersion = fInfo->GetClassVersion();
            }
         }
      }
   }

   //
   //  Fixup cached streamer info if necessary.
   //
   // FIXME:  What if the class code was unloaded/reloaded since we were cached?

   if (fInfo) {

      if ( GetID()>-1 && (!fInfo->IsCompiled() || fInfo->IsOptimized()) ) {
         // Streamer info has not yet been compiled.
         //
         // Optimizing does not work with splitting.
         fInfo->SetBit(TVirtualStreamerInfo::kCannotOptimize);
         fInfo->Compile();
      }
      if (!fInit) {
         // We were read in from a file, figure out what our fID should be,
         // schema evolution must be considered.
         //
         // Force our fID to be the id of the first streamer element that matches our name.
         //
         if (GetID() > -1) {
            // We are *not* a top-level branch.
            std::string s(GetName());
            size_t pos = s.rfind('.');
            if (pos != std::string::npos) {
               s = s.substr(pos+1);
            }
            while ((pos = s.rfind('[')) != std::string::npos) {
               s = s.substr(0, pos);
            }
            int offset = 0;
            TStreamerElement* elt = fInfo->GetStreamerElement(s.c_str(), offset);
            if (elt && offset!=TStreamerInfo::kMissing) {
               size_t ndata = fInfo->GetNdata();
               ULong_t* elems = fInfo->GetElems();
               fIDs.clear();
               for (size_t i = 0; i < ndata; ++i) {
                  if (((TStreamerElement*) elems[i]) == elt) {
                     if (elt->TestBit (TStreamerElement::kCache)
                         && elt->TestBit(TStreamerElement::kRepeat)
                         && (i+1) < ndata
                         && s == ((TStreamerElement*) elems[i])->GetName())
                     {
                        // If the TStreamerElement we found is storing the information in the
                        // cache and is a repeater, we need to use the real one (the next one).
                        // (At least until the cache/repeat mechanism is properly handle by
                        // ReadLeaves).
                        // fID = i+1;
                        fID = i;
                        fIDs.push_back(fID+1);
                     } else {
                        fID = i;
                     }
                     if (elt->TestBit (TStreamerElement::kCache)) {
                        SetBit(TBranchElement::kCache);
                     }
                     break;
                  }
               }
               for (size_t i = fID+1+(fIDs.size()); i < ndata; ++i) {
                  TStreamerElement *nextel = ((TStreamerElement*) elems[i]);
                  // Add all (and only) the Artificial Elements that follows this StreamerInfo.
                  if (fType==31||fType==41) {
                     // The nested objects are unfolded and their branch can not be used to
                     // execute StreamerElements of this StreamerInfo.
                     if (nextel->GetType() == TStreamerInfo::kObject
                         || nextel->GetType() == TStreamerInfo::kAny) {
                        continue;
                     }
                  }
                  if (nextel->GetOffset() ==  TStreamerInfo::kMissing) {
                     // This element will be 'skipped', it's TBranchElement's fObject will null
                     // and thus can not be used to execute the artifical StreamerElements
                     continue;
                  }
                  if (nextel->IsA() != TStreamerArtificial::Class()
                      || nextel->GetType() == TStreamerInfo::kCacheDelete ) {
                     break;
                  }
                  fIDs.push_back(i);
               }
            } else if (elt && offset==TStreamerInfo::kMissing) {
               // Still re-assign fID properly.
               fIDs.clear();
               size_t ndata = fInfo->GetNdata();
               ULong_t* elems = fInfo->GetElems();
               for (size_t i = 0; i < ndata; ++i) {
                  if (((TStreamerElement*) elems[i]) == elt) {
                     fID = i;
                     break;
                  }
               }
            } else {
               // We have not even found the element .. this is strange :(
               // fIDs.clear();
               // fID = -3;
               // SetBit(kDoNotProcess);
            }
            if (fOnfileObject==0 && (fType==31 || fType==41 || (0 <= fType && fType <=2) ) && fInfo->GetNdata()
                && ((TStreamerElement*) fInfo->GetElems()[0])->GetType() == TStreamerInfo::kCacheNew)
            {
               Int_t arrlen = 1;
               if (fType==31 || fType==41) {
                  TLeaf *leaf = (TLeaf*)fLeaves.At(0);
                  if (leaf) {
                     arrlen = leaf->GetMaximum();
                  }
               }
               fOnfileObject = new TVirtualArray( ((TStreamerElement*) fInfo->GetElems()[0])->GetClassPointer(), arrlen );
               // Propage this to all the other branch of this type.
               TObjArray *branches = GetMother()->GetSubBranch(this)->GetListOfBranches();
               Int_t nbranches = branches->GetEntriesFast();
               TBranchElement *lastbranch = this;
               for (Int_t i = 0; i < nbranches; ++i) {
                  TBranchElement* subbranch = (TBranchElement*)branches->At(i);
                  if (this!=subbranch && subbranch->fBranchClass == fBranchClass && subbranch->fCheckSum == fCheckSum) {
                     subbranch->fOnfileObject = fOnfileObject;
                     lastbranch = subbranch;
                  }
               }
               lastbranch->SetBit(kOwnOnfileObj);
            }
         }
         fInit = kTRUE;

         // Get the action sequence we need to copy for reading.
         SetReadActionSequence();
      } else if (!fReadActionSequence) {
         // Get the action sequence we need to copy for reading.
         SetReadActionSequence();
      }
      SetReadLeavesPtr();
   }
}

//______________________________________________________________________________
TVirtualCollectionProxy* TBranchElement::GetCollectionProxy()
{
   // -- Return the collection proxy describing the branch content, if any.

   if (fCollProxy) {
      return fCollProxy;
   }
   TBranchElement* thiscast = const_cast<TBranchElement*>(this);
   if (fType == 4) {
      // STL container top-level branch.
      const char* className = 0;
      if (fID < 0) {
         // We are a top-level branch.
         if (fBranchClass.GetClass()) {
           className = fBranchClass.GetClass()->GetName();
         }
      } else {
         // We are not a top-level branch.
         TVirtualStreamerInfo* si = thiscast->GetInfoImp();
         TStreamerElement* se = (TStreamerElement*) si->GetElems()[fID];
         className = se->GetTypeName();
      }
      TClass* cl = TClass::GetClass(className);
      TVirtualCollectionProxy* proxy = cl->GetCollectionProxy();
      fCollProxy = proxy->Generate();
      fSTLtype = className ? TClassEdit::IsSTLCont(className) : 0;
      if (fSTLtype < 0) {
        fSTLtype = -fSTLtype;
      }
   } else if (fType == 41) {
      // STL container sub-branch.
      thiscast->fCollProxy = fBranchCount->GetCollectionProxy();
   }
   return fCollProxy;
}

//______________________________________________________________________________
TClass* TBranchElement::GetCurrentClass()
{
   // -- Return a pointer to the current type of the data member corresponding to branch element.

   TClass* cl = fCurrentClass;
   if (cl) {
      return cl;
   }

   TStreamerInfo* brInfo = (TStreamerInfo*)GetInfoImp();
   if (!brInfo) {
      cl = TClass::GetClass(GetClassName());
      R__ASSERT(cl && cl->GetCollectionProxy());
      fCurrentClass = cl;
      return cl;
   }
   TClass* motherCl = brInfo->GetClass();
   if (motherCl->GetCollectionProxy()) {
      cl = motherCl->GetCollectionProxy()->GetCollectionClass();
      if (cl) {
         fCurrentClass = cl;
      }
      return cl;
   }
   if (GetID() < 0 || GetID()>=brInfo->GetNdata()) {
      return 0;
   }
   TStreamerElement* currentStreamerElement = ((TStreamerElement*) brInfo->GetElems()[GetID()]);
   TDataMember* dm = (TDataMember*) motherCl->GetListOfDataMembers()->FindObject(currentStreamerElement->GetName());

   TString newType;
   if (!dm) {
      // Either the class is not loaded or the data member is gone
      if (!motherCl->IsLoaded()) {
         TVirtualStreamerInfo* newInfo = motherCl->GetStreamerInfo();
         if (newInfo != brInfo) {
            TStreamerElement* newElems = (TStreamerElement*) newInfo->GetElements()->FindObject(currentStreamerElement->GetName());
            if (newElems) {
               newType = newElems->GetClassPointer()->GetName();
            }
         }
         if (newType.Length()==0) {
            newType = currentStreamerElement->GetClassPointer()->GetName();
         }
      }
   } else {
      newType = dm->GetTypeName();
   }
   cl = TClass::GetClass(newType);
   if (cl) {
      fCurrentClass = cl;
   }
   return cl;
}

//______________________________________________________________________________
Int_t TBranchElement::GetEntry(Long64_t entry, Int_t getall)
{
   // -- Read all branches of a BranchElement and return total number of bytes.
   //
   // If entry = 0, then use current entry number + 1.
   // If entry < 0, then reset entry number to 0.
   //
   // Returns the number of bytes read from the input buffer.
   // If entry does not exist, then returns 0.
   // If an I/O error occurs, then returns -1.
   //
   // See IMPORTANT REMARKS in TTree::GetEntry.
   //

   // Remember which entry we are reading.
   fReadEntry = entry;

   // If our tree has a branch ref, make it remember the entry and
   // this branch.  This allows a TRef::GetObject() call done during
   // the following I/O operation, for example in a custom streamer,
   // to search for the referenced object in the proper element of the
   // proper branch.
   TBranchRef* bref = fTree->GetBranchRef();
   if (bref) {
      fBranchID = bref->SetParent(this, fBranchID);
      bref->SetReadEntry(entry);
   }

   Int_t nbytes = 0;

   if (IsAutoDelete()) {
      SetBit(kDeleteObject);
      SetAddress(fAddress);
   }
   SetupAddresses();

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches) {
      // -- Branch has daughters.
      // One must always read the branch counter.
      // In the case when one reads consecutively twice the same entry,
      // the user may have cleared the TClonesArray between the GetEntry calls.
      if ((fType == 3) || (fType == 4)) {
         Int_t nb = TBranch::GetEntry(entry, getall);
         if (nb < 0) {
            return nb;
         }
         nbytes += nb;
      }
      switch(fSTLtype) {
         case TClassEdit::kSet:
         case TClassEdit::kMultiSet:
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
            break;
         default:
            for (Int_t i = 0; i < nbranches; ++i) {
               TBranch* branch = (TBranch*) fBranches.UncheckedAt(i);
               Int_t nb = branch->GetEntry(entry, getall);
               if (nb < 0) {
                  return nb;
               }
               nbytes += nb;
            }
            break;
      }
   } else {
      // -- Terminal branch.
      if (fBranchCount && (fBranchCount->GetReadEntry() != entry)) {
         Int_t nb = fBranchCount->TBranch::GetEntry(entry, getall);
         if (nb < 0) {
            return nb;
         }
         nbytes += nb;
      }
      Int_t nb = TBranch::GetEntry(entry, getall);
      if (nb < 0) {
         return nb;
      }
      nbytes += nb;
   }

   if (fTree->Debug() > 0) {
      if ((entry >= fTree->GetDebugMin()) && (entry <= fTree->GetDebugMax())) {
         Info("GetEntry", "%lld, branch=%s, nbytes=%d", entry, GetName(), nbytes);
      }
   }
   return nbytes;
}

//______________________________________________________________________________
Int_t TBranchElement::GetExpectedType(TClass *&expectedClass,EDataType &expectedType)
{
   // Fill expectedClass and expectedType with information on the data type of the
   // object/values contained in this branch (and thus the type of pointers
   // expected to be passed to Set[Branch]Address
   // return 0 in case of success and > 0 in case of failure.

   expectedClass = 0;
   expectedType = kOther_t;

   Int_t type = GetStreamerType();
   if ((type == -1) || (fID == -1)) {
      expectedClass = fBranchClass;
   } else {
      // Case of an object data member.  Here we allow for the
      // variable name to be ommitted.  Eg, for Event.root with split
      // level 1 or above  Draw("GetXaxis") is the same as Draw("fH.GetXaxis()")
      TStreamerElement* element = (TStreamerElement*) GetInfoImp()->GetElems()[fID];
      if (element) {
         expectedClass = element->GetClassPointer();
         if (!expectedClass) {
            TDataType* data = gROOT->GetType(element->GetTypeNameBasic());
            if (!data) {
               Error("GetExpectedType", "Did not find the type number for %s", element->GetTypeNameBasic());
               return 1;
            } else {
               expectedType = (EDataType) data->GetType();
            }
         }
      } else {
         Error("GetExpectedType", "Did not find the type for %s",GetName());
         return 2;
      }
   }
   return 0;
}

//______________________________________________________________________________
const char* TBranchElement::GetIconName() const
{
   // -- Return icon name depending on type of branch element.

   if (IsFolder()) {
      return "TBranchElement-folder";
   } else {
      return "TBranchElement-leaf";
   }
}

//______________________________________________________________________________
Bool_t TBranchElement::GetMakeClass() const
{
   // Return whether this branch is in a mode where the object are decomposed
   // or not (Also known as MakeClass mode).

   return TestBit(kDecomposedObj); // Same as TestBit(kMakeClass)
}

//______________________________________________________________________________
Int_t TBranchElement::GetMaximum() const
{
   // -- Return maximum count value of the branchcount if any.

   if (fBranchCount) {
      return fBranchCount->GetMaximum();
   }
   return fMaximum;
}

//______________________________________________________________________________
char* TBranchElement::GetObject() const
{
   // -- Return a pointer to our object.

   ValidateAddress();
   return fObject;
}

//______________________________________________________________________________
TClass* TBranchElement::GetParentClass()
{
   // -- Return a pointer to the parent class of the branch element.
   return fParentClass.GetClass();
}

//______________________________________________________________________________
const char* TBranchElement::GetTypeName() const
{
   // -- Return type name of element in the branch.

   if (fType == 3  || fType == 4) {
      return "Int_t";
   }
   // FIXME: Use symbolic constants here.
   if ((fStreamerType < 1) || (fStreamerType > 59)) {
      if (fBranchClass.GetClass()) {
         if (fID>=0) {
            ULong_t* elems = GetInfoImp()->GetElems();
            return ((TStreamerElement*) elems[fID])->GetTypeName();
         } else {
            return fBranchClass.GetClass()->GetName();
         }
      } else {
         return 0;
      }
   }
   const char *types[20] = {
      "",
      "Char_t",
      "Short_t",
      "Int_t",
      "Long_t",
      "Float_t",
      "Int_t",
      "char*",
      "Double_t",
      "Double32_t",
      "",
      "UChar_t",
      "UShort_t",
      "UInt_t",
      "ULong_t",
      "UInt_t",
      "Long64_t",
      "ULong64_t",
      "Bool_t",
      "Float16_t"
   };
   Int_t itype = fStreamerType % 20;
   return types[itype];
}

//______________________________________________________________________________
Double_t TBranchElement::GetValue(Int_t j, Int_t len, Bool_t subarr) const
{
   // -- Returns the branch value.
   //
   // If the leaf is an array, j is the index in the array.
   //
   // If leaf is an array inside a TClonesArray, len should be the length
   // of the array.
   //
   // If subarr is true, then len is actually the index within the sub-array.
   //

   ValidateAddress();

   Int_t prID = fID;
   char *object = fObject;
   if (TestBit(kCache)) {
      if (GetInfoImp()->GetElements()->At(fID)->TestBit(TStreamerElement::kRepeat)) {
         prID = fID+1;
      } else if (fOnfileObject) {
         object = fOnfileObject->GetObjectAt(0);
      }
   }

   if (!j && fBranchCount) {
      Long64_t entry = fTree->GetReadEntry();
      // Since reloading the index, will reset the ClonesArray, let's
      // skip the load if we already read this entry.
      if (entry != fBranchCount->GetReadEntry()) {
         fBranchCount->TBranch::GetEntry(entry);
      }
      if (fBranchCount2 && entry != fBranchCount2->GetReadEntry()) {
         fBranchCount2->TBranch::GetEntry(entry);
      }
   }

   if (fTree->GetMakeClass()) {
      if (!fAddress) {
         return 0;
      }
      if ((fType == 3) || (fType == 4)) {
         // Top-level branch of a TClonesArray.
         return (Double_t) fNdata;
      } else if ((fType == 31) || (fType == 41)) {
         // sub branch of a TClonesArray
         Int_t atype = fStreamerType;
         if (atype < 20) {
            atype += 20;
         }
         return GetInfoImp()->GetValue(fAddress, atype, j, 1);
      } else if (fType <= 2) {
         // branch in split mode
         // FIXME: This should probably be < 60 instead!
         if ((fStreamerType > 40) && (fStreamerType < 55)) {
            Int_t atype = fStreamerType - 20;
            return GetInfoImp()->GetValue(fAddress, atype, j, 1);
         } else {
            return GetInfoImp()->GetValue(object, prID, j, -1);
         }
      }
   }

   if (fType == 31) {
      TClonesArray* clones = (TClonesArray*) object;
      if (subarr) {
         return GetInfoImp()->GetValueClones(clones, prID, j, len, fOffset);
      }
      return GetInfoImp()->GetValueClones(clones, prID, j/len, j%len, fOffset);
   } else if (fType == 41) {
      TVirtualCollectionProxy::TPushPop helper(((TBranchElement*) this)->GetCollectionProxy(), object);
      if( fSplitLevel < TTree::kSplitCollectionOfPointers )
      {
         if (subarr)
            return GetInfoImp()->GetValueSTL(((TBranchElement*) this)->GetCollectionProxy(), prID, j, len, fOffset);

         return GetInfoImp()->GetValueSTL(((TBranchElement*) this)->GetCollectionProxy(), prID, j/len, j%len, fOffset);
      }
      else
      {
         if (subarr)
            return GetInfoImp()->GetValueSTLP(((TBranchElement*) this)->GetCollectionProxy(), prID, j, len, fOffset);
         return GetInfoImp()->GetValueSTLP(((TBranchElement*) this)->GetCollectionProxy(), prID, j/len, j%len, fOffset);
      }
   } else {
      if (GetInfoImp()) {
         return GetInfoImp()->GetValue(object, prID, j, -1);
      }
      return 0;
   }
}

//______________________________________________________________________________
void* TBranchElement::GetValuePointer() const
{
   // -- Returns pointer to first data element of this branch.
   // Currently used only for members of type character.

   ValidateAddress();

   Int_t prID = fID;
   char *object = fObject;
   if (TestBit(kCache)) {
      if (GetInfoImp()->GetElements()->At(fID)->TestBit(TStreamerElement::kRepeat)) {
         prID = fID+1;
      } else if (fOnfileObject) {
         object = fOnfileObject->GetObjectAt(0);
      }
   }

   if (fBranchCount) {
      Long64_t entry = fTree->GetReadEntry();
      fBranchCount->TBranch::GetEntry(entry);
      if (fBranchCount2) fBranchCount2->TBranch::GetEntry(entry);
   }
   if (fTree->GetMakeClass()) {
      if (!fAddress) {
         return 0;
      }
      if (fType == 3) {    //top level branch of a TClonesArray
         //return &fNdata;
         return 0;
      } else if (fType == 4) {    //top level branch of a TClonesArray
         //return &fNdata;
         return 0;
      } else if (fType == 31) {    // sub branch of a TClonesArray
         //Int_t atype = fStreamerType;
         //if (atype < 20) atype += 20;
         //return GetInfoImp()->GetValue(fAddress, atype, j, 1);
         return 0;
      } else if (fType == 41) {    // sub branch of a TClonesArray
         //Int_t atype = fStreamerType;
         //if (atype < 20) atype += 20;
         //return GetInfoImp()->GetValue(fAddress, atype, j, 1);
         return 0;
      } else if (fType <= 2) {     // branch in split mode
         // FIXME: This should probably be < 60 instead!
         if (fStreamerType > 40 && fStreamerType < 55) {
            //Int_t atype = fStreamerType - 20;
            //return GetInfoImp()->GetValue(fAddress, atype, j, 1);
            return 0;
         } else {
            //return GetInfoImp()->GetValue(object, fID, j, -1);
            return 0;
         }
      }
   }

   if (fType == 31) {
      return 0;
   } else if (fType == 41) {
      return 0;
   } else {
      //return GetInfoImp()->GetValue(object,fID,j,-1);
      if (!GetInfoImp() || !object) return 0;
      char **val = (char**)(object+GetInfoImp()->GetOffsets()[fID]);
      return *val;
   }
}

//______________________________________________________________________________
void TBranchElement::InitializeOffsets()
{
   // -- Initialize the base class subobjects offsets of our sub-branches and set fOffset if we are a container sub-branch.
   //
   // Note: The offsets are zero for data members so that when
   //       SetAddress recursively sets their address, they will get the
   //       same address as their containing class because i/o is based
   //       on streamer info offsets from the addresss of the containing
   //       class.
   //
   //       Offsets are non-zero for base-class sub-branches that are
   //       not the leftmost direct base class.  They are laid out in
   //       memory sequentially and only the leftmost direct base class
   //       has the same address as the derived class.  The streamer
   //       offsets need to be added to the address of the base class
   //       subobject which is not the same as the address of the
   //       derived class for the non-leftmost direct base classes.

   Int_t nbranches = fBranches.GetEntriesFast();

   if (fID < 0) {
      // -- We are a top-level branch.  Let's mark whether we need to use MapObject.
      if (CanSelfReference(fBranchClass)) {
         if (fBranchClass.GetClass()->InheritsFrom(TObject::Class())) {
            SetBit(kBranchObject);
         } else {
            SetBit(kBranchAny);
         }
      }
   }
   if (nbranches) {
      // Allocate space for the new sub-branch offsets.
      delete[] fBranchOffset;
      fBranchOffset = 0;
      fBranchOffset = new Int_t[nbranches];
      // Make sure we can instantiate our class meta info.
      if (!fBranchClass.GetClass()) {
         Warning("InitializeOffsets", "No branch class set for branch: %s", GetName());
         fInitOffsets = kTRUE;
         return;
      }
      // Make sure we can instantiate our class streamer info.
      if (!GetInfoImp()) {
         Warning("InitializeOffsets", "No streamer info available for branch: %s of class: %s", GetName(), fBranchClass.GetClass()->GetName());
         fInitOffsets = kTRUE;
         return;
      }

      // Get the class we are a member of now (which is the
      // type of our containing subobject) and get our offset
      // inside of our containing subobject (our local offset).
      // Note: branchElem stays zero if we are a top-level branch,
      //       we have to be careful about this later.
      TStreamerElement* branchElem = 0;
      Int_t localOffset = 0;
      TClass* branchClass = fBranchClass.GetClass();
      if (fID > -1) {
         // -- Branch is *not* a top-level branch.
         // Instead of the streamer info class, we want the class of our
         // specific element in the streamer info.  We could be a data
         // member of a base class or a split class, in which case our
         // streamer info will be for our containing sub-object, while
         // we are actually a different type.
         TVirtualStreamerInfo* si = GetInfoImp();
         // Note: We tested to make sure the streamer info was available previously.
         ULong_t* elems = si->GetElems();
         if (!elems) {
            Warning("InitializeOffsets", "Streamer info for branch: %s has no elements array!", GetName());
            fInitOffsets = kTRUE;
            return;
         }
         // FIXME: Check that fID is in range.
         branchElem = (TStreamerElement*) elems[fID];
         if (!branchElem) {
            Warning("InitializeOffsets", "Cannot get streamer element for branch: %s!", GetName());
            fInitOffsets = kTRUE;
            return;
         }
         localOffset = branchElem->GetOffset();
         branchClass = branchElem->GetClassPointer();
         if (localOffset == TStreamerInfo::kMissing) {
            fObject = 0;
         }
      }
      if (!branchClass) {
         Error("InitializeOffsets", "Could not find class for branch: %s", GetName());
         fInitOffsets = kTRUE;
         return;
      }

      //------------------------------------------------------------------------
      // Extract the name of the STL branch in case we're splitting the
      // collection of pointers
      //------------------------------------------------------------------------
      TString stlParentName;
      Bool_t stlParentNameUpdated = kFALSE;
      if( fType == 4 && fSplitLevel > TTree::kSplitCollectionOfPointers )
      {
         TBranch *br = GetMother()->GetSubBranch( this );
         stlParentName = br->GetName();
         stlParentName.Strip( TString::kTrailing, '.' );

         // We may ourself contain the 'Mother' branch name.
         // To avoid code duplication, we delegate the removal
         // of the mother's name to the first sub-branch loop.
      }

      // Loop over our sub-branches and compute their offsets.
      for (Int_t subBranchIdx = 0; subBranchIdx < nbranches; ++subBranchIdx) {
         fBranchOffset[subBranchIdx] = 0;
         TBranchElement* subBranch = dynamic_cast<TBranchElement*> (fBranches[subBranchIdx]);
         if (subBranch == 0) {
            // -- Skip sub-branches that are not TBranchElements.
            continue;
         }

         TVirtualStreamerInfo* sinfo = subBranch->GetInfoImp();
         if (!sinfo) {
            Warning("InitializeOffsets", "No streamer info for branch: %s subbranch: %s", GetName(), subBranch->GetName());
            fInitOffsets = kTRUE;
            return;
         }
         ULong_t* subBranchElems = sinfo->GetElems();
         if (!subBranchElems) {
            Warning("InitializeOffsets", "No elements array for branch: %s subbranch: %s", GetName(), subBranch->GetName());
            fInitOffsets = kTRUE;
            return;
         }
         // FIXME: Make sure subBranch->fID is in range.
         TStreamerElement* subBranchElement = (TStreamerElement*) subBranchElems[subBranch->fID];
         if (!subBranchElement) {
            Warning("InitializeOffsets", "No streamer element for branch: %s subbranch: %s", GetName(), subBranch->GetName());
            fInitOffsets = kTRUE;
            return;
         }

         localOffset = subBranchElement->GetOffset();
         if (localOffset == TStreamerInfo::kMissing) {
            subBranch->fObject = 0;
         }

         {
            Int_t streamerType = subBranchElement->GetType();
            if (streamerType > TStreamerInfo::kObject
                && subBranch->GetListOfBranches()->GetEntries()==0
                && CanSelfReference(subBranchElement->GetClass()))
            {
               subBranch->SetBit(kBranchAny);
            } else {
               subBranch->ResetBit(kBranchAny);
            }
         }

         if (subBranchElement->GetNewType()<0) {
            subBranch->ResetBit(kBranchAny);
            subBranch->ResetBit(kBranchObject);
         }

         // Note: This call is expensive, do it only once.
         TBranch* mother = GetMother();
         if (!mother) {
            Warning("InitializeOffsets", "Branch '%s' has no mother!", GetName());
            fInitOffsets = kTRUE;
            return;
         }
         TString motherName(mother->GetName());
         Bool_t motherDot = kFALSE;
         if (motherName.Length() && strchr(motherName.Data(), '.')) {
            motherDot = kTRUE;
         }
         Bool_t motherDotAtEnd = kFALSE;
         if (motherName.Length() && (motherName[motherName.Length()-1] == '.')) {
            motherDotAtEnd = kTRUE;
         }

         Bool_t isBaseSubBranch = kFALSE;
         if ((subBranch->fType == 1) || (subBranchElement && subBranchElement->IsBase())) {
            // -- Base class sub-branch (1).
            //
            // Note: Our type will not be 1, even though we are
            // a base class branch, if we are not split (see the
            // constructor), or if we are an STL container master
            // branch and a base class branch at the same time
            // or an std::string.
            isBaseSubBranch = kTRUE;
         }

         Bool_t isContDataMember = kFALSE;
         if ((subBranch->fType == 31) || (subBranch->fType == 41)) {
            // -- Container data member sub-branch (31 or 41).
            isContDataMember = kTRUE;
         }

         // I am either a data member sub-branch (0), or a base class
         // sub-branch (1), or TClonesArray master sub-branch (3),
         // or an STL container master sub-branch (4), or TClonesArray
         // data member sub-branch (31), or an STL container data member
         // sub-branch (41).
         //
         // My parent branch is either a top-level branch ((0), fID==(-2,-1)),
         // or a base class sub-branch (1), or a split-class branch (2),
         // or a TClonesArray master branch (3), or an STL container
         // master branch (4).
         //

         //
         // We need to extract from our name the name
         // of the data member which contains us, so
         // that we may then do a by-name lookup in the
         // dictionary meta info of our parent class to
         // get our offset in our parent class.
         //

         // Get our name.
         TString dataName(subBranch->GetName());
         if (motherDotAtEnd) {
            // -- Remove the top-level branch name from our name.
            dataName.Remove(0, motherName.Length());
            // stlParentNameUpdated is false the first time in this loop.
            if (!stlParentNameUpdated && stlParentName.Length()) {
               stlParentName.Remove(0, motherName.Length());
               stlParentNameUpdated = kTRUE;
            }
         } else if (motherDot) {
            // -- Remove the top-level branch name from our name, folder case.
            //
            // Note: We are in the case where our mother was created
            //       by the branch constructor which takes a folder
            //       as an argument.  The mother branch has internal
            //       dots in its name to represent the folder heirarchy.
            //       The TTree::Bronch() routine has handled us as a
            //       special case, we must compensate.
            if ((fID < 0) && (subBranchElement->IsA() == TStreamerBase::Class())) {
               // -- Our name is the mother name, remove it.
               // Note: The test is our parent is a top-level branch
               //       and our streamer is the base class streamer,
               //       this matches the exact test in TTree::Bronch().
               if (dataName.Length() == motherName.Length()) {
                  dataName.Remove(0, motherName.Length());
                  // stlParentNameUpdated is false the first time in this loop.
                  if (!stlParentNameUpdated && stlParentName.Length()) {
                     stlParentName.Remove(0, motherName.Length());
                  }
               }
            } else {
               // -- Remove the mother name and the dot.
               if (dataName.Length() > motherName.Length()) {
                  dataName.Remove(0, motherName.Length() + 1);
                  if (!stlParentNameUpdated && stlParentName.Length()) {
                     stlParentName.Remove(0, motherName.Length());
                  }
               }
            }
         }
         stlParentNameUpdated = kTRUE;
         if (isBaseSubBranch) {
            // -- Remove the base class name suffix from our name.
            // Note: The pattern is the name of the base class.
            TString pattern(subBranchElement->GetName());
            if (pattern.Length() <= dataName.Length()) {
               if (!strcmp(dataName.Data() + (dataName.Length() - pattern.Length()), pattern.Data())) {
                  // The branch name contains the name of the base class in it.
                  // This name is not reproduced in the sub-branches, so we need to
                  // remove it.
                  dataName.Remove(dataName.Length() - pattern.Length());
               }
            }
            // Remove any leading dot.
            if (dataName.Length()) {
               if (dataName[0] == '.') {
                  dataName.Remove(0, 1);
               }
            }
            // Note: We intentionally leave any trailing dot
            //       in our modified name here.
         }

         // Get our parent branch's name.
         TString parentName(GetName());
         if (motherDotAtEnd) {
            // -- Remove the top-level branch name from our parent's name.
            parentName.Remove(0, motherName.Length());
         } else if (motherDot) {
            // -- Remove the top-level branch name from our parent's name, folder case.
            //
            // Note: We are in the case where our mother was created
            //       by the branch constructor which takes a folder
            //       as an argument.  The mother branch has internal
            //       dots in its name to represent the folder heirarchy.
            //       The TTree::Bronch() routine has handled us as a
            //       special case, we must compensate.
            if ((fID > -1) && (mother == mother->GetSubBranch(this)) && (branchElem->IsA() == TStreamerBase::Class())) {
               // -- Our parent's name is the mother name, remove it.
               // Note: The test is our parent's parent is a top-level branch
               //       and our parent's streamer is the base class streamer,
               //       this matches the exact test in TTree::Bronch().
               if (parentName.Length() == motherName.Length()) {
                  parentName.Remove(0, motherName.Length());
               }
            } else {
               // -- Remove the mother name and the dot.
               if (parentName.Length() > motherName.Length()) {
                  parentName.Remove(0, motherName.Length() + 1);
               }
            }
         }
         // FIXME: Do we need to use the other tests for a base class here?
         if (fType == 1) {
            // -- Our parent is a base class sub-branch, remove the base class name suffix from its name.
            if (mother != mother->GetSubBranch(this)) {
               // -- My parent's parent is not a top-level branch.
               // Remove the base class name suffix from the parent name.
               // Note: The pattern is the name of the base class.
               TString pattern(branchElem->GetName());
               if (pattern.Length() <= parentName.Length()) {
                  if (!strcmp(parentName.Data() + (parentName.Length() - pattern.Length()), pattern.Data())) {
                     // The branch name contains the name of the base class in it.
                     // This name is not reproduced in the sub-branches, so we need to
                     // remove it.
                     parentName.Remove(parentName.Length() - pattern.Length());
                  }
               }
            }
            // Note: We intentionally leave any trailing dots
            //       in the modified parent name here.
         }

         // Remove the parent branch name part from our name,
         // but only if the parent branch is not a top-level branch.
         // FIXME: We should not assume parent name does not have length 0.
         if (fID > -1) {
           RemovePrefix(dataName, parentName);
         }

         // Remove any leading dot.
         if (dataName.Length()) {
            if (dataName[0] == '.') {
               dataName.Remove(0, 1);
            }
         }

         // Remove any trailing dot.
         if (dataName.Length()) {
            if (dataName[dataName.Length()-1] == '.') {
               dataName.Remove(dataName.Length() - 1, 1);
            }
         }

         //
         // Now that we have our data member name, find our offset
         // in our parent class.
         //
         // Note:  Our data member name can have many dots in it
         //        if branches were elided between our parent branch
         //        and us by Unroll().
         //
         // FIXME: This may not work if our member name is ambiguous.
         //

         Int_t offset = 0;
         if (dataName.Length()) {
            // -- We have our data member name, do a lookup in the dictionary meta info of our parent class.
            // Get our parent class.
            TClass* pClass = 0;
            // First check whether this sub-branch is part of the 'cache' (because the data member it
            // represent is no longer in the current class layout.
            TStreamerInfo *subInfo = subBranch->GetInfoImp();
            if (subInfo && subBranch->TestBit(kCache)) { // subInfo->GetElements()->At(subBranch->GetID())->TestBit(TStreamerElement::kCache)) {
               pClass = ((TStreamerElement*)subInfo->GetElements()->At(0))->GetClassPointer();
            }
            // FIXME: Do we need the other base class tests here?
            if (!pClass) {
               if (fType == 1) {
                  // -- Parent branch is a base class branch.
                  // FIXME: Is using branchElem here the right thing?
                  pClass = branchElem->GetClassPointer();
               } else {
                  // -- Parent branch is *not* a base class branch.
                  // FIXME: This sometimes returns a null pointer.
                  pClass = subBranch->GetParentClass();
               }
            }
            if (!pClass) {
               // -- No parent class, fix it.
               // FIXME: This is probably wrong!
               // Assume parent class is our parent branch's clones class or value class.
               if (GetClonesName() && strlen(GetClonesName())) {
                  pClass = TClass::GetClass(GetClonesName());
                  if (!pClass) {
                     Warning("InitializeOffsets", "subBranch: '%s' has no parent class, and cannot get class for clones class: '%s'!", subBranch->GetName(), GetClonesName());
                     fInitOffsets = kTRUE;
                     return;
                  }
                  Warning("InitializeOffsets", "subBranch: '%s' has no parent class!  Assuming parent class is: '%s'.", subBranch->GetName(), pClass->GetName());
               }
               if (fBranchCount && fBranchCount->fCollProxy && fBranchCount->fCollProxy->GetValueClass()) {
                  pClass = fBranchCount->fCollProxy->GetValueClass();
                  Warning("InitializeOffsets", "subBranch: '%s' has no parent class!  Assuming parent class is: '%s'.", subBranch->GetName(), pClass ? pClass->GetName() : "unknowned class");
               }
               if (!pClass) {
                  // -- Still no parent class, assume our parent class is our parent branch's class.
                  // FIXME: This is probably wrong!
                  pClass = branchClass;
                  // FIXME: Enable this warning!
                  //Warning("InitializeOffsets", "subBranch: '%s' has no parent class!  Assuming parent class is: '%s'.", subBranch->GetName(), pClass->GetName());
               }
            }

            //------------------------------------------------------------------
            // If we have the are the sub-branch of the TBranchSTL, we need
            // to remove it's name to get the correct real data offsets
            //-----------------------------------------------------------------
            if( stlParentName.Length() )
            {
               if( !strncmp( stlParentName.Data(), dataName.Data(), stlParentName.Length()-1 ))
                  dataName.Remove( 0, stlParentName.Length()+1 );
            }

            // Find our offset in our parent class using
            // a lookup by name in the dictionary meta info
            // for our parent class.
            TRealData* rd = pClass->GetRealData(dataName);
            if (rd && !rd->TestBit(TRealData::kTransient)) {
               // -- Data member exists in the dictionary meta info, get the offset.
               offset = rd->GetThisOffset();
            } else {
               // -- No dictionary meta info for this data member, it must no longer exist.
               localOffset = TStreamerInfo::kMissing;
            }
         } else {
            // -- We have no data member name, ok for a base class, not good otherwise.
            if (isBaseSubBranch) {
               // I am a direct base class of my parent class, my local offset is enough.
            } else {
               Warning("InitializeOffsets", "Could not find the data member name for branch '%s' with parent branch '%s', assuming offset is zero!", subBranch->GetName(), GetName());
            }
         }

         //
         // Ok, do final calculations for fOffset and fBranchOffset.
         //

         if (isContDataMember) {
            // -- Container data members set fOffset instead of fBranchOffset.
            // The fOffset is what should be added to the start of the entry
            // in the collection (i.e., its current absolute address) to find
            // the beginning of the data member described by the current branch.
            //
            // Compensate for the i/o routines adding our local offset later.
            if (subBranch->fObject == 0 && localOffset == TStreamerInfo::kMissing) {
               subBranch->SetOffset(TStreamerInfo::kMissing);
            } else {
               if (isBaseSubBranch) {
                  // The value of 'offset' for a base class does not include its
                  // 'localOffset'.
                  subBranch->SetOffset(offset);
               } else {
                  // The value of 'offset' for a regular data member does include its
                  // 'localOffset', we need to remove it explicitly.
                  subBranch->SetOffset(offset - localOffset);
               }
            }
         } else {
            // -- Set fBranchOffset for sub-branch.
            Int_t isSplit = 0 != subBranch->GetListOfBranches()->GetEntriesFast();
            if (subBranch->fObject == 0 && localOffset == TStreamerInfo::kMissing) {
               // The branch is missing
               fBranchOffset[subBranchIdx] = TStreamerInfo::kMissing;

            } else if (isSplit) {
               if (isBaseSubBranch) {
                  // We are split, so we need to add in our local offset
                  // to get our absolute address for our children.
                  fBranchOffset[subBranchIdx] = offset + localOffset;
               } else {
                  // We are split so our offset will never be
                  // used in an i/o, so we do not have to subtract
                  // off our local offset like below.
                  fBranchOffset[subBranchIdx] = offset;
               }
            } else {
               if (isBaseSubBranch) {
                  // We are not split, so our local offset will be
                  // added later by the i/o routines.
                  fBranchOffset[subBranchIdx] = offset;
               } else {
                  // Compensate for the fact that the i/o routines
                  // are going to add my local offset later.
                  fBranchOffset[subBranchIdx] = offset - localOffset;
               }
            }
         }
      }
   }

   fInitOffsets = kTRUE;
}

//______________________________________________________________________________
Bool_t TBranchElement::IsFolder() const
{
   // -- Return kTRUE if more than one leaf, kFALSE otherwise.

   Int_t nbranches = fBranches.GetEntriesFast();
   if (nbranches >= 1) {
      return kTRUE;
   }
   TList* browsables = const_cast<TBranchElement*>(this)->GetBrowsables();
   return browsables && browsables->GetSize();
}

//______________________________________________________________________________
Bool_t TBranchElement::IsMissingCollection() const
{
   // -- Detect a collection written using a zero pointer in old versions of root.
   // In versions of ROOT older than 4.00/03, if a collection (TClonesArray
   // or STL container) was split but the pointer to the collection was zeroed
   // out, nothing was saved.  Hence there is no __easy__ way to detect the
   // case.  In newer versions, a zero is written so that a 'missing' collection
   // appears to be an empty collection.

   Bool_t ismissing = kFALSE;
   TBasket* basket = (TBasket*) fBaskets.UncheckedAt(fReadBasket);
   if (basket && fTree) {
      Long64_t entry = fTree->GetReadEntry();
      Long64_t first  = fBasketEntry[fReadBasket];
      Long64_t last;
      if (fReadBasket == fWriteBasket) {
         last = fEntryNumber - 1;
      } else {
         last = fBasketEntry[fReadBasket+1] - 1;
      }
      Int_t* entryOffset = basket->GetEntryOffset();
      Int_t bufbegin;
      Int_t bufnext;
      if (entryOffset) {
         bufbegin = entryOffset[entry-first];

         if (entry < last) {
            bufnext = entryOffset[entry+1-first];
         } else {
            bufnext = basket->GetLast();
         }
         if (bufnext == bufbegin) {
            ismissing = kTRUE;
         } else {
            // fixed length buffer so this is not the case here.
            if (basket->GetNevBufSize() == 0) {
               ismissing = kTRUE;
            }
         }
      }
   }
   return ismissing;
}

//______________________________________________________________________________
void TBranchElement::Print(Option_t* option) const
{
   // Print branch parameters.

   Int_t nbranches = fBranches.GetEntriesFast();
   if (strncmp(option,"debugAddress",strlen("debugAddress"))==0) {
      if (strlen(option)==strlen("debugAddress")) {
         Printf("%-24s %-16s %2s %4s %-16s %-16s %8s %8s %s\n",
                "Branch Name", "Streamer Class", "ID", "Type", "Class", "Parent", "pOffset", "fOffset", "fObject");
      }
      if (strlen(GetName())>24) Printf("%-24s\n%-24s ", GetName(),"");
      else Printf("%-24s ", GetName());

      TBranchElement *parent = dynamic_cast<TBranchElement*>(GetMother()->GetSubBranch(this));
      Int_t ind = parent ? parent->GetListOfBranches()->IndexOf(this) : -1;
      TVirtualStreamerInfo *info = ((TBranchElement*)this)->GetInfoImp();

      Printf("%-16s %2d %4d %-16s %-16s %8x %8x %s\n",
             info ? info->GetName() : "StreamerInfo unvailable", GetID(), GetType(),
             GetClassName(), GetParentName(),
             (fBranchOffset&&parent && ind>=0) ? parent->fBranchOffset[ind] : 0,
             GetOffset(), GetObject());
      for (Int_t i = 0; i < nbranches; ++i) {
         TBranchElement* subbranch = (TBranchElement*)fBranches.At(i);
         subbranch->Print("debugAddressSub");
      }
      return;
   }
   if (strncmp(option,"debugInfo",strlen("debugInfo"))==0)  {
      Printf("Branch %s uses:\n",GetName());
      if (fID>=0) {
         ULong_t* elems = GetInfoImp()->GetElems();
         ((TStreamerElement*) elems[fID])->ls();
         for(UInt_t i=0; i< fIDs.size(); ++i) {
            ((TStreamerElement*) elems[fIDs[i]])->ls();
         }
      }
      for (Int_t i = 0; i < nbranches; ++i) {
         TBranchElement* subbranch = (TBranchElement*)fBranches.At(i);
         subbranch->Print("debugInfoSub");
      }
      return;
   }
   if (nbranches) {
      if (fID == -2) {
         if (strcmp(GetName(),GetTitle()) == 0) {
            Printf("*Branch  :%-66s *",GetName());
         } else {
            Printf("*Branch  :%-9s : %-54s *",GetName(),GetTitle());
         }
         Printf("*Entries : %8d : BranchElement (see below)                              *",Int_t(fEntries));
         Printf("*............................................................................*");
      }
      if (fType >= 2) {
         TBranch::Print(option);
      }
      for (Int_t i=0;i<nbranches;i++) {
         TBranch *branch = (TBranch*)fBranches.At(i);
         branch->Print(option);
      }
   } else {
      TBranch::Print(option);
   }
}

//______________________________________________________________________________
void TBranchElement::PrintValue(Int_t lenmax) const
{
   // -- Prints values of leaves.

   ValidateAddress();

   TStreamerInfo *info = GetInfoImp();
   Int_t prID = fID;
   char *object = fObject;
   if (TestBit(kCache)) {
      if (info->GetElements()->At(fID)->TestBit(TStreamerElement::kRepeat)) {
         prID = fID+1;
      } else if (fOnfileObject) {
         object = fOnfileObject->GetObjectAt(0);
      }
   }

   if (fTree->GetMakeClass()) {
      if (!fAddress) {
         return;
      }
      if (fType == 3 || fType == 4) {
         // TClonesArray or STL container top-level branch.
         printf(" %-15s = %d\n", GetName(), fNdata);
         return;
      } else if (fType == 31 || fType == 41) {
         // TClonesArray or STL container sub-branch.
         Int_t n = TMath::Min(10, fNdata);
         Int_t atype = fStreamerType + TVirtualStreamerInfo::kOffsetL;
         if (fStreamerType == TVirtualStreamerInfo::kChar) {
            // TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kChar is
            // printed as a string and could print weird characters.
            // So we print an unsigned char instead (not perfect, but better).
            atype = TVirtualStreamerInfo::kOffsetL + TVirtualStreamerInfo::kUChar;
         }
         if (atype > 54) {
            // FIXME: More logic required here (like in ReadLeaves)
            printf(" %-15s = %d\n", GetName(), fNdata);
            return;
         }
         if (fStreamerType > 20) {
            atype -= 20;
            TLeafElement* leaf = (TLeafElement*) fLeaves.UncheckedAt(0);
            n = n * leaf->GetLenStatic();
         }
         if (GetInfoImp()) {
            GetInfoImp()->PrintValue(GetName(), fAddress, atype, n, lenmax);
         }
         return;
      } else if (fType <= 2) {
         // Branch in split mode.
         // FIXME: This should probably be < 60 instead.
         if ((fStreamerType > 40) && (fStreamerType < 55)) {
            Int_t atype = fStreamerType - 20;
            TBranchElement* counterElement = (TBranchElement*) fBranchCount;
            Int_t n = (Int_t) counterElement->GetValue(0, 0);
            if (GetInfoImp()) {
               GetInfoImp()->PrintValue(GetName(), fAddress, atype, n, lenmax);
            }
         } else {
            if (GetInfoImp()) {
               GetInfoImp()->PrintValue(GetName(), object, prID, -1, lenmax);
            }
         }
         return;
      }
   } else if (fType == 3) {
      printf(" %-15s = %d\n", GetName(), fNdata);
   } else if (fType == 31) {
      TClonesArray* clones = (TClonesArray*) object;
      if (GetInfoImp()) {
         GetInfoImp()->PrintValueClones(GetName(), clones, prID, fOffset, lenmax);
      }
   } else if (fType == 41) {
      TVirtualCollectionProxy::TPushPop helper(((TBranchElement*) this)->GetCollectionProxy(), object);
      if (GetInfoImp()) {
         GetInfoImp()->PrintValueSTL(GetName(), ((TBranchElement*) this)->GetCollectionProxy(), prID, fOffset, lenmax);
      }
   } else {
      if (GetInfoImp()) {
         GetInfoImp()->PrintValue(GetName(), object, prID, -1, lenmax);
      }
   }
}

//______________________________________________________________________________
void  TBranchElement::ReadLeavesImpl(TBuffer&)
{
   // -- Unconfiguration Read Leave function.

   Fatal("ReadLeaves","The ReadLeaves function has not been configured for %s",GetName());
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesMakeClass(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // For the case where the branch is set in MakeClass mode (decomposed object).

   ValidateAddress();

   if (fType == 3 || fType == 4) {
      // Top level branch of a TClonesArray.
      Int_t *n = (Int_t*) fAddress;
      b >> n[0];
      if ((n[0] < 0) || (n[0] > fMaximum)) {
         if (IsMissingCollection()) {
            n[0] = 0;
            b.SetBufferOffset(b.Length() - sizeof(n));
         } else {
            Error("ReadLeaves", "Incorrect size read for the container in %s\nThe size read is %d when the maximum is %d\nThe size is reset to 0 for this entry (%lld)", GetName(), n[0], fMaximum, GetReadEntry());
            n[0] = 0;
         }
      }
      fNdata = n[0];
      if (fType == 4)   {
         Int_t nbranches = fBranches.GetEntriesFast();
         switch(fSTLtype) {
            case TClassEdit::kSet:
            case TClassEdit::kMultiSet:
            case TClassEdit::kMap:
            case TClassEdit::kMultiMap:
               for (Int_t i=0; i<nbranches; i++) {
                  TBranch *branch = (TBranch*)fBranches[i];
                  Int_t nb = branch->GetEntry(GetReadEntry(), 1);
                  if (nb < 0) {
                     break;
                  }
               }
               break;
            default:
               break;
         }
      }
      return;
   } else if (fType == 31 || fType == 41) {    // sub branch of a TClonesArray
      fNdata = fBranchCount->GetNdata();
      Int_t atype = fStreamerType;
      // FIXME: This should probably be > 59 instead.
      if (atype > 54) return;
      if (!fAddress) {
         return;
      }
      Int_t n = fNdata;
      if (atype>40) {
         atype -= 40;
         if (!fBranchCount2) return;
         const char *len_where = (char*)fBranchCount2->fAddress;
         if (!len_where) return;
         Int_t len_atype = fBranchCount2->fStreamerType;
         Int_t length;
         Int_t k;
         Char_t isArray;
         for( k=0; k<n; k++) {
            char **where = &(((char**)fAddress)[k]);
            delete [] *where;
            *where = 0;
            switch(len_atype) {
               case  1:  {length = ((Char_t*)   len_where)[k]; break;}
               case  2:  {length = ((Short_t*)  len_where)[k]; break;}
               case  3:  {length = ((Int_t*)    len_where)[k]; break;}
               case  4:  {length = ((Long_t*)   len_where)[k]; break;}
                  //case  5:  {length = ((Float_t*) len_where)[k]; break;}
               case  6:  {length = ((Int_t*)    len_where)[k]; break;}
                  //case  8:  {length = ((Double_t*)len_where)[k]; break;}
               case 11:  {length = ((UChar_t*)  len_where)[k]; break;}
               case 12:  {length = ((UShort_t*) len_where)[k]; break;}
               case 13:  {length = ((UInt_t*)   len_where)[k]; break;}
               case 14:  {length = ((ULong_t*)  len_where)[k]; break;}
               case 15:  {length = ((UInt_t*)   len_where)[k]; break;}
               case 16:  {length = ((Long64_t*) len_where)[k]; break;}
               case 17:  {length = ((ULong64_t*)len_where)[k]; break;}
               case 18:  {length = ((Bool_t*)   len_where)[k]; break;}
               default: continue;
            }
            b >> isArray;
            if (length <= 0)  continue;
            if (isArray == 0) continue;
            switch (atype) {
               case  1:  {*where=new char[sizeof(Char_t)*length]; b.ReadFastArray((Char_t*) *where, length); break;}
               case  2:  {*where=new char[sizeof(Short_t)*length]; b.ReadFastArray((Short_t*) *where, length); break;}
               case  3:  {*where=new char[sizeof(Int_t)*length]; b.ReadFastArray((Int_t*)   *where, length); break;}
               case  4:  {*where=new char[sizeof(Long_t)*length]; b.ReadFastArray((Long_t*)  *where, length); break;}
               case  5:  {*where=new char[sizeof(Float_t)*length]; b.ReadFastArray((Float_t*) *where, length); break;}
               case  6:  {*where=new char[sizeof(Int_t)*length]; b.ReadFastArray((Int_t*)   *where, length); break;}
               case  8:  {*where=new char[sizeof(Double_t)*length]; b.ReadFastArray((Double_t*)*where, length); break;}
               case 11:  {*where=new char[sizeof(UChar_t)*length]; b.ReadFastArray((UChar_t*) *where, length); break;}
               case 12:  {*where=new char[sizeof(UShort_t)*length]; b.ReadFastArray((UShort_t*)*where, length); break;}
               case 13:  {*where=new char[sizeof(UInt_t)*length]; b.ReadFastArray((UInt_t*)  *where, length); break;}
               case 14:  {*where=new char[sizeof(ULong_t)*length]; b.ReadFastArray((ULong_t*) *where, length); break;}
               case 15:  {*where=new char[sizeof(UInt_t)*length]; b.ReadFastArray((UInt_t*)  *where, length); break;}
               case 16:  {*where=new char[sizeof(Long64_t)*length]; b.ReadFastArray((Long64_t*)  *where, length); break;}
               case 17:  {*where=new char[sizeof(ULong64_t)*length]; b.ReadFastArray((ULong64_t*)*where, length); break;}
               case 18:  {*where=new char[sizeof(Bool_t)*length]; b.ReadFastArray((Bool_t*) *where, length); break;}
            }
         }
         return;
      }
      if (atype > 20) {
         atype -= 20;
         TLeafElement *leaf = (TLeafElement*)fLeaves.UncheckedAt(0);
         n *= leaf->GetLenStatic();
      }
      switch (atype) {
         case  1:  {b.ReadFastArray((Char_t*)  fAddress, n); break;}
         case  2:  {b.ReadFastArray((Short_t*) fAddress, n); break;}
         case  3:  {b.ReadFastArray((Int_t*)   fAddress, n); break;}
         case  4:  {b.ReadFastArray((Long_t*)  fAddress, n); break;}
         case  5:  {b.ReadFastArray((Float_t*) fAddress, n); break;}
         case  6:  {b.ReadFastArray((Int_t*)   fAddress, n); break;}
         case  8:  {b.ReadFastArray((Double_t*)fAddress, n); break;}
         case 11:  {b.ReadFastArray((UChar_t*) fAddress, n); break;}
         case 12:  {b.ReadFastArray((UShort_t*)fAddress, n); break;}
         case 13:  {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
         case 14:  {b.ReadFastArray((ULong_t*) fAddress, n); break;}
         case 15:  {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
         case 16:  {b.ReadFastArray((Long64_t*)fAddress, n); break;}
         case 17:  {b.ReadFastArray((ULong64_t*)fAddress, n); break;}
         case 18:  {b.ReadFastArray((Bool_t*)  fAddress, n); break;}
         case  9:  {
            TVirtualStreamerInfo* si = GetInfoImp();
            TStreamerElement* se = (TStreamerElement*) si->GetElems()[fID];
            Double_t *xx = (Double_t*) fAddress;
            for (Int_t ii=0;ii<n;ii++) {
               b.ReadDouble32(&(xx[ii]),se);
            }
            break;
         }
         case  19:  {
            TVirtualStreamerInfo* si = GetInfoImp();
            TStreamerElement* se = (TStreamerElement*) si->GetElems()[fID];
            Float_t *xx = (Float_t*) fAddress;
            for (Int_t ii=0;ii<n;ii++) {
               b.ReadFloat16(&(xx[ii]),se);
            }
            break;
         }
      }
      return;
   } else if (fType <= 2) {     // branch in split mode
      // FIXME: This should probably be < 60 instead.
      if (fStreamerType > 40 && fStreamerType < 55) {
         Int_t atype = fStreamerType - 40;
         Int_t n;
         if (fBranchCount==0) {
            // Missing fBranchCount.  let's attempts to recover.

            TString countname( GetName() );
            Ssiz_t dot = countname.Last('.');
            if (dot>=0) {
               countname.Remove(dot+1);
            } else {
               countname = "";
            }
            TString counter( GetTitle() );
            Ssiz_t loc = counter.Last('[');
            if (loc>=0) {
               counter.Remove(0,loc+1);
            }
            loc = counter.Last(']');
            if (loc>=0) {
               counter.Remove(loc);
            }
            countname += counter;
            SetBranchCount((TBranchElement *)fTree->GetBranch(countname));
         }
         if (fBranchCount) {
            n = (Int_t)fBranchCount->GetValue(0,0);
         } else {
            Warning("ReadLeaves","Missing fBranchCount for %s.  Data will not be read correctly by the MakeClass mode.",GetName());
            n = 0;
         }
         fNdata = n;
         Char_t isArray;
         b >> isArray;
         switch (atype) {
            case  1:  {b.ReadFastArray((Char_t*)  fAddress, n); break;}
            case  2:  {b.ReadFastArray((Short_t*) fAddress, n); break;}
            case  3:  {b.ReadFastArray((Int_t*)   fAddress, n); break;}
            case  4:  {b.ReadFastArray((Long_t*)  fAddress, n); break;}
            case  5:  {b.ReadFastArray((Float_t*) fAddress, n); break;}
            case  6:  {b.ReadFastArray((Int_t*)   fAddress, n); break;}
            case  8:  {b.ReadFastArray((Double_t*)fAddress, n); break;}
            case 11:  {b.ReadFastArray((UChar_t*) fAddress, n); break;}
            case 12:  {b.ReadFastArray((UShort_t*)fAddress, n); break;}
            case 13:  {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
            case 14:  {b.ReadFastArray((ULong_t*) fAddress, n); break;}
            case 15:  {b.ReadFastArray((UInt_t*)  fAddress, n); break;}
            case 16:  {b.ReadFastArray((Long64_t*) fAddress, n); break;}
            case 17:  {b.ReadFastArray((ULong64_t*)fAddress, n); break;}
            case 18:  {b.ReadFastArray((Bool_t*)   fAddress, n); break;}
            case  9:  {
               TVirtualStreamerInfo* si = GetInfoImp();
               TStreamerElement* se = (TStreamerElement*) si->GetElems()[fID];
               Double_t *xx = (Double_t*) fAddress;
               for (Int_t ii=0;ii<n;ii++) {
                  b.ReadDouble32(&(xx[ii]),se);
               }
               break;
            }
            case  19:  {
               TVirtualStreamerInfo* si = GetInfoImp();
               TStreamerElement* se = (TStreamerElement*) si->GetElems()[fID];
               Float_t *xx = (Float_t*) fAddress;
               for (Int_t ii=0;ii<n;ii++) {
                  b.ReadFloat16(&(xx[ii]),se);
               }
               break;
            }
         }
      } else {
         fNdata = 1;
         if (fAddress) {
            if (fType<0) {
               // Non TObject, Non collection classes with a custom streamer.

               // if (fObject)
               fBranchClass->Streamer(fObject,b);
            } else {
               GetInfoImp()->ReadBuffer(b, (char**) &fObject, fID);
            }
            if (fStreamerType == TVirtualStreamerInfo::kCounter) {
               fNdata = (Int_t) GetValue(0, 0);
            }
         } else {
            fNdata = 0;
         }
      }
      return;
   }
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesCollection(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // Case of a collection (fType == 4).

   ValidateAddress();
   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);

   // STL container master branch (has only the number of elements).
   Int_t n;
   b >> n;
   if ((n < 0) || (n > fMaximum)) {
      if (IsMissingCollection()) {
         n = 0;
         b.SetBufferOffset(b.Length()-sizeof(n));
      } else {
         Error("ReadLeaves", "Incorrect size read for the container in %s\n\tThe size read is %d while the maximum is %d\n\tThe size is reset to 0 for this entry (%lld)", GetName(), n, fMaximum, GetReadEntry());
         n = 0;
      }
   }
   fNdata = n;
   if (!fObject) {
      return;
   }
   // Note: Proxy-helper needs to "embrace" the entire
   //       streaming of this STL container if the container
   //       is a set/multiset/map/multimap (what we do not
   //       know here).
   //       For vector/list/deque Allocate == Resize
   //                         and Commit   == noop.
   // TODO: Exception safety a la TPushPop
   TVirtualCollectionProxy* proxy = GetCollectionProxy();
   TVirtualCollectionProxy::TPushPop helper(proxy, fObject);
   void* alternate = proxy->Allocate(fNdata, true);
   if(fSTLtype != TClassEdit::kVector && proxy->HasPointers() && fSplitLevel > TTree::kSplitCollectionOfPointers ) {
      fPtrIterators->CreateIterators(alternate);
   } else {
      fIterators->CreateIterators(alternate);
   }

   Int_t nbranches = fBranches.GetEntriesFast();
   switch (fSTLtype) {
      case TClassEdit::kSet:
      case TClassEdit::kMultiSet:
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
         for (Int_t i = 0; i < nbranches; ++i) {
            TBranch *branch = (TBranch*) fBranches[i];
            Int_t nb = branch->GetEntry(GetReadEntry(), 1);
            if (nb < 0) {
               // Give up on i/o failure.
               // FIXME: We need an error message here.
               break;
            }
         }
         break;
      default:
         break;
   }
   //------------------------------------------------------------------------
   // We have split this stuff, so we need to create the the pointers
   //-----------------------------------------------------------------------
   if( proxy->HasPointers() && fSplitLevel > TTree::kSplitCollectionOfPointers )
   {
      TClass *elClass = proxy->GetValueClass();

      //--------------------------------------------------------------------
      // The allocation is done in this strange way because ReadLeaves
      // is being called many times by TTreeFormula!!!
      //--------------------------------------------------------------------
      Int_t i = 0;
      if( !fNdata || *(void**)proxy->At( 0 ) != 0 )
         i = fNdata;

      for( ; i < fNdata; ++i )
      {
         void **el = (void**)proxy->At( i );
         // coverity[dereference] since this is a member streaming action by definition the collection contains objects and elClass is not null.
         *el = elClass->New();
      }
   }

   proxy->Commit(alternate);
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesCollectionSplitPtrMember(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // Case of a data member within a collection (fType == 41).

   ValidateAddress();
   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);

   // STL container sub-branch (contains the elements).
   fNdata = fBranchCount->GetNdata();
   if (!fNdata || !fObject) {
      return;
   }
   TStreamerInfo *info = GetInfoImp();
   if (info == 0) return;

   TVirtualCollectionProxy *proxy = GetCollectionProxy();
   TVirtualCollectionProxy::TPushPop helper(proxy, fObject);

   // R__ASSERT(0);
   TVirtualCollectionPtrIterators *iter = fBranchCount->fPtrIterators;
   b.ReadSequence(*fReadActionSequence,iter->fBegin,iter->fEnd);

   //   char **arr = (char **)proxy->At(0);
   //   char **end = arr + proxy->Size();
   //   fReadActionSequence->ReadBufferVecPtr(b,arr,end);

   //   info->ReadBufferSTLPtrs(b, proxy, fNdata, fID, fOffset);
   //   for(UInt_t ii=0; ii < fIDs.size(); ++ii) {
   //      info->ReadBufferSTLPtrs(b, proxy, fNdata, fIDs[ii], fOffset);
   //   }
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesCollectionSplitVectorPtrMember(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // Case of a data member within a collection (fType == 41).

   ValidateAddress();
   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);

   // STL container sub-branch (contains the elements).
   fNdata = fBranchCount->GetNdata();
   if (!fNdata || !fObject) {
      return;
   }
   TStreamerInfo *info = GetInfoImp();
   if (info == 0) return;

   TVirtualCollectionProxy *proxy = GetCollectionProxy();
   TVirtualCollectionProxy::TPushPop helper(proxy, fObject);


   TVirtualCollectionIterators *iter = fBranchCount->fIterators;
   b.ReadSequenceVecPtr(*fReadActionSequence,iter->fBegin,iter->fEnd);
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesCollectionMember(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // Case of a data member within a collection (fType == 41).

   ValidateAddress();
   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);

   // STL container sub-branch (contains the elements).
   fNdata = fBranchCount->GetNdata();
   if (!fNdata || !fObject) {
      return;
   }
   TStreamerInfo *info = GetInfoImp();
   if (info == 0) return;
   // Since info is not null, fReadActionSequence is not null either.

   // Still calling PushPop for the legacy entries.
   TVirtualCollectionProxy *proxy = GetCollectionProxy();
   TVirtualCollectionProxy::TPushPop helper(proxy, fObject);

   TVirtualCollectionIterators *iter = fBranchCount->fIterators;
   b.ReadSequence(*fReadActionSequence,iter->fBegin,iter->fEnd);
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesClones(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // Case of a TClonesArray (fType == 3).

   ValidateAddress();
   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);

   // TClonesArray master branch (has only the number of elements).
   Int_t n;
   b >> n;
   if ((n < 0) || (n > fMaximum)) {
      if (IsMissingCollection()) {
         n = 0;
         b.SetBufferOffset(b.Length()-sizeof(n));
      } else {
         Error("ReadLeaves", "Incorrect size read for the container in %s\n\tThe size read is %d while the maximum is %d\n\tThe size is reset to 0 for this entry (%lld)", GetName(), n, fMaximum, GetReadEntry());
         n = 0;
      }
   }
   fNdata = n;
   TClonesArray* clones = (TClonesArray*) fObject;
   if (!clones) {
      return;
   }
   if (clones->IsZombie()) {
      return;
   }
   clones->Clear();
   clones->ExpandCreateFast(fNdata);
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesClonesMember(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // Case of a data member within a TClonesArray (fType == 31).

   ValidateAddress();

   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);

   // TClonesArray sub-branch (contains the elements).
   fNdata = fBranchCount->GetNdata();
   TClonesArray* clones = (TClonesArray*) fObject;
   if (!clones) {
      return;
   }
   if (clones->IsZombie()) {
      return;
   }
   TStreamerInfo *info = GetInfoImp();
   if (info==0) return;
   // Since info is not null, fReadActionSequence is not null either.

   char **arr = (char **)clones->GetObjectRef(0);
   char **end = arr + fNdata;
   b.ReadSequenceVecPtr(*fReadActionSequence,arr,end);
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesMember(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // For split-class branch, base class branch, data member branch, or top-level branch.
   // which do not have a branch count and are not a counter.

   R__ASSERT(fBranchCount==0);
   R__ASSERT(fStreamerType != TVirtualStreamerInfo::kCounter);

   ValidateAddress();
   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);
   // If not a TClonesArray or STL container master branch
   // or sub-branch and branch inherits from tobject,
   // then register with the buffer so that pointers are
   // handled properly.
   if (TestBit(kBranchObject)) {
      b.MapObject((TObject*) fObject);
   } else if (TestBit(kBranchAny)) {
      b.MapObject(fObject, fBranchClass);
   }

   fNdata = 1;
   TStreamerInfo *info = GetInfoImp();
   if (!info) {
      return;
   }
   // Since info is not null, fReadActionSequence is not null either.
   b.ReadSequence(*fReadActionSequence, fObject);
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesMemberBranchCount(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // For split-class branch, base class branch, data member branch, or top-level branch.
   // which do have a branch count and are not a counter.

   R__ASSERT(fStreamerType != TVirtualStreamerInfo::kCounter);

   ValidateAddress();
   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);
   // If not a TClonesArray or STL container master branch
   // or sub-branch and branch inherits from tobject,
   // then register with the buffer so that pointers are
   // handled properly.
   if (TestBit(kBranchObject)) {
      b.MapObject((TObject*) fObject);
   } else if (TestBit(kBranchAny)) {
      b.MapObject(fObject, fBranchClass);
   }

   fNdata = (Int_t) fBranchCount->GetValue(0, 0);
   TStreamerInfo *info = GetInfoImp();
   if (!info) {
      return;
   }
   // Since info is not null, fReadActionSequence is not null either.
   b.ReadSequence(*fReadActionSequence, fObject);
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesMemberCounter(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // For split-class branch, base class branch, data member branch, or top-level branch.
   // which do not have a branch count and are a counter.

   ValidateAddress();
   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);

   // If not a TClonesArray or STL container master branch
   // or sub-branch and branch inherits from tobject,
   // then register with the buffer so that pointers are
   // handled properly.
   if (TestBit(kBranchObject)) {
      b.MapObject((TObject*) fObject);
   } else if (TestBit(kBranchAny)) {
      b.MapObject(fObject, fBranchClass);
   }

   TStreamerInfo *info = GetInfoImp();
   if (!info) {
      return;
   }
   // Since info is not null, fReadActionSequence is not null either.
   b.ReadSequence(*fReadActionSequence, fObject);
   fNdata = (Int_t) GetValue(0, 0);
}

//______________________________________________________________________________
void TBranchElement::ReadLeavesCustomStreamer(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.
   // Non TObject, Non collection classes with a custom streamer.

   ValidateAddress();
   if (fObject == 0)
   {
      // We have nowhere to copy the data (probably because the data member was
      // 'dropped' from the current schema) so let's no copy it in a random place.
      return;
   }

   R__PushCache onfileObject(((TBufferFile&)b),fOnfileObject);
   fBranchClass->Streamer(fObject,b);
}

//______________________________________________________________________________
void TBranchElement::ReleaseObject()
{
   // -- Delete any object we may have allocated on a previous call to SetAddress.

   if (fObject && TestBit(kDeleteObject)) {
      if (IsAutoDelete() && fAddress != (char*)&fObject) {
         *((char**) fAddress) = 0;
      }
      ResetBit(kDeleteObject);
      if (fType == 3) {
         // -- We are a TClonesArray master branch.
         TClonesArray::Class()->Destructor(fObject);
         fObject = 0;
         if ((fStreamerType == TVirtualStreamerInfo::kObjectp) ||
             (fStreamerType == TVirtualStreamerInfo::kObjectP)) {
            // -- We are a pointer to a TClonesArray.
            // We must zero the pointer in the object.
            *((char**) fAddress) = 0;
         }
      } else if (fType == 4) {
         // -- We are an STL container master branch.
         TVirtualCollectionProxy* proxy = GetCollectionProxy();
         if (!proxy) {
            Warning("ReleaseObject", "Cannot delete allocated STL container because I do not have a proxy!  branch: %s", GetName());
            fObject = 0;
         } else {
            proxy->Destructor(fObject);
            fObject = 0;
         }
         if (fStreamerType == TVirtualStreamerInfo::kSTLp) {
            // -- We are a pointer to an STL container.
            // We must zero the pointer in the object.
            *((char**) fAddress) = 0;
         }
      } else {
         // We are *not* a TClonesArray master branch and we are *not* an STL container master branch.
         TClass* cl = fBranchClass.GetClass();
         if (!cl) {
            Warning("ReleaseObject", "Cannot delete allocated object because I cannot instantiate a TClass object for its class!  branch: '%s' class: '%s'", GetName(), fBranchClass.GetClassName());
            fObject = 0;
         } else {
            cl->Destructor(fObject);
            fObject = 0;
         }
      }
   }
}

//______________________________________________________________________________
void TBranchElement::Reset(Option_t* option)
{
   // -- Reset a Branch.
   //
   // Existing i/o buffers are deleted.
   // Entries, max and min are reset.
   //

   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i) {
      TBranch* branch = (TBranch*) fBranches[i];
      branch->Reset(option);
   }
   fBranchID = -1;
   TBranch::Reset(option);
}

//______________________________________________________________________________
void TBranchElement::ResetAddress()
{
   // Set branch address to zero and free all allocated memory.

   for (Int_t i = 0; i < fNleaves; ++i) {
      TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(i);
      //if (leaf) leaf->SetAddress(0);
      leaf->SetAddress(0);
   }

   // Note: We *must* do the sub-branches first, otherwise
   //       we may delete the object containing the sub-branches
   //       before giving them a chance to cleanup.
   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i)  {
      TBranch* br = (TBranch*) fBranches[i];
      if (br) br->ResetAddress();
   }

   //
   // SetAddress may have allocated an object.
   //

   ReleaseObject();

   ResetBit(kAddressSet);
   fAddress = 0;
   fObject = 0;
}

//______________________________________________________________________________
void TBranchElement::ResetDeleteObject()
{
   // -- Release ownership of any allocated objects.
   //
   // Note: This interface was added so that clone trees could
   //       be told they do not own the allocated objects.

   ResetBit(kDeleteObject);
   Int_t nb = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nb; ++i)  {
      TBranch* br = (TBranch*) fBranches[i];
      if (br->InheritsFrom(TBranchElement::Class())) {
         ((TBranchElement*) br)->ResetDeleteObject();
      }
   }
}

//______________________________________________________________________________
void TBranchElement::SetAddress(void* addr)
{
   // -- Point this branch at an object.
   //
   // For a sub-branch, addr is a pointer to the branch object.
   //
   // For a top-level branch the meaning of addr is as follows:
   //
   // If addr is zero, then we allocate a branch object
   // internally and the branch is the owner of the allocated
   // object, not the caller.  However the caller may obtain
   // a pointer to the branch object with GetObject().
   //
   // Example:
   //
   //    branch->SetAddress(0);
   //    Event* event = branch->GetObject();
   //    ... Do some work.
   //
   // If addr is not zero, but the pointer addr points at is
   // zero, then we allocate a branch object and set the passed
   // pointer to point at the allocated object.  The caller
   // owns the allocated object and is responsible for deleting
   // it when it is no longer needed.
   //
   // Example:
   //
   //    Event* event = 0;
   //    branch->SetAddress(&event);
   //    ... Do some work.
   //    delete event;
   //    event = 0;
   //
   // If addr is not zero and the pointer addr points at is
   // also not zero, then the caller has allocated a branch
   // object and is asking us to use it.  The caller owns it
   // and must delete it when it is no longer needed.
   //
   // Example:
   //
   //    Event* event = new Event();
   //    branch->SetAddress(&event);
   //    ... Do some work.
   //    delete event;
   //    event = 0;
   //
   // These rules affect users of TTree::Branch(),
   // TTree::SetBranchAddress(), and TChain::SetBranchAddress()
   // as well because those routines call this one.
   //
   // An example of a tree with branches with objects allocated
   // and owned by us:
   //
   //    TFile* f1 = new TFile("myfile_original.root");
   //    TTree* t1 = (TTree*) f->Get("MyTree");
   //    TFile* f2 = new TFile("myfile_copy.root", "recreate");
   //    TTree* t2 = t1->Clone(0);
   //    for (Int_t i = 0; i < 10; ++i) {
   //       t1->GetEntry(i);
   //       t2->Fill();
   //    }
   //    t2->Write()
   //    delete f2;
   //    f2 = 0;
   //    delete f1;
   //    f1 = 0;
   //
   // An example of a branch with an object allocated by us,
   // but owned by the caller:
   //
   //    TFile* f = new TFile("myfile.root", "recreate");
   //    TTree* t = new TTree("t", "A test tree.")
   //    Event* event = 0;
   //    TBranchElement* br = t->Branch("event.", &event);
   //    for (Int_t i = 0; i < 10; ++i) {
   //       ... Fill event with meaningful data in some way.
   //       t->Fill();
   //    }
   //    t->Write();
   //    delete event;
   //    event = 0;
   //    delete f;
   //    f = 0;
   //
   // Notice that the only difference between this example
   // and the following example is that the event pointer
   // is zero when the branch is created.
   //
   // An example of a branch with an object allocated and
   // owned by the caller:
   //
   //    TFile* f = new TFile("myfile.root", "recreate");
   //    TTree* t = new TTree("t", "A test tree.")
   //    Event* event = new Event();
   //    TBranchElement* br = t->Branch("event.", &event);
   //    for (Int_t i = 0; i < 10; ++i) {
   //       ... Fill event with meaningful data in some way.
   //       t->Fill();
   //    }
   //    t->Write();
   //    delete event;
   //    event = 0;
   //    delete f;
   //    f = 0;
   //
   // If AutoDelete is on (see TBranch::SetAutoDelete),
   // the top level objet will be deleted and recreate
   // each time an entry is read, whether or not the
   // TTree owns the object.
   //

   //
   //  Don't bother if we are disabled.
   //

   if (TestBit(kDoNotProcess)) {
      return;
   }

   //
   //  FIXME: When would this happen?
   //

   if (fType < -1) {
      return;
   }

   //
   //  Special case when called from code generated by TTree::MakeClass.
   //

   if (Long_t(addr) == -1) {
      // FIXME: Do we have to release an object here?
      // ReleaseObject();
      fAddress = (char*) -1;
      fObject = (char*) -1;
      ResetBit(kDeleteObject);
      return;
   }

   //
   //  Reset last read entry number, we have a new user object now.
   //

   fReadEntry = -1;

   //
   // Make sure our branch class is instantiated.
   //
   TClass* clOfBranch = fBranchClass.GetClass();
   if( fTargetClass.GetClassName()[0] ) {
      clOfBranch = fTargetClass;
   }

   //
   // Try to build the streamer info.
   //

   TVirtualStreamerInfo *info = GetInfoImp();

   // FIXME: Warn about failure to get the streamer info here?

   //
   // We may have allocated an object last time we were called.
   //

   if (fObject && TestBit(kDeleteObject)){
      ReleaseObject();
   }

   //
   //  Remember the pointer to the pointer to our object.
   //

   fAddress = (char*) addr;
   if (fAddress != (char*)(&fObject)) {
      fObject = 0;
   }
   ResetBit(kDeleteObject);

   //
   //  Do special stuff if we got called from a MakeClass class.
   //  Allow sub-branches to have independently set addresses.
   //

   if (fTree->GetMakeClass()) {
      if (fID > -1) {
         // We are *not* a top-level branch.
         if (!info) {
            // No streamer info, give up.
            // FIXME: We should have an error message here.
            fObject = fAddress;
         } else {
            // Compensate for the fact that the i/o routines
            // will add the streamer offset to the address.
            fObject = fAddress - info->GetOffsets()[fID];
         }
         return;
      }
   }

   //
   //  Check whether the container type is still the same
   //  to support schema evolution; what is written on the file
   //  may no longer match the class code which is loaded.
   //

   if (fType == 3) {
      // split TClonesArray, counter/master branch.
      TClass* clm = TClass::GetClass(GetClonesName());
      if (clm) {
         // In case clm derives from an abstract class.
         clm->BuildRealData();
         clm->GetStreamerInfo();
      }
      TClass* newType = GetCurrentClass();
      if (newType && (newType != TClonesArray::Class())) {
         // The data type of the container has changed.
         //
         // Let's check if it is a compatible type:
         Bool_t matched = kFALSE;
         if (newType->GetCollectionProxy()) {
            TClass *content = newType->GetCollectionProxy()->GetValueClass();
            if (clm == content) {
               matched = kTRUE;
            } else {
               Warning("SetAddress", "The type of %s was changed from TClonesArray to %s but the content do not match (was %s)!", GetName(), newType->GetName(), GetClonesName());
            }
         } else {
            Warning("SetAddress", "The type of the %s was changed from TClonesArray to %s but we do not have a TVirtualCollectionProxy for that container type!", GetName(), newType->GetName());
         }
         if (matched) {
            // Change from 3/31 to 4/41
            SetType(4);
            // Set the proxy.
            fSTLtype = TMath::Abs(TClassEdit::IsSTLCont(newType->GetName()));
            fCollProxy = newType->GetCollectionProxy()->Generate();

            SwitchContainer(GetListOfBranches());
            SetReadLeavesPtr();

            if(fSTLtype != TClassEdit::kVector && fCollProxy->HasPointers() && fSplitLevel > TTree::kSplitCollectionOfPointers ) {
               fPtrIterators = new TVirtualCollectionPtrIterators(fCollProxy);
            } else {
               fIterators = new TVirtualCollectionIterators(fCollProxy);
            }
         } else {
            // FIXME: Must maintain fObject here as well.
            fAddress = 0;
         }
      }
   } else if (fType == 4) {
      // split STL container, counter/master branch.
      TClass* newType = GetCurrentClass();
      if (newType && (newType != GetCollectionProxy()->GetCollectionClass())) {
         // Let's check if it is a compatible type:
         TVirtualCollectionProxy* newProxy = newType->GetCollectionProxy();
         TVirtualCollectionProxy* oldProxy = GetCollectionProxy();
         if (newProxy && (oldProxy->GetValueClass() == newProxy->GetValueClass()) && ((!oldProxy->GetValueClass() && (oldProxy->GetType() == newProxy->GetType())) || (oldProxy->GetValueClass() && (oldProxy->HasPointers() == newProxy->HasPointers())))) {
            if (fSTLtype == TClassEdit::kNotSTL) {
               fSTLtype = TMath::Abs(TClassEdit::IsSTLCont(newType->GetName()));
            }
            delete fCollProxy;
            Int_t nbranches = GetListOfBranches()->GetEntries();
            fCollProxy = newType->GetCollectionProxy()->Generate();
            for (Int_t i = 0; i < nbranches; ++i) {
               TBranchElement* br = (TBranchElement*) GetListOfBranches()->UncheckedAt(i);
               br->fCollProxy = 0;
               if (br->fReadActionSequence) {
                  br->SetReadActionSequence();
               }
            }
            SetReadActionSequence();
            SetReadLeavesPtr();
            delete fIterators;
            delete fPtrIterators;
            if(fSTLtype != TClassEdit::kVector && fCollProxy->HasPointers() && fSplitLevel > TTree::kSplitCollectionOfPointers ) {
               fPtrIterators = new TVirtualCollectionPtrIterators(fCollProxy);
            } else {
               fIterators = new TVirtualCollectionIterators(fCollProxy);
            }
         }
         else if ((newType == TClonesArray::Class()) && (oldProxy->GetValueClass() && !oldProxy->HasPointers() && oldProxy->GetValueClass()->InheritsFrom(TObject::Class())))
         {
            // The new collection and the old collection are not compatible,
            // we cannot use the new collection to read the data.
            // Actually we could check if the new collection is a
            // compatible ROOT collection.

            // We cannot insure that the TClonesArray is set for the
            // proper class (oldProxy->GetValueClass()), so we assume that
            // the transformation was done properly by the class designer.

            // Change from 4/41 to 3/31
            SetType(3);
            // Reset the proxy.
            fSTLtype = kNone;
            switch(fStreamerType) {
               case TVirtualStreamerInfo::kAny:
               case TVirtualStreamerInfo::kSTL:
                  fStreamerType = TVirtualStreamerInfo::kObject;
                  break;
               case TVirtualStreamerInfo::kAnyp:
               case TVirtualStreamerInfo::kSTLp:
                  fStreamerType = TVirtualStreamerInfo::kObjectp;
                  break;
               case TVirtualStreamerInfo::kAnyP:
                  fStreamerType = TVirtualStreamerInfo::kObjectP;
                  break;
            }
            fClonesName = oldProxy->GetValueClass()->GetName();
            delete fCollProxy;
            fCollProxy = 0;
            TClass* clm = TClass::GetClass(GetClonesName());
            if (clm) {
               clm->BuildRealData(); //just in case clm derives from an abstract class
               clm->GetStreamerInfo();
            }
            SwitchContainer(GetListOfBranches());
            SetReadLeavesPtr();
            delete fIterators;
            fIterators = 0;
            delete fPtrIterators;
            fPtrIterators =0;
         } else {
            // FIXME: We must maintain fObject here as well.
            fAddress = 0;
         }
      } else {
         if (!fIterators && !fPtrIterators) {
            if(fSTLtype != TClassEdit::kVector && GetCollectionProxy()->HasPointers() && fSplitLevel > TTree::kSplitCollectionOfPointers ) {
               fPtrIterators = new TVirtualCollectionPtrIterators(GetCollectionProxy());
            } else {
               fIterators = new TVirtualCollectionIterators(GetCollectionProxy());
            }
         }
      }
   }

   //
   //  Establish the semantics of fObject and fAddress.
   //
   //  Top-level branch:
   //       fObject is a ptr to the object,
   //       fAddress is a ptr to a pointer to the object.
   //
   //  Sub-branch:
   //       fObject is a ptr to the object,
   //       fAddress is the same as fObject.
   //
   //
   //  There are special cases for TClonesArray and STL containers.
   //  If there is no user-provided object, we allocate one.  We must
   //  also initialize any STL container proxy.
   //

   if (fType == 3) {
      // -- We are a TClonesArray master branch.
      if (fAddress) {
         // -- We have been given a non-zero address, allocate if necessary.
         if (fStreamerType == TVirtualStreamerInfo::kObject) {
            // -- We are *not* a top-level branch and we are *not* a pointer to a TClonesArray.
            // Case of an embedded TClonesArray.
            fObject = fAddress;
            // Check if it has already been properly built.
            TClonesArray* clones = (TClonesArray*) fObject;
            if (!clones->GetClass()) {
               new(fObject) TClonesArray(GetClonesName());
            }
         } else {
            // -- We are either a top-level branch or we are a subbranch which is a pointer to a TClonesArray.
            // Streamer type should be -1 (for a top-level branch) or kObject(p|P) here.
            if ((fStreamerType != -1) &&
                (fStreamerType != TVirtualStreamerInfo::kObjectp) &&
                (fStreamerType != TVirtualStreamerInfo::kObjectP)) {
               Error("SetAddress", "TClonesArray with fStreamerType: %d", fStreamerType);
            } else if (fStreamerType == -1) {
               // -- We are a top-level branch.
               TClonesArray** pp = (TClonesArray**) fAddress;
               if (!*pp) {
                  // -- Caller wants us to allocate the clones array, but he will own it.
                  *pp = new TClonesArray(GetClonesName());
               }
               fObject = (char*) *pp;
            } else {
               // -- We are a pointer to a TClonesArray.
               // Note: We do this so that the default constructor,
               //       or the i/o constructor can be lazy.
               TClonesArray** pp = (TClonesArray**) fAddress;
               if (!*pp) {
                  // -- Caller wants us to allocate the clones array, but he will own it.
                  *pp = new TClonesArray(GetClonesName());
               }
               fObject = (char*) *pp;
            }
         }
      } else {
         // -- We have been given a zero address, allocate for top-level only.
         if (fStreamerType == TVirtualStreamerInfo::kObject) {
            // -- We are *not* a top-level branch and we are *not* a pointer to a TClonesArray.
            // Case of an embedded TClonesArray.
            Error("SetAddress", "Embedded TClonesArray given a zero address for branch '%s'", GetName());
         } else {
            // -- We are either a top-level branch or we are a subbranch which is a pointer to a TClonesArray.
            // Streamer type should be -1 (for a top-level branch) or kObject(p|P) here.
            if ((fStreamerType != -1) &&
                (fStreamerType != TVirtualStreamerInfo::kObjectp) &&
                (fStreamerType != TVirtualStreamerInfo::kObjectP)) {
               Error("SetAddress", "TClonesArray with fStreamerType: %d", fStreamerType);
            } else if (fStreamerType == -1) {
               // -- We are a top-level branch.
               // Idea: Consider making a zero address not allocate.
               SetBit(kDeleteObject);
               fObject = (char*) new TClonesArray(GetClonesName());
               fAddress = (char*) &fObject;
            } else {
               // -- We are a sub-branch which is a pointer to a TClonesArray.
               Error("SetAddress", "Embedded pointer to a TClonesArray given a zero address for branch '%s'", GetName());
            }
         }
      }
   } else if (fType == 4) {
      // -- We are an STL container master branch.
      //
      // Initialize fCollProxy.
      TVirtualCollectionProxy* proxy = GetCollectionProxy();
      if (fAddress) {
         // -- We have been given a non-zero address, allocate if necessary.
         if ((fStreamerType == TVirtualStreamerInfo::kObject) ||
             (fStreamerType == TVirtualStreamerInfo::kAny) ||
             (fStreamerType == TVirtualStreamerInfo::kSTL)) {
            // We are *not* a top-level branch and we are *not* a pointer to an STL container.
            // Case of an embedded STL container.
            // Note: We test for the kObject and kAny types to support
            //       the (unwise) choice of inheriting from an STL container.
            fObject = fAddress;
         } else {
            // We are either a top-level branch or subbranch which is a pointer to an STL container.
            // Streamer type should be -1 (for a top-level branch) or kSTLp here.
            if ((fStreamerType != -1) && (fStreamerType != TVirtualStreamerInfo::kSTLp)) {
               Error("SetAddress", "STL container with fStreamerType: %d", fStreamerType);
            } else if (fStreamerType == -1) {
               // -- We are a top-level branch.
               void** pp = (void**) fAddress;
               if (!*pp) {
                  // -- Caller wants us to allocate the STL container, but he will own it.
                  *pp = proxy->New();
                  if (!(*pp)) {
                     Error("SetAddress", "Failed to allocate STL container for branch '%s'", GetName());
                     // FIXME: Should we do this?  Lots of other code wants
                     //        fAddress to be zero if no fObject, but is
                     //        that a good thing?
                     fAddress = 0;
                  }
               }
               fObject = (char*) *pp;
            } else {
               // -- We are a pointer to an STL container.
               // Note: We do this so that the default constructor,
               //       or the i/o constructor can be lazy.
               void** pp = (void**) fAddress;
               if (!*pp) {
                  // -- Caller wants us to allocate the STL container, but he will own it.
                  *pp = proxy->New();
                  if (!(*pp)) {
                     Error("SetAddress", "Failed to allocate STL container for branch '%s'", GetName());
                     // FIXME: Should we do this?  Lots of other code wants
                     //        fAddress to be zero if no fObject, but is
                     //        that a good thing?
                     fAddress = 0;
                  }
               }
               fObject = (char*) *pp;
            }
         }
      } else {
         // -- We have been given a zero address, allocate for top-level only.
         if ((fStreamerType == TVirtualStreamerInfo::kObject) ||
             (fStreamerType == TVirtualStreamerInfo::kAny) ||
             (fStreamerType == TVirtualStreamerInfo::kSTL)) {
            // We are *not* a top-level branch and we are *not* a pointer to an STL container.
            // Case of an embedded STL container.
            // Note: We test for the kObject and kAny types to support
            //       the (unwise) choice of inheriting from an STL container.
            Error("SetAddress", "Embedded STL container given a zero address for branch '%s'", GetName());
         } else {
            // We are either a top-level branch or sub-branch which is a pointer to an STL container.
            // Streamer type should be -1 (for a top-level branch) or kSTLp here.
            if ((fStreamerType != -1) && (fStreamerType != TVirtualStreamerInfo::kSTLp)) {
               Error("SetAddress", "STL container with fStreamerType: %d", fStreamerType);
            } else if (fStreamerType == -1) {
               // -- We are a top-level branch, allocate.
               SetBit(kDeleteObject);
               fObject = (char*) proxy->New();
               if (fObject) {
                  fAddress = (char*) &fObject;
               } else {
                  Error("SetAddress", "Failed to allocate STL container for branch '%s'", GetName());
                  // FIXME: Should we do this?  Lots of other code wants
                  //        fAddress to be zero if no fObject, but is
                  //        that a good thing?
                  fAddress = 0;
               }
            } else {
               // -- We are a sub-branch which is a pointer to an STL container.
               Error("SetAddress", "Embedded pointer to an STL container given a zero address for branch '%s'", GetName());
            }
         }
      }
   } else if (fType == 41) {
      // -- We are an STL container sub-branch.
      // Initialize fCollProxy.
      GetCollectionProxy();
      // We are not at top-level branch.
      fObject = fAddress;
   } else if (fID < 0) {
      // -- We are a top-level branch.
      char** pp = (char**) fAddress;
      if (pp && *pp) {
         // -- Caller provided an i/o buffer for us to use.
         fObject = *pp;
      } else {
         // -- Caller did not provide an i/o buffer for us to use, we must make one for ourselves.
         if (clOfBranch) {
            if (!pp) {
               // -- Caller wants us to own the object.
               SetBit(kDeleteObject);
            }
            fObject = (char*) clOfBranch->New();
            if (pp) {
               *pp = fObject;
            } else {
               fAddress = (char*) &fObject;
            }
         } else {
            Error("SetAddress", "I have no TClass for branch %s, so I cannot allocate an I/O buffer!", GetName());
            if (pp) {
               fObject = 0;
               *pp = 0;
            }
         }
      }
   } else {
      // -- We are *not* a top-level branch.
      fObject = fAddress;
   }

   if (!info) {
      // FIXME: We need and error message here, no streamer info, so cannot set offsets.
      return;
   }

   // We do this only once because it depends only on
   // the type of our object, not on its address.
   if (!fInitOffsets) {
      InitializeOffsets();
   }

   // We are split, recurse down to our sub-branches.
   //
   // FIXME: This is a tail recursion, we burn stack.
   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i) {
      TBranch* abranch = (TBranch*) fBranches.UncheckedAt(i);
      // FIXME: This is a tail recursion!
      if (fBranchOffset[i] != TStreamerInfo::kMissing) {
         abranch->SetAddress(fObject + fBranchOffset[i]);
         abranch->SetBit(kAddressSet);
      } else {
         // When the member is missing, just leave the address alone
         // (since setting explicitly to 0 would trigger error/warning
         // messages).
         // abranch->SetAddress(0);
         abranch->SetBit(kAddressSet);
      }
   }
}

//______________________________________________________________________________
void TBranchElement::SetBasketSize(Int_t buffsize)
{
   // -- Reset the basket size for all sub-branches of this branch element.

   TBranch::SetBasketSize(buffsize);
   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i) {
      TBranch* branch = (TBranch*) fBranches[i];
      branch->SetBasketSize(fBasketSize);
   }
}

//______________________________________________________________________________
void TBranchElement::SetBranchCount(TBranchElement* brOfCounter)
{
   // -- Set the branch counter for this branch.

   fBranchCount = brOfCounter;
   if (fBranchCount==0) return;

   TLeafElement* leafOfCounter  = (TLeafElement*) brOfCounter->GetListOfLeaves()->At(0);
   TLeafElement* leaf = (TLeafElement*) GetListOfLeaves()->At(0);
   if (leafOfCounter && leaf) {
      leaf->SetLeafCount(leafOfCounter);
   } else {
      if (!leafOfCounter) {
         Warning("SetBranchCount", "Counter branch %s for branch %s has no leaves!", brOfCounter->GetName(), GetName());
      }
      if (!leaf) {
         Warning("SetBranchCount", "Branch %s has no leaves!", GetName());
      }
   }
}

//______________________________________________________________________________
Bool_t TBranchElement::SetMakeClass(Bool_t decomposeObj)
{
   // Set the branch in a mode where the object are decomposed
   // (Also known as MakeClass mode).
   // Return whether the setting was possible (it is not possible for
   // TBranch and TBranchObject).

   if (decomposeObj)
      SetBit(kDecomposedObj);   // Same as SetBit(kMakeClass)
   else
      ResetBit(kDecomposedObj);

   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i) {
      TBranchElement* branch = (TBranchElement*) fBranches[i];
      branch->SetMakeClass(decomposeObj);
   }
   SetReadLeavesPtr();

   return kTRUE;
}

//______________________________________________________________________________
void TBranchElement::SetObject(void* obj)
{
   // Set object this branch is pointing to.

   if (TestBit(kDoNotProcess)) {
      return;
   }
   fObject = (char*)obj;
   SetAddress( &fObject );
}

//______________________________________________________________________________
void TBranchElement::SetOffset(Int_t offset)
{
   // Set offset of the object (to which the data member represented by this
   // branch belongs) inside its containing object (if any).

   // We need to make sure that the Read and Write action's configuration
   // properly reflect this value.

   if (fReadActionSequence) {
      fReadActionSequence->AddToOffset(offset - fOffset);
   }
   fOffset = offset;
}

//______________________________________________________________________________
void TBranchElement::SetReadActionSequence()
{
   // Set the sequence of actions needed to read the data out of the buffer.

   if (fInfo == 0) {
      // We are called too soon.  We will be called again by InitInfo
      return;
   }

   // Get the action sequence we need to copy for reading.
   TStreamerInfoActions::TActionSequence *original = 0;
   TStreamerInfoActions::TActionSequence *transient = 0;
   if (fType == 41) {
      if( fSplitLevel >= TTree::kSplitCollectionOfPointers && fBranchCount->fSTLtype == TClassEdit::kVector) {
         original = fInfo->GetReadMemberWiseActions(kTRUE);
      } else {
         TVirtualStreamerInfo *info = GetInfoImp();
         if (GetParentClass() == info->GetClass()) {
            if( fTargetClass.GetClassName()[0] && fBranchClass != fTargetClass ) {
               original = GetCollectionProxy()->GetConversionReadMemberWiseActions(fBranchClass.GetClass(), fClassVersion);
            } else {
               original = GetCollectionProxy()->GetReadMemberWiseActions(fClassVersion);
            }
         } else {
            // Base class and embedded objects.

            transient = TStreamerInfoActions::TActionSequence::CreateReadMemberWiseActions(info,*GetCollectionProxy());
            original = transient;
         }
      }
   } else if (fType == 31) {
      original = fInfo->GetReadMemberWiseActions(kTRUE);
   } else if (0<=fType && fType<=2) {
      // Note: this still requires the ObjectWise sequence to not be optimized!
      original = fInfo->GetReadMemberWiseActions(kFALSE);
   }
   if (original) {
      fIDs.insert(fIDs.begin(),fID); // Include the main element in the sequence.
      if (fReadActionSequence) delete fReadActionSequence;
      fReadActionSequence = original->CreateSubSequence(fIDs,fOffset);
      fIDs.erase(fIDs.begin());
   }
   delete transient;
}

//______________________________________________________________________________
void TBranchElement::SetReadLeavesPtr()
{
   // Set the ReadLeaves pointer to execute the expected operations.

   if (TestBit(kDecomposedObj)) {
      fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesMakeClass;
   } else if (fType == 4) {
      fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesCollection;
   } else if (fType == 41) {
      if( fSplitLevel >= TTree::kSplitCollectionOfPointers ) {
         if (fBranchCount->fSTLtype == TClassEdit::kVector) {
            fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesCollectionSplitVectorPtrMember;
         } else {
            fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesCollectionSplitPtrMember;
         }
      } else {
         fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesCollectionMember;
      }
   } else if (fType == 3) {
      fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesClones;
   } else if (fType == 31) {
      fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesClonesMember;
   } else if (fType < 0) {
      fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesCustomStreamer;
   } else if (fType <=2) {
      // split-class branch, base class branch, data member branch, or top-level branch.
      if (fBranchCount) {
         fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesMemberBranchCount;
      } else if (fStreamerType == TVirtualStreamerInfo::kCounter) {
         fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesMemberCounter;
      } else {
         fReadLeaves = (ReadLeaves_t)&TBranchElement::ReadLeavesMember;
      }
   } else {
      Fatal("SetReadLeavePtr","Unexpected branch type %d for %s",fType,GetName());
   }

   SetReadActionSequence();
}

//______________________________________________________________________________
void TBranchElement::SetTargetClass(const char *name)
{
   // Set the name of the class of the in-memory object into which the data will
   // loaded.

   if (name == 0) return;

   if (strcmp(fTargetClass.GetClassName(),name) != 0 )
   {
      // We are changing target class, let's reset the meta information and
      // the sub-branches.

      fInfo = 0;
      fInit = kFALSE;
      fInitOffsets = kFALSE;
      delete fReadActionSequence;
      fReadActionSequence = 0;

      Int_t nbranches = fBranches.GetEntriesFast();
      for (Int_t i = 0; i < nbranches; ++i) {
         TBranchElement *sub = (TBranchElement*) fBranches[i];
         if (sub->fTargetClass == fTargetClass ) {
            sub->SetTargetClass(name);
         }
         if (sub->fParentClass == fTargetClass ) {
            sub->SetParentClass(TClass::GetClass(name));
         }
      }
      fTargetClass = name;
   }

}

//______________________________________________________________________________
void TBranchElement::SetupAddresses()
{
   // -- If the branch address is not set,  we set all addresses starting with
   // the top level parent branch.  This is required to be done in order for
   // GetOffset to be correct and for GetEntry to run.

   // Check to see if the user changed the branch address on us.
   ValidateAddress();

   if (fAddress || fTree->GetMakeClass()) {
      // -- Do nothing if already setup or if we are a MakeClass tree.
      return;
   }

   if (TestBit(kDoNotProcess|kAddressSet)) {
      // -- Do nothing if we have been told not to.
      // Or the data member in this branch is not longer part of the
      // parent's layout.
      return;
   }

   //--------------------------------------------------------------------------
   // Check if we are splited STL collection of pointers
   //--------------------------------------------------------------------------
   if( fType == 41 && fSplitLevel >= TTree::kSplitCollectionOfPointers )
   {
      TBranchElement *parent = (TBranchElement *)GetMother()->GetSubBranch( this );

      TVirtualStreamerInfo *sinfo = GetInfoImp();
      if (sinfo && sinfo->IsCompiled())
      {
         // If our streamer info has already been compiled,
         // then we must try to deal with schema evolution here.
         // FIXME: We must not optimize here or InitializeOffsets will crash!
         sinfo->BuildOld();
      }

      if( !parent->GetAddress() )
         parent->SetAddress( 0 );
      return;
   }

   //--------------------------------------------------------------------------
   // Any other case
   //--------------------------------------------------------------------------
   TBranchElement* mother = (TBranchElement*) GetMother();
   if (!mother) {
      return;
   }
   TClass* cl = TClass::GetClass(mother->GetClassName());

   {
      TVirtualStreamerInfo *sinfo = GetInfoImp();
      // FIXME: Should this go after the mother and cl test?
      if (sinfo && sinfo->IsCompiled()) {
         // If our streamer info has already been compiled,
         // then we must try to deal with schema evolution here.
         // FIXME: We must not optimize here or InitializeOffsets will crash!
         sinfo->BuildOld();
      }
   }

   if (!cl) {
      return;
   }

   if (!mother->GetAddress()) {
      // -- Our top-level branch has no address.
      Bool_t motherStatus = mother->TestBit(kDoNotProcess);
      mother->ResetBit(kDoNotProcess);
      // Note: This will allocate an object.
      mother->SetAddress(0);
      mother->SetBit(kDoNotProcess, motherStatus);
   }
}

//______________________________________________________________________________
void TBranchElement::Streamer(TBuffer& R__b)
{
   // -- Stream an object of class TBranchElement.
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(TBranchElement::Class(), this);
      fParentClass.SetName(fParentName);
      fBranchClass.SetName(fClassName);
      fTargetClass.SetName( fClassName );
      // The fAddress and fObject data members are not persistent,
      // therefore we do not own anything.
      // Also clear the bit possibly set by the schema evolution.
      ResetBit(kDeleteObject|kCache|kOwnOnfileObj|kAddressSet);
      // Fixup a case where the TLeafElement was missing
      if ((fType == 0) && (fLeaves.GetEntriesFast() == 0)) {
         TLeaf* leaf = new TLeafElement(this, GetTitle(), fID, fStreamerType);
         leaf->SetTitle(GetTitle());
         fNleaves = 1;
         fLeaves.Add(leaf);
         fTree->GetListOfLeaves()->Add(leaf);
      }
      // SetReadLeavesPtr();
   }
   else {
      TDirectory* dirsav = fDirectory;
      fDirectory = 0;  // to avoid recursive calls
      {
         // Save class version.
         Int_t classVersion = fClassVersion;
         // Record only positive 'version number'
         if (fClassVersion < 0) {
            fClassVersion = -fClassVersion;
         }
         // TODO: Should we clear the kDeleteObject bit before writing?
         //       If we did we would have to remember the old value and
         //       put it back, we wouldn't want to forget that we owned
         //       something just because we got written to disk.
         R__b.WriteClassBuffer(TBranchElement::Class(), this);
         // Restore class version.
         fClassVersion = classVersion;
      }
      //
      //  Mark all streamer infos used by this branch element
      //  to be written to our output file.
      //
      {
         R__b.ForceWriteInfo(GetInfoImp(), kTRUE);
      }
      //
      //  If we are a clones array master branch, or an
      //  STL container master branch, we must also mark
      //  the streamer infos used by the value class to
      //  be written to our output file.
      //
      if (fType == 3) {
         // -- TClonesArray, counter/master branch
         //
         //  We must mark the streamer info for the
         //  value class to be written to the file.
         //
         const char* nm = GetClonesName();
         if (nm && strlen(nm)) {
            TClass* cl = TClass::GetClass(nm);
            if (cl) {
               R__b.ForceWriteInfo(cl->GetStreamerInfo(), kTRUE);
            }
         }
      }
      else if (fType == 4) {
         // -- STL container, counter/master branch
         //
         //  We must mark the streamer info for the
         //  value class to be written to the file.
         //
         TVirtualCollectionProxy* cp = GetCollectionProxy();
         if (cp) {
            TClass* cl = cp->GetValueClass();
            if (cl) {
               R__b.ForceWriteInfo(cl->GetStreamerInfo(), kTRUE);
            }
         }
      }
      // If we are in a separate file, then save
      // ourselves as an independent key.
      if (!dirsav) {
         // Note: No need to restore fDirectory, it was already zero.
         return;
      }
      if (!dirsav->IsWritable()) {
         fDirectory = dirsav;
         return;
      }
      TDirectory* pdirectory = fTree->GetDirectory();
      if (!pdirectory) {
         fDirectory = dirsav;
         return;
      }
      const char* treeFileName = pdirectory->GetFile()->GetName();
      TBranch* mother = GetMother();
      const char* motherFileName = treeFileName;
      if (mother && (mother != this)) {
         motherFileName = mother->GetFileName();
      }
      if ((fFileName.Length() > 0) && strcmp(motherFileName, fFileName.Data())) {
         dirsav->WriteTObject(this);
      }
      fDirectory = dirsav;
   }
}

//______________________________________________________________________________
Int_t TBranchElement::Unroll(const char* name, TClass* clParent, TClass* cl, char* ptr, Int_t basketsize, Int_t splitlevel, Int_t btype)
{
   // -- Split class cl into sub-branches of this branch.
   //
   // Create a sub-branch of this branch for each non-empty,
   // non-abstract base class of cl (unless we are a sub-branch
   // of a TClonesArray or an STL container, in which case we
   // do *not* create a sub-branch), and for each non-split data
   // member of cl.
   //
   // Note: We do *not* create sub-branches for base classes of cl
   //       if we are a sub-branch of a TClonesArray or an STL container.
   //
   // Note: We do *not* create sub-branches for data members which
   //       have a class type and which we are splitting.
   //
   // Note: The above rules imply that the branch heirarchy increases
   //       in depth only for base classes of cl (unless we are inside
   //       of a TClonesArray or STL container, in which case the depth
   //       does *not* increase, the base class is elided) and for
   //       TClonesArray or STL container data members (which have one
   //       additional level of sub-branches).  The only other way the
   //       depth increases is when the top-level branch has a split
   //       class data member, in that case the constructor will create
   //       a sub-branch for it.  In other words, the interior nodes of
   //       the branch tree are all either: base class nodes; split
   //       class nodes which are direct sub-branches of top-level nodes
   //       (created by TClass::Bronch usually); or TClonesArray or STL
   //       container master nodes.
   //
   // Note: The exception to the above is for the top-level branches,
   //       Tree::Bronch creates nodes for everything in that case,
   //       except for a TObject base class of a class which has the
   //       can ignore tobject streamer flag set.

   //----------------------------------------------------------------------------
   // Handling the case of STL collections of pointers
   //----------------------------------------------------------------------------
   Int_t splitSTLP = splitlevel - (splitlevel%TTree::kSplitCollectionOfPointers);
   splitlevel %= TTree::kSplitCollectionOfPointers;


   TString branchname;

   if ((cl == TObject::Class()) && clParent->CanIgnoreTObjectStreamer()) {
      return 0;
   }

   //
   //  Rebuild the streamer info for cl unoptimized
   //  so that each data member has its own streamer
   //  element.  This allows each data member to have
   //  its own branch and thus be stored and queried
   //  independently in the tree.
   //
   TStreamerInfo* sinfo = fTree->BuildStreamerInfo(cl);
   if (splitlevel > 0) {
      sinfo->SetBit(TVirtualStreamerInfo::kCannotOptimize);
      sinfo->Compile();
   }

   //
   //  Do nothing if we couldn't build the streamer info for cl.
   //

   if (!sinfo) {
      return 0;
   }

   Int_t ndata = sinfo->GetNdata();
   ULong_t* elems = sinfo->GetElems();

   if ((ndata == 1) && cl->GetCollectionProxy() && !strcmp(((TStreamerElement*) elems[0])->GetName(), "This")) {
      // -- Class cl is an STL collection, refuse to split it.
      // Question: Why?  We certainly could by switching to the value class.
      // Partial Answer: Only the branch element constructor can split STL containers.
      return 1;
   }

   for (Int_t elemID = 0; elemID < ndata; ++elemID) {
      // -- Loop over all the streamer elements and create sub-branches as needed.
      TStreamerElement* elem = (TStreamerElement*) elems[elemID];
      if (elem->IsA() == TStreamerArtificial::Class()) {
         continue;
      }
      if (elem->TestBit(TStreamerElement::kRepeat)) {
         continue;
      }
      Int_t offset = elem->GetOffset();
      // FIXME: An STL container as a base class gets TStreamerSTL as its class, so this test is not enough.
      // See InitializeOffsets() for the proper test.
      if (elem->IsA() == TStreamerBase::Class()) {
         // -- This is a base class of cl.
         TClass* clOfBase = TClass::GetClass(elem->GetName());
         if ((clOfBase->Property() & kIsAbstract) && cl->InheritsFrom(TCollection::Class())) {
            // -- Do nothing if we are abstract.
            // FIXME: We should not test for TCollection here.
            return -1;
         }
         if ((btype == 31) || (btype == 41)) {
            // -- Elide the base-class sub-branches of a split TClonesArray or STL container.
            //
            // Note: We are eliding the base class here, that is, we never
            //       create a branch for it, so the branch heirarchy is not
            //       complete.
            // Note: The clParent parameter is the value class of the
            //       container which we are splitting.  It does not
            //       appear in the branch heirarchy either.
            // Note: We can use parent class (clParent) != branch class (elemClass) to detection elision.
            Int_t unroll = Unroll(name, clParent, clOfBase, ptr + offset, basketsize, splitlevel+splitSTLP, btype);
            if (unroll < 0) {
               // FIXME: We could not split because we are abstract, should we be doing this?
               if (strlen(name)) {
                  branchname.Form("%s.%s", name, elem->GetFullName());
               } else {
                  branchname.Form("%s", elem->GetFullName());
               }
               TBranchElement* branch = new TBranchElement(this, branchname, sinfo, elemID, 0, basketsize, 0, btype);
               branch->SetParentClass(clParent);
               fBranches.Add(branch);
            }
         } else if (clOfBase->GetListOfRealData()->GetSize()) {
            // -- Create a branch for a non-empty base class.
            if (strlen(name)) {
               branchname.Form("%s.%s", name, elem->GetFullName());
               // Elide the base class name when creating the sub-branches.
               // Note: The branch names for sub-branches of a base class branch
               //       do not represent the full class heirarchy because we do
               //       this, however it does keep the branch names for the
               //       inherited data members simple.
               TBranchElement* branch = new TBranchElement(this, name, sinfo, elemID, ptr + offset, basketsize, splitlevel+splitSTLP, btype);
               // Then reset it to the proper name.
               branch->SetName(branchname);
               branch->SetTitle(branchname);
               branch->SetParentClass(clParent);
               fBranches.Add(branch);
            } else {
               branchname.Form("%s", elem->GetFullName());
               TBranchElement* branch = new TBranchElement(this, branchname, sinfo, elemID, ptr + offset, basketsize, splitlevel+splitSTLP, btype);
               branch->SetParentClass(clParent);
               fBranches.Add(branch);
            }
         }
      } else {
         // -- This is a data member of cl.
         if (strlen(name)) {
            branchname.Form("%s.%s", name, elem->GetFullName());
         } else {
            branchname.Form("%s", elem->GetFullName());
         }
         if ((splitlevel > 1) && ((elem->IsA() == TStreamerObject::Class()) || (elem->IsA() == TStreamerObjectAny::Class()))) {
            // -- We are splitting a non-TClonesArray (may inherit from TClonesArray though), non-STL container object.
            //
            // Ignore an abstract class.
            // FIXME: How could an abstract class get here?
            //        Partial answer: It is a base class.  But this is a data member!
            TClass* elemClass = TClass::GetClass(elem->GetTypeName());
            if (elemClass->Property() & kIsAbstract) {
               return -1;
            }
            if (elem->CannotSplit()) {
               // We are not splitting.
               TBranchElement* branch = new TBranchElement(this, branchname, sinfo, elemID, ptr + offset, basketsize, 0, btype);
               branch->SetParentClass(clParent);
               fBranches.Add(branch);
            } else if (elemClass->InheritsFrom(TClonesArray::Class())) {
               // Splitting something derived from TClonesArray.
               Int_t subSplitlevel = splitlevel-1;
               if (btype == 31 || btype == 41 || elem->CannotSplit()) {
                  // -- We split the sub-branches of a TClonesArray or an STL container only once.
                  subSplitlevel = 0;
               }
               TBranchElement* branch = new TBranchElement(this, branchname, sinfo, elemID, ptr + offset, basketsize, subSplitlevel, btype);
               branch->SetParentClass(clParent);
               fBranches.Add(branch);
            } else {
               // Splitting a normal class.
               // FIXME: We are eliding the class we are splitting here,
               //        i.e., we do not create a branch for it, so the
               //        branch heirarchy does not match the class heirarchy.
               // Note: clParent is the class which contains a data member of
               //       the class type which we are splitting.
               // Note: We can use parent class (clParent) != branch class (elemClass) to detection elision.
               Int_t unroll = Unroll(branchname, clParent, elemClass, ptr + offset, basketsize, splitlevel-1+splitSTLP, btype);
               if (unroll < 0) {
                  // FIXME: We could not split because we are abstract, should we be doing this?
                  TBranchElement* branch = new TBranchElement(this, branchname, sinfo, elemID, ptr + offset, basketsize, 0, btype);
                  branch->SetParentClass(clParent);
                  fBranches.Add(branch);
               }
            }
         }
         else if( elem->GetClassPointer() &&
                  elem->GetClassPointer()->GetCollectionProxy() &&
                  elem->GetClassPointer()->GetCollectionProxy()->HasPointers() &&
                  splitSTLP && fType != 4 )
         {

            TBranchSTL* branch = new TBranchSTL( this, branchname,
                                                 elem->GetClassPointer()->GetCollectionProxy(),
                                                 basketsize, splitlevel - 1+splitSTLP, sinfo, elemID );
            branch->SetAddress( ptr+offset );
            fBranches.Add( branch );
         }
         else if ((elem->IsA() == TStreamerSTL::Class()) && !elem->IsaPointer()) {
            // -- We have an STL container.
            // Question: What if splitlevel == 0 here?
            // Answer: then we should not be here.
            Int_t subSplitlevel = splitlevel - 1;
            if ((btype == 31) || (btype == 41) || elem->CannotSplit()) {
               // -- We split the sub-branches of a TClonesArray or an STL container only once.
               subSplitlevel = 0;
            }
            TBranchElement* branch = new TBranchElement(this, branchname, sinfo, elemID, ptr + offset, basketsize, subSplitlevel+splitSTLP, btype);
            branch->SetParentClass(clParent);
            fBranches.Add(branch);
         } else if (((btype != 31) && (btype != 41)) && ptr && ((elem->GetClassPointer() == TClonesArray::Class()) || ((elem->IsA() == TStreamerSTL::Class()) && !elem->CannotSplit()))) {
            // -- We have a TClonesArray.
            // FIXME: We could get a ptr to a TClonesArray here by mistake.
            // Question: What if splitlevel == 0 here?
            // Answer: then we should not be here.
            // Note: ptr may be null in case of a TClonesArray inside another
            //       TClonesArray or STL container, see the else clause.
            TBranchElement* branch = new TBranchElement(this, branchname, sinfo, elemID, ptr + offset, basketsize, splitlevel-1+splitSTLP, btype);
            branch->SetParentClass(clParent);
            fBranches.Add(branch);
         } else {
            // -- We are not going to split this element any farther.
            TBranchElement* branch = new TBranchElement(this, branchname, sinfo, elemID, 0, basketsize, splitSTLP, btype);
            branch->SetType(btype);
            branch->SetParentClass(clParent);
            fBranches.Add(branch);
         }
      }
   }

   return 1;
}

//______________________________________________________________________________
void TBranchElement::UpdateFile()
{
   // Refresh the value of fDirectory (i.e. where this branch writes/reads its buffers)
   // with the current value of fTree->GetCurrentFile unless this branch has been
   // redirected to a different file.  Also update the sub-branches.

   // The BranchCount and BranchCount2 are part of higher level branches' list of
   // branches.
   // if (fBranchCount) fBranchCount->UpdateFile();
   // if (fBranchCount2) fBranchCount2->UpdateFile();
   TBranch::UpdateFile();
}
