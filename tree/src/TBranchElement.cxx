// @(#)root/tree:$Name:  $:$Id: TBranchElement.cxx,v 1.207 2006/08/08 20:56:25 pcanal Exp $
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

#include "Api.h"
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
#include "TFile.h"
#include "TFolder.h"
#include "TLeafElement.h"
#include "TROOT.h"
#include "TRealData.h"
#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TTree.h"
#include "TVirtualCollectionProxy.h"
#include "TVirtualPad.h"

R__EXTERN  TTree* gTree;

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
}

//______________________________________________________________________________
namespace {
   void SwitchContainer(TObjArray* branches) {
      // -- Modify the container type of the branches

      const Int_t nbranches = branches->GetEntriesFast();
      for (Int_t i = 0; i < nbranches; ++i) {
         TBranchElement* br = (TBranchElement*) branches->At(i);
         switch (br->GetType()) {
            case 31: br->SetType(41); break;
            case 41: br->SetType(31); break;
         };
         // FIXME: This is a tail recursion.
         SwitchContainer(br->GetListOfBranches());
      }
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
, fInit(kFALSE)
, fInitOffsets(kFALSE)
, fCurrentClass()
, fParentClass()
, fBranchClass()
, fBranchOffset(0)
{
   // -- Default constructor.
   fNleaves = 1;
}

//______________________________________________________________________________
TBranchElement::TBranchElement(const char* bname, TStreamerInfo* sinfo, Int_t id, char* pointer, Int_t basketsize, Int_t splitlevel, Int_t btype)
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
// FIXME: What if the streamer info is optimized here?
, fInfo(sinfo)
, fObject(0)
, fInit(kTRUE)
, fInitOffsets(kFALSE)
, fCurrentClass()
, fParentClass()
, fBranchClass(sinfo->GetClass())
, fBranchOffset(0)
{
   // -- Constructor when the branch object is not a TClonesArray nor an STL container.
   //
   // If splitlevel > 0 this branch in turn is split into sub-branches.

   TString name(bname);

   // Set our TNamed attributes.
   SetName(name);
   SetTitle(name);

   // Set our TBranch attributes.
   fSplitLevel = splitlevel;
   fTree = gTree;
   fDirectory = fTree->GetDirectory();
   fFileName = "";

   // Clear the bit kAutoDelete to specify that when reading
   // the object should not be deleted before calling Streamer.

   SetAutoDelete(kFALSE);

   fCompress = -1;
   if (gTree->GetDirectory()) {
      TFile* bfile = gTree->GetDirectory()->GetFile();
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
   if (btype || (fStreamerType <= TStreamerInfo::kBase) || (fStreamerType == TStreamerInfo::kCharStar) || (fStreamerType == TStreamerInfo::kBits) || (fStreamerType > TStreamerInfo::kBool)) {
      fEntryOffsetLen = 1000;
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

   // Create a basket for the branch.
   TBasket* basket = new TBasket(name, fTree->GetName(), this);
   fBaskets.Add(basket);

   // We need to keep track of the counter branch if we have
   // one, since we cannot set it until we have created our
   // leaf, which we do last.
   TBranchElement* brOfCounter = 0;

   if (id < 0) {
      // -- We are a top-level branch.  Don't split a top-level branch, TTree::Bronch will do that work.
      // FIXME: It probably shouldn't.
      // FIXME: fBranchClass.GetClass() could return a null pointer.
      // FIXME: Change this to a dynamic cast.
      if (fBranchClass.GetClass()->InheritsFrom(TObject::Class())) {
        SetBit(kBranchObject);
      }
   } else {
      // -- We are a sub-branch of a split object.
      ULong_t* elems = sinfo->GetElems();
      TStreamerElement* element = (TStreamerElement*) elems[id];
      if ((fStreamerType == TStreamerInfo::kObject) || (fStreamerType == TStreamerInfo::kBase) || (fStreamerType == TStreamerInfo::kTNamed) || (fStreamerType == TStreamerInfo::kTObject) || (fStreamerType == TStreamerInfo::kObjectp) || (fStreamerType == TStreamerInfo::kObjectP)) {
         // -- If we are an object data member which inherits from TObject,
         // flag it so that later during i/o we will register the object
         // with the buffer so that pointers are handled correctly.
         // FIXME: fBranchClass::operator TClass*() could return a null pointer.
         // FIXME: Change this to a dynamic cast.
         if (fBranchClass.GetClass()->InheritsFrom(TObject::Class())) {
            SetBit(kBranchObject);
         }
      }
      if (element->IsA() == TStreamerBasicPointer::Class()) {
         // -- Fixup title with counter if we are a varying length array data member.
         TStreamerBasicPointer *bp = (TStreamerBasicPointer *)element;
         TString countname;
         countname = bname;
         Ssiz_t dot = countname.Last('.');
         if (dot>=0) {
            countname.Remove(dot);
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
            // FIXME: This does not include an STL container class which is
            //        being used as a base class because the streamer element
            //        in that case is not the base streamer element it is the
            //        STL streamer element.
            fType = 1;
            TClass* clOfElement = gROOT->GetClass(element->GetName());
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
               // FIXME: We are eliding the base class here, creating a break in the branch heirarchy.
               // Note: We can use parent class (cltop) != branch class (elemClass) to detection elision.
               Unroll("", fBranchClass.GetClass(), clOfElement, pointer, basketsize, splitlevel, 0);
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
            Unroll(name, clOfElement, clOfElement, pointer, basketsize, splitlevel, 0);
            if (strchr(bname, '.')) {
               // FIXME: How can this happen?
               // FIXME: Answer: This is the case when using the new branch
               //        naming convention where the top-level branch ends in dot.
               // FIXME: Well actually not entirely, we could also be a sub-branch
               //        of a split class, even when the top-level branch does not
               //        end in a dot.
               // FIXME: Or the top-level branch could have been created by the
               //        branch constructor which takes a folder as input, in which
               //        case the top-level branch name will have internal dots
               //        representing the folder heirarchy.
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
            basket->DeleteEntryOffset(); //entryoffset not required for the clonesarray counter
            fEntryOffsetLen = 0;
            // ===> Create a leafcount
            TLeaf* leaf = new TLeafElement(name, fID, fStreamerType);
            leaf->SetBranch(this);
            fNleaves = 1;
            fLeaves.Add(leaf);
            fTree->GetListOfLeaves()->Add(leaf);
            if (!clones) return;
            TClass* clOfClones = clones->GetClass();
            if (!clOfClones) {
               return;
            }
            // Create a basket for the leafcount
            TBasket *basket2 = new TBasket(name,fTree->GetName(),this);
            fBaskets.Add(basket2);
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
            Unroll(name, clOfClones, clOfClones, pointer, basketsize, splitlevel, 31);
            BuildTitle(name);
            return;
         } else if (((fSTLtype >= TClassEdit::kVector) && (fSTLtype <= TClassEdit::kMultiSet)) || ((fSTLtype >= -TClassEdit::kMultiSet) && (fSTLtype <= -TClassEdit::kVector))) {
            // -- We are an STL container element.
            TClass* contCl = gROOT->GetClass(elem_type);
            fCollProxy = contCl->GetCollectionProxy()->Generate();
            TClass* valueClass = GetCollectionProxy()->GetValueClass();
            // Check to see if we can split the container.
            Bool_t cansplit = kTRUE;
            if (!valueClass) {
               cansplit = kFALSE;
            } else if ((valueClass == TString::Class()) || (valueClass == gROOT->GetClass("string"))) {
               cansplit = kFALSE;
            } else if (GetCollectionProxy()->HasPointers()) {
               cansplit = kFALSE;
            } else if (!valueClass->CanSplit()) {
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
               TLeaf *leaf = new TLeafElement(name, fID, fStreamerType);
               leaf->SetBranch(this);
               fNleaves = 1;
               fLeaves.Add(leaf);
               fTree->GetListOfLeaves()->Add(leaf);
               // Create a basket for the master branch (the counter).
               TBasket *basket2 = new TBasket(name,fTree->GetName(),this);
               fBaskets.Add(basket2);
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
               Unroll(name, valueClass, valueClass, pointer, basketsize, splitlevel, 41);
               BuildTitle(name);
               return;
            }
         } else if (!strchr(elem_type, '*') && ((fStreamerType == TStreamerInfo::kObject) || (fStreamerType == TStreamerInfo::kAny))) {
            // -- Create sub-branches for members that are classes.
            //
            // Note: This can only happen if we were called directly
            //       (usually by TClass::Bronch) because Unroll never
            //       calls us for an element of this type.
            fType = 2;
            TClass* clm = gROOT->GetClass(elem_type);
            Int_t err = Unroll(name, clm, clm, pointer, basketsize, splitlevel, 0);
            if (err >= 0) {
               // Return on success.
               // FIXME: Why not on error too?
               return;
            }
         }
      }
   }

   //
   // Create a leaf to represent this branch.
   //

   TLeaf* leaf = new TLeafElement(GetTitle(), fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   leaf->SetBranch(this);
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
}

//______________________________________________________________________________
TBranchElement::TBranchElement(const char* bname, TClonesArray* clones, Int_t basketsize, Int_t splitlevel, Int_t compress)
: TBranch()
, fClassName("TClonesArray")
, fParentName()
// FIXME: Bad, the streamer info will be optimized here.
, fInfo(TClonesArray::Class()->GetStreamerInfo())
, fCurrentClass()
, fParentClass()
, fBranchClass(TClonesArray::Class())
{
   // -- Constructor when the branch object is a TClonesArray.
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
   fMaximum       = 0;
   fBranchOffset  = 0;
   fSTLtype       = TClassEdit::kNotSTL;
   fInitOffsets   = kFALSE;

   fTree          = gTree;
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

   // Create a basket for the terminal branch
   TBasket *basket = new TBasket(name, fTree->GetName(), this);
   fBaskets.Add(basket);

   // Reset the bit kAutoDelete to specify that when reading
   // the object should not be deleted before calling the streamer.
   SetAutoDelete(kFALSE);

   // create sub branches if requested by splitlevel
   if (splitlevel > 0) {
      fType = 3;
      // ===> Create a leafcount
      TLeaf* leaf = new TLeafElement(name, fID, fStreamerType);
      leaf->SetBranch(this);
      fNleaves = 1;
      fLeaves.Add(leaf);
      fTree->GetListOfLeaves()->Add(leaf);
      // Create a basket for the leafcount
      TBasket* basket = new TBasket(name, fTree->GetName(), this);
      fBaskets.Add(basket);
      // ===> create sub branches for each data member of a TClonesArray
      TClass* clonesClass = clones->GetClass();
      if (!clonesClass) {
         // FIXME: Need an error message here.
         return;
      }
      fClonesName = clonesClass->GetName();
      std::string branchname = name + std::string("_");
      SetTitle(branchname.c_str());
      leaf->SetName(branchname.c_str());
      leaf->SetTitle(branchname.c_str());
      Unroll(name, clonesClass, clonesClass, 0, basketsize, splitlevel, 31);
      BuildTitle(name);
      return;
   }

   SetBit(kBranchObject);

   TLeaf *leaf = new TLeafElement(GetTitle(), fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   leaf->SetBranch(this);
   fNleaves = 1;
   fLeaves.Add(leaf);
   gTree->GetListOfLeaves()->Add(leaf);
}

//______________________________________________________________________________
TBranchElement::TBranchElement(const char* bname, TVirtualCollectionProxy* cont, Int_t basketsize, Int_t splitlevel, Int_t compress)
: TBranch()
, fClassName(cont->GetCollectionClass()->GetName())
, fParentName()
, fCurrentClass()
, fParentClass()
, fBranchClass(cont->GetCollectionClass())
{
   // -- Constructor when the branch object is an STL collection.
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
   fStreamerType  = -1; // TStreamerInfo::kSTLp;
   fType          = 0;
   fClassVersion  = cont->GetCollectionClass()->GetClassVersion();
   fCheckSum      = cont->GetCollectionClass()->GetCheckSum();
   fBranchCount   = 0;
   fBranchCount2  = 0;
   fObject        = 0;
   fMaximum       = 0;
   fBranchOffset  = 0;
   fSTLtype       = TClassEdit::kNotSTL;

   fTree          = gTree;
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

   fBasketEntry[0] = fEntryNumber;
   fBasketBytes[0] = 0;

   // Create a basket for the terminal branch
   TBasket* basket = new TBasket(name, fTree->GetName(), this);
   fBaskets.Add(basket);

   // Reset the bit kAutoDelete to specify that, when reading,
   // the object should not be deleted before calling the streamer.
   SetAutoDelete(kFALSE);

   // create sub branches if requested by splitlevel
   if ((splitlevel > 0) && fBranchClass.GetClass() && fBranchClass.GetClass()->CanSplit()) {
      fType = 4;
      // ===> Create a leafcount
      TLeaf* leaf = new TLeafElement(name, fID, fStreamerType);
      leaf->SetBranch(this);
      fNleaves = 1;
      fLeaves.Add(leaf);
      fTree->GetListOfLeaves()->Add(leaf);
      // Create a basket for the leafcount
      TBasket* basket = new TBasket(name, fTree->GetName(), this);
      fBaskets.Add(basket);
      // ===> create sub branches for each data member of an STL container value class
      TClass* valueClass = cont->GetValueClass();
      if (!valueClass) {
         return;
      }
      fClonesName = valueClass->GetName();
      TString branchname;
      branchname += "_";
      SetTitle(branchname);
      leaf->SetName(branchname);
      leaf->SetTitle(branchname);
      Unroll(name, valueClass, valueClass, 0, basketsize, splitlevel, 41);
      BuildTitle(name);
      return;
   }

   TLeaf *leaf = new TLeafElement(GetTitle(), fID, fStreamerType);
   leaf->SetTitle(GetTitle());
   leaf->SetBranch(this);
   fNleaves = 1;
   fLeaves.Add(leaf);
   gTree->GetListOfLeaves()->Add(leaf);
}

//______________________________________________________________________________
TBranchElement::~TBranchElement()
{
   // -- Destructor.

   // Release any allocated I/O buffers.
   // FIXME: Temporarily disable until we sort out the interface/documentation
   // issues related the TTree ownership of the objects (See ReleaseObject)
   // ResetAddress();

   delete[] fBranchOffset;
   fBranchOffset = 0;

   fInfo = 0;
   fBranchCount2 = 0;
   fBranchCount = 0;

   //delete fCollProxy;
   fCollProxy = 0;
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
               cl=gROOT->GetClass(GetClonesName());
            else {
               cl=gROOT->GetClass(GetClassName());

               // check if we're in a sub-branch of this class
               // we can only find out asking the streamer given our ID
               ULong_t *elems=0;
               TStreamerElement *element=0;
               TClass* clsub=0;
               if (fID>=0 && GetInfo()
                   && ((elems=GetInfo()->GetElems()))
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
      // FIXME: This will fail if our sub-branches are not containers.
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
         TStreamerBasicPointer *el = (TStreamerBasicPointer*)bre->GetInfo()->GetElements()->FindObject(name2.Data()+bn+1);
         name2.Remove(bn+1);
         name2 += el->GetCountName();
         TBranchElement *bc2 = (TBranchElement*)fBranches.FindObject(name2);
         bre->SetBranchCount2(bc2);
      }
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
   if ((fType >= 0) && (fType < 10)) {
      TBranchRef* bref = fTree->GetBranchRef();
      if (bref) {
         bref->SetParent(this);
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
   // FIXME: Does this mean that pointers to objects which do not
   //        inherit from tobject are not handled correctly?
   //

   if ((fType <= 2) && TestBit(kBranchObject)) {
      // We are not a TClonesArray master/sub nor an STL container master/sub,
      // so we are either a top-level branch, a base class branch, a split class
      // branch, or a data member branch.
      // FIXME: We should probably only map data member branches.
      // FIXME: We should only map addresses we actually do i/o on.
      // FIXME: We should map fAddress instead for MakeClass() trees.
      b.MapObject((TObject*) fObject);
   }

   //
   // Do the actual buffer filling now.
   //

   if (fType <= 2) {
      // -- Top-level, data member, base class, or split class branch.
      // A non-split top-level branch (0, and fID == -1)), a non-split object (0, and fID > -1), or a base class (1), or a split (non-TClonesArray, non-STL container) object (2).  Write out the object.
      // Note: A split top-level branch (0, and fID == -2) should not happen here, see Fill().
      // FIXME: What happens with a split base class branch,
      //        or a split class branch???
      TStreamerInfo* si = GetInfo();
      if (!si) {
         Error("FillLeaves", "Cannot get streamer info for branch '%s'", GetName());
         return;
      }
      Int_t n = si->WriteBufferAux(b, &fObject, fID, 1, 0, 0);
      if ((fStreamerType == TStreamerInfo::kCounter) && (n > fMaximum)) {
         fMaximum = n;
      }
   } else if (fType == 3) {
      // -- TClonesArray top-level branch.  Write out number of entries, sub-branch writes the entries themselves.
      if (fTree->GetMakeClass()) {
         // FIXME: What if GetClonesName() is the zero pointer or an empty string?
         TClass* cl = gROOT->GetClass(GetClonesName());
         // FIXME: What if cl is a zero pointer here?
         TStreamerInfo* si = cl->GetStreamerInfo();
         if (!si) {
            Error("FillLeaves", "Cannot get streamer info for branch '%s' class '%s'", GetName(), cl->GetName());
            return;
         }
         // FIXME: What if GetParent() returns a zero pointer here?
         si->ForceWriteInfo((TFile *) b.GetParent());
         Int_t* nptr = (Int_t*) fAddress;
         b << *nptr;
      } else {
         if (!fObject) {
            // FIXME: This cannot happen, see test at begin of function.
            b << 0;
         } else {
            TClonesArray* clones = (TClonesArray*) fObject;
            Int_t n = clones->GetEntriesFast();
            if (n > fMaximum) {
               fMaximum = n;
            }
            b << n;
         }
      }
   } else if (fType == 4) {
      // -- STL container top-level branch.  Write out number of entries, sub-branch writes the entries themselves.
      if (!fObject) {
         // FIXME: This cannot happen, see test at begin of function.
         b << 0;
      } else {
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
      }
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
            case TStreamerInfo::kChar     /*  1 */: { b.WriteFastArray((Char_t*)    fAddress, n); break; }
            case TStreamerInfo::kShort    /*  2 */: { b.WriteFastArray((Short_t*)   fAddress, n); break; }
            case TStreamerInfo::kInt      /*  3 */: { b.WriteFastArray((Int_t*)     fAddress, n); break; }
            case TStreamerInfo::kLong     /*  4 */: { b.WriteFastArray((Long_t*)    fAddress, n); break; }
            case TStreamerInfo::kFloat    /*  5 */: { b.WriteFastArray((Float_t*)   fAddress, n); break; }
            case TStreamerInfo::kCounter  /*  6 */: { b.WriteFastArray((Int_t*)     fAddress, n); break; }
            // FIXME: We do nothing with type 7 (TStreamerInfo::kCharStar, char*) here!
            case TStreamerInfo::kDouble   /*  8 */: { b.WriteFastArray((Double_t*)  fAddress, n); break; }
            case TStreamerInfo::kDouble32 /*  9 */: {
               Double_t* xx = (Double_t*) fAddress;
               for (Int_t ii = 0; ii < n; ++ii) {
                  b << (Float_t) xx[ii];
               }
               break;
            }
            // Note: Type 10 is unused for now.
            case TStreamerInfo::kUChar    /* 11 */: { b.WriteFastArray((UChar_t*)   fAddress, n); break; }
            case TStreamerInfo::kUShort   /* 12 */: { b.WriteFastArray((UShort_t*)  fAddress, n); break; }
            case TStreamerInfo::kUInt     /* 13 */: { b.WriteFastArray((UInt_t*)    fAddress, n); break; }
            case TStreamerInfo::kULong    /* 14 */: { b.WriteFastArray((ULong_t*)   fAddress, n); break; }
            // FIXME: This is wrong!!! TStreamerInfo::kBits is a variable length type.
            case TStreamerInfo::kBits     /* 15 */: { b.WriteFastArray((UInt_t*)    fAddress, n); break; }
            case TStreamerInfo::kLong64   /* 16 */: { b.WriteFastArray((Long64_t*)  fAddress, n); break; }
            case TStreamerInfo::kULong64  /* 17 */: { b.WriteFastArray((ULong64_t*) fAddress, n); break; }
            case TStreamerInfo::kBool     /* 18 */: { b.WriteFastArray((Bool_t*)    fAddress, n); break; }
            // Note: Type 19 is unused for now.
         }
      } else {
         if (!fObject) {
            // FIXME: This cannot happen, see test at begin of function.
            return;
         } else {
            TClonesArray* clones = (TClonesArray*) fObject;
            Int_t n = clones->GetEntriesFast();
            TStreamerInfo* si = GetInfo();
            if (!si) {
               Error("FillLeaves", "Cannot get streamer info for branch '%s'", GetName());
               return;
            }
            si->WriteBufferClones(b, clones, n, fID, fOffset);
         }
      }
   } else if (fType == 41) {
      // -- STL container sub-branch.  Write out the entries in the STL container.
      if (!fObject) {
         // FIXME: This cannot happen, see test at begin of function.
         return;
      } else {
         // FIXME: This wont work if a pointer to vector is split!
         // FIXME: What if GetCollectionProxy() returns a zero pointer here?
         Int_t n = 0;
         TVirtualCollectionProxy::TPushPop helper(GetCollectionProxy(), fObject);
         n = GetCollectionProxy()->Size();
         // Note: We cannot pop the proxy here because we need it for the i/o.
         TStreamerInfo* si = GetInfo();
         if (!si) {
            Error("FillLeaves", "Cannot get streamer info for branch '%s'", GetName());
            return;
         }
         si->WriteBufferSTL(b, GetCollectionProxy(), n, fID, fOffset);
      }
   }
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
TStreamerInfo* TBranchElement::GetInfo() const
{
   // -- Get streamer info for the branch class, try to compensate for class code unload/reload and schema evolution.

   if (!fInfo) {
      // We did not already have streamer info, so now we must find it.
      TClass* cl = fBranchClass.GetClass();
      if (cl) {
         if (cl == TClonesArray::Class()) {
            const_cast<TBranchElement*>(this)->fClassVersion = TClonesArray::Class()->GetClassVersion();
         }
         Bool_t optim = TStreamerInfo::CanOptimize();
         TStreamerInfo::Optimize(kFALSE);
         const_cast<TBranchElement*>(this)->fInfo = cl->GetStreamerInfo(fClassVersion);
         TStreamerInfo::Optimize(optim);
         // FIXME: Check that the found streamer info checksum matches our branch class checksum here.
         // Check to see if the class code was unloaded/reloaded
         // since we were created.
         if (fCheckSum && (cl->IsForeign() || (!cl->IsLoaded() && (fClassVersion == 1) && cl->GetStreamerInfos()->At(1) && (fCheckSum != ((TStreamerInfo*) cl->GetStreamerInfos()->At(1))->GetCheckSum())))) {
            // Try to compensate for a class that got unloaded on us.
            // Search through the streamer infos by checksum
            // and take the first match.
            Int_t ninfos = cl->GetStreamerInfos()->GetEntriesFast();
            for (Int_t i = 1; i < ninfos; ++i) {
               TStreamerInfo* info = (TStreamerInfo*) cl->GetStreamerInfos()->At(i);
               if (!info) {
                  continue;
               }
               if (info->GetCheckSum() == fCheckSum) {
                  const_cast<TBranchElement*>(this)->fClassVersion = i;
                  Bool_t optim = TStreamerInfo::CanOptimize();
                  TStreamerInfo::Optimize(kFALSE);
                  const_cast<TBranchElement*>(this)->fInfo = cl->GetStreamerInfo(fClassVersion);
                  TStreamerInfo::Optimize(optim);
                  break;
               }
            }
         }
      }
   }

   //
   //  Fixup cached streamer info if necessary.
   //
   // FIXME:  What if the class code was unloaded/reloaded since we were cached?

   if (fInfo) {
      if (!fInfo->GetOffsets()) {
         // Streamer info has not yet been compiled.
         //
         // Optimizing does not work with splitting.
         Bool_t optim = TStreamerInfo::CanOptimize();
         TStreamerInfo::Optimize(kFALSE);
         const_cast<TBranchElement*>(this)->fInfo->Compile();
         TStreamerInfo::Optimize(optim);
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
            if (elt) {
               size_t ndata = fInfo->GetNdata();
               ULong_t* elems = fInfo->GetElems();
               for (size_t i = 0; i < ndata; ++i) {
                  if (((TStreamerElement*) elems[i]) == elt) {
                     const_cast<TBranchElement*>(this)->fID = i;
                     break;
                  }
               }
            }
         }
         const_cast<TBranchElement*>(this)->fInit = kTRUE;
      }
   }

   return fInfo;
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
         TStreamerInfo* si = thiscast->GetInfo();
         TStreamerElement* se = (TStreamerElement*) si->GetElems()[fID];
         className = se->GetTypeName();
      }
      TClass* cl = gROOT->GetClass(className);
      TVirtualCollectionProxy* proxy = cl->GetCollectionProxy();
      fCollProxy = proxy->Generate();
      fSTLtype = TClassEdit::IsSTLCont(className);
      if (fSTLtype < 0) {
        fSTLtype = -fSTLtype;
      }
   } else if (fType == 41) {
      // STL container sub-branch.
      thiscast->fCollProxy = fBranchCount->fCollProxy;
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

   TStreamerInfo* brInfo = GetInfo();
   if (!brInfo) {
      cl = gROOT->GetClass(GetClassName());
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
   TStreamerElement* currentStreamerElement = ((TStreamerElement*) brInfo->GetElems()[GetID()]);
   TDataMember* dm = (TDataMember*) motherCl->GetListOfDataMembers()->FindObject(currentStreamerElement->GetName());

   TString newType;
   if (!dm) {
      // Either the class is not loaded or the data member is gone
      if (! motherCl->IsLoaded()) {
         TStreamerInfo* newInfo = motherCl->GetStreamerInfo();
         if (newInfo != brInfo) {
            TStreamerElement* newElems = (TStreamerElement*) newInfo->GetElements()->FindObject(currentStreamerElement->GetName());
            newType = newElems->GetClassPointer()->GetName();
         }
      }
   } else {
      newType = dm->GetTypeName();
   }
   cl = gROOT->GetClass(newType);
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
      bref->SetParent(this);
      bref->SetReadEntry(entry);
   }

   Int_t nbytes = 0;

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
               TBranch* branch = (TBranch*) fBranches[i];
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
         return fBranchClass.GetClass()->GetName();
      } else {
         return 0;
      }
   }
   const char *types[19] = {
      "",
      "Char_t",
      "Short_t",
      "Int_t",
      "Long_t",
      "Float_t",
      "Int_t",
      "",
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
      "Bool_t"
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

   if (!j && fBranchCount) {
      Int_t entry = fTree->GetReadEntry();
      fBranchCount->TBranch::GetEntry(entry);
      if (fBranchCount2) {
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
         return GetInfo()->GetValue(fAddress, atype, j, 1);
      } else if (fType <= 2) {
         // branch in split mode
         // FIXME: This should probably be < 60 instead!
         if ((fStreamerType > 40) && (fStreamerType < 55)) {
            Int_t atype = fStreamerType - 20;
            return GetInfo()->GetValue(fAddress, atype, j, 1);
         } else {
            return GetInfo()->GetValue(fObject, fID, j, -1);
         }
      }
   }

   if (fType == 31) {
      TClonesArray* clones = (TClonesArray*) fObject;
      if (subarr) {
         return GetInfo()->GetValueClones(clones, fID, j, len, fOffset);
      }
      return GetInfo()->GetValueClones(clones, fID, j/len, j%len, fOffset);
   } else if (fType == 41) {
      TVirtualCollectionProxy::TPushPop helper(((TBranchElement*) this)->GetCollectionProxy(), fObject);
      if (subarr) {
         return GetInfo()->GetValueSTL(((TBranchElement*) this)->GetCollectionProxy(), fID, j, len, fOffset);
      }
      return GetInfo()->GetValueSTL(((TBranchElement*) this)->GetCollectionProxy(), fID, j/len, j%len, fOffset);
   } else {
      if (GetInfo()) {
         return GetInfo()->GetValue(fObject, fID, j, -1);
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

   if (fBranchCount) {
      Int_t entry = fTree->GetReadEntry();
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
         //return GetInfo()->GetValue(fAddress, atype, j, 1);
         return 0;
      } else if (fType == 41) {    // sub branch of a TClonesArray
         //Int_t atype = fStreamerType;
         //if (atype < 20) atype += 20;
         //return GetInfo()->GetValue(fAddress, atype, j, 1);
         return 0;
      } else if (fType <= 2) {     // branch in split mode
         // FIXME: This should probably be < 60 instead!
         if (fStreamerType > 40 && fStreamerType < 55) {
            //Int_t atype = fStreamerType - 20;
            //return GetInfo()->GetValue(fAddress, atype, j, 1);
            return 0;
         } else {
            //return GetInfo()->GetValue(fObject, fID, j, -1);
            return 0;
         }
      }
   }

   if (fType == 31) {
      return 0;
   } else if (fType == 41) {
      return 0;
   } else {
      //return GetInfo()->GetValue(fObject,fID,j,-1);
      if (!GetInfo() || !fObject) return 0;
      char **val = (char**)(fObject+GetInfo()->GetOffsets()[fID]);
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
      if (!GetInfo()) {
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
         TStreamerInfo* si = GetInfo();
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
      }
      if (!branchClass) {
         Error("InitializeOffsets", "Could not find class for branch: %s", GetName());
         fInitOffsets = kTRUE;
         return;
      }
      // Loop over our sub-branches and compute their offsets.
      for (Int_t subBranchIdx = 0; subBranchIdx < nbranches; ++subBranchIdx) {
         fBranchOffset[subBranchIdx] = 0;
         TBranch* aSubBranch = (TBranch*) fBranches[subBranchIdx];
         // FIXME: Switch to a dynamic cast here.
         if (!aSubBranch->InheritsFrom(TBranchElement::Class())) {
            // -- Skip sub-branches that are not TBranchElements.
            continue;
         }
         TBranchElement* subBranch = (TBranchElement*) aSubBranch;
         TStreamerInfo* sinfo = subBranch->GetInfo();
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
         Int_t localOffset = subBranchElement->GetOffset();
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
         if ((subBranch->fType == 1) || (subBranchElement->IsA() == TStreamerBase::Class()) || ((subBranch->fType == 4) && subBranchElement->IsBase())) {
            // -- Base class sub-branch (1).
            //
            // Note: Our type will not be 1, even though we are
            // a base class branch, if we are not split (see the
            // constructor), or if we are an STL container master
            // branch and a base class branch at the same time.
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

         // FIXME: fprintf(stderr, "mother: %-*s  branch: %s  fType: %d  fID: %d class: %s subBranch: %-16s  subBranch->fType: %d  subBranch->fSTLtype: %d  isBase: %d\n", motherName.Length(), motherName.Data(), GetName(), fType, fID, branchClass->GetName(), subBranch->GetName(), subBranch->fType, subBranch->fSTLtype, (subBranch->fType == 1) || (subBranchElement->IsA() == TStreamerBase::Class()) || ((subBranch->fType ==4) && subBranchElement->IsBase()));

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
               }
            } else {
               // -- Remove the mother name and the dot.
               if (dataName.Length() > motherName.Length()) {
                  dataName.Remove(0, motherName.Length() + 1);
               }
            }
         }
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
            // FIXME: Do we need the other base class tests here?
            if (fType == 1) {
               // -- Parent branch is a base class branch.
               // FIXME: Is using branchElem here the right thing?
               pClass = branchElem->GetClassPointer();
            } else {
               // -- Parent branch is *not* a base class branch.
               // FIXME: This sometimes returns a null pointer.
               pClass = subBranch->GetParentClass();
            }
            if (!pClass) {
               // -- No parent class, fix it.
               // FIXME: This is probably wrong!
               // Assume parent class is our parent branch's clones class or value class.
               if (GetClonesName() && (strlen(GetClonesName()) != 0)) {
                  pClass = gROOT->GetClass(GetClonesName());
                  Warning("InitializeOffsets", "subBranch: '%s' has no parent class!  Assuming parent class is: '%s'.", subBranch->GetName(), pClass->GetName());
               }
               if (fBranchCount && fBranchCount->fCollProxy && fBranchCount->fCollProxy->GetValueClass()) {
                  pClass = fBranchCount->fCollProxy->GetValueClass();
                  Warning("InitializeOffsets", "subBranch: '%s' has no parent class!  Assuming parent class is: '%s'.", subBranch->GetName(), pClass->GetName());
               }
               if (!pClass) {
                  // -- Still no parent class, assume our parent class is our parent branch's class.
                  // FIXME: This is probably wrong!
                  pClass = branchClass;
                  // FIXME: Enable this warning!
                  //Warning("InitializeOffsets", "subBranch: '%s' has no parent class!  Assuming parent class is: '%s'.", subBranch->GetName(), pClass->GetName());
               }
            }
            // Find our offset in our parent class using
            // a lookup by name in the dictionary meta info
            // for our parent class.
            TRealData* rd = pClass->GetRealData(dataName);
            if (rd) {
               // -- Data member exists in the dictionary meta info, get the offset.
               offset = rd->GetThisOffset();
            } else {
               // -- No dictionary meta info for this data member, it must no longer exist.
               // FIXME: Enable this warning!
               //Warning("InitializeOffsets", "Class '%s' version %d checksum %08x has no data member named '%s', assuming offset is zero!", pClass->GetName(), pClass->GetClassVersion(), pClass->GetCheckSum(), dataName.Data());
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
            subBranch->fOffset = offset - localOffset;
         } else {
            // -- Set fBranchOffset for sub-branch.
            Int_t numOfSubSubBranches = subBranch->GetListOfBranches()->GetEntriesFast();
            if (numOfSubSubBranches) {
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
      Int_t entry = fTree->GetReadEntry();
      Int_t first  = fBasketEntry[fReadBasket];
      Int_t last;
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
   // -- Print TBranch parameters.

   Int_t nbranches = fBranches.GetEntriesFast();
   if (strncmp(option,"debugAddress",strlen("debugAddress"))==0) {
      if (strlen(option)==strlen("debugAddress")) {
         Printf("%-24s %-16s %2s %4s %-16s %-16s %8s %8s %s\n",
                "Branch Name", "Streamer Class", "ID", "Type", "Class", "Parent", "pOffset", "fOffset", "fObject");
      }
      TBranchElement *parent = (TBranchElement*) GetMother()->GetSubBranch(this);
      Int_t ind = parent->GetListOfBranches()->IndexOf(this);
      if (strlen(GetName())>24) Printf("%-24s\n%-24s ", GetName(),"");
      else Printf("%-24s ", GetName());
      Printf("%-16s %2d %4d %-16s %-16s %8x %8x %8x\n",
             ((TBranchElement*)this)->GetInfo()->GetName(), GetID(), GetType(),
             GetClassName(), GetParentName(),
             (fBranchOffset&&parent) ? parent->fBranchOffset[ind] : 0,
             GetOffset(), GetObject());
      TObjArray* brl = ((TBranchElement*)this)->GetListOfBranches();
      Int_t nbranches = brl->GetEntriesFast();
      for (Int_t i = 0; i < nbranches; ++i) {
         TBranchElement* subbranch = (TBranchElement*)brl->At(i);
         subbranch->Print("debugAddressSub");
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
         Int_t atype = fStreamerType + TStreamerInfo::kOffsetL;
         if (fStreamerType == TStreamerInfo::kChar) {
            // TStreamerInfo::kOffsetL + TStreamerInfo::kChar is
            // printed as a string and could print weird characters.
            // So we print an unsigned char instead (not perfect, but better).
            atype = TStreamerInfo::kOffsetL + TStreamerInfo::kUChar;
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
         if (GetInfo()) {
            GetInfo()->PrintValue(GetName(), fAddress, atype, n, lenmax);
         }
         return;
      } else if (fType <= 2) {
         // Branch in split mode.
         // FIXME: This should probably be < 60 instead.
         if ((fStreamerType > 40) && (fStreamerType < 55)) {
            Int_t atype = fStreamerType - 20;
            TBranchElement* counterElement = (TBranchElement*) fBranchCount;
            Int_t n = (Int_t) counterElement->GetValue(0, 0);
            if (GetInfo()) {
               GetInfo()->PrintValue(GetName(), fAddress, atype, n, lenmax);
            }
         } else {
            if (GetInfo()) {
               GetInfo()->PrintValue(GetName(), fObject, fID, -1, lenmax);
            }
         }
         return;
      }
   } else if (fType == 3) {
      printf(" %-15s = %d\n", GetName(), fNdata);
   } else if (fType == 31) {
      TClonesArray* clones = (TClonesArray*) fObject;
      if (GetInfo()) {
         GetInfo()->PrintValueClones(GetName(), clones, fID, fOffset, lenmax);
      }
   } else if (fType == 41) {
      TVirtualCollectionProxy::TPushPop helper(((TBranchElement*) this)->GetCollectionProxy(), fObject);
      if (GetInfo()) {
         GetInfo()->PrintValueSTL(GetName(), ((TBranchElement*) this)->GetCollectionProxy(), fID, fOffset, lenmax);
      }
   } else {
      if (GetInfo()) {
         GetInfo()->PrintValue(GetName(), fObject, fID, -1, lenmax);
      }
   }
}

//______________________________________________________________________________
void TBranchElement::ReadLeaves(TBuffer& b)
{
   // -- Read leaves into i/o buffers for this branch.

   ValidateAddress();

   if (fTree->GetMakeClass()) {
      if (fType == 3 || fType == 4) {
         // Top level branch of a TClonesArray.
         Int_t *n = (Int_t*) fAddress;
         b >> n[0];
         if ((n[0] < 0) || (n[0] > fMaximum)) {
            if (IsMissingCollection()) {
               n[0] = 0;
               b.SetBufferOffset(b.Length() - sizeof(n));
            } else {
               Error("ReadLeaves", "Incorrect size read for the container in %s\nThe size read is %d when the maximum is %d\nThe size is reset to 0 for this entry (%d)", GetName(), n, fMaximum, GetReadEntry());
               n[0] = 0;
            }
         }
         fNdata = n[0];
         if ( fType == 4)   {
            Int_t i, nbranches = fBranches.GetEntriesFast();
            switch(fSTLtype) {
               case TClassEdit::kSet:
               case TClassEdit::kMultiSet:
               case TClassEdit::kMap:
               case TClassEdit::kMultiMap:
                  for (i=0;i<nbranches;i++) {
                     TBranch *branch = (TBranch*)fBranches[i];
                     Int_t    nb = branch->GetEntry(GetReadEntry(), 1);
                     if (nb < 0) break;
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
               Double_t *xx = (Double_t*) fAddress;
               Float_t afloat;
               for (Int_t ii=0;ii<n;ii++) {
                  b >> afloat; xx[ii] = Double_t(afloat);
               }
               break;
            }
         }
         return;
      } else if (fType <= 2) {     // branch in split mode
         // FIXME: This should probably be < 60 instead.
         if (fStreamerType > 40 && fStreamerType < 55) {
            Int_t atype = fStreamerType - 40;
            Int_t n = (Int_t)fBranchCount->GetValue(0,0);
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
                  Double_t *xx = (Double_t*)fAddress;
                  Float_t afloat;
                  for (Int_t ii=0;ii<n;ii++) {
                     b>> afloat; xx[ii] = Double_t(afloat);
                  }
                  break;
               }
            }
         } else {
            fNdata = 1;
            if (fAddress) {
               GetInfo()->ReadBuffer(b, (char**) &fObject, fID);
            } else {
               fNdata = 0;
            }
         }
         return;
      }
   }

   // If not a TClonesArray or STL container master branch
   // or sub-branch and branch inherits from tobject,
   // then register with the buffer so that pointers are
   // handled properly.
   // FIXME: Does this mean that pointers to objects which
   //        do not inherit from tobject are not handled correctly?

   if ((fType <= 2) && TestBit(kBranchObject)) {
      b.MapObject((TObject*) fObject);
   }

   if (fType == 4) {
      // STL container master branch (has only the number of elements).
      Int_t n;
      b >> n;
      if ((n < 0) || (n > fMaximum)) {
         if (IsMissingCollection()) {
            n = 0;
            b.SetBufferOffset(b.Length()-sizeof(n));
         } else {
            Error("ReadLeaves", "Incorrect size read for the container in %s\n\tThe size read is %d while the maximum is %d\n\tThe size is reset to 0 for this entry (%d)", GetName(), n, fMaximum, GetReadEntry());
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
      // FIXME: What to do if this fails (which it will if fNdata is unreasonable)?
      void* env = proxy->Allocate(fNdata, true);
      Int_t i, nbranches = fBranches.GetEntriesFast();
      switch (fSTLtype) {
         case TClassEdit::kSet:
         case TClassEdit::kMultiSet:
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
            for (i = 0; i < nbranches; ++i) {
               TBranch* branch = (TBranch*) fBranches[i];
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
      proxy->Commit(env);
   } else if (fType == 41) {
      // STL container sub-branch (contains the elements).
      fNdata = fBranchCount->GetNdata();
      if (!fObject) {
         return;
      }
      TVirtualCollectionProxy::TPushPop helper(GetCollectionProxy(), fObject);
      GetInfo()->ReadBufferSTL(b, GetCollectionProxy(), fNdata, fID, fOffset);
   } else if (fType == 3) {
      // TClonesArray master branch (has only the number of elements).
      Int_t n;
      b >> n;
      if ((n < 0) || (n > fMaximum)) {
         if (IsMissingCollection()) {
            n = 0;
            b.SetBufferOffset(b.Length()-sizeof(n));
         } else {
            Error("ReadLeaves", "Incorrect size read for the container in %s\n\tThe size read is %d while the maximum is %d\n\tThe size is reset to 0 for this entry (%d)", GetName(), n, fMaximum, GetReadEntry());
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
   } else if (fType == 31) {
      // TClonesArray sub-branch (contains the elements).
      fNdata = fBranchCount->GetNdata();
      TClonesArray* clones = (TClonesArray*) fObject;
      if (!clones) {
         return;
      }
      if (clones->IsZombie()) {
         return;
      }
      GetInfo()->ReadBufferClones(b, clones, fNdata, fID, fOffset);
   } else if (fType <= 2) {
      // split-class branch, base class branch, data member branch, or top-level branch.
      if (fBranchCount) {
         fNdata = (Int_t) fBranchCount->GetValue(0, 0);
      } else {
         fNdata = 1;
      }
      if (!GetInfo()) {
         return;
      }
      GetInfo()->ReadBuffer(b, (char**) &fObject, fID);
      if (fStreamerType == TStreamerInfo::kCounter) {
         fNdata = (Int_t) GetValue(0, 0);
      }
   }
}

//______________________________________________________________________________
void TBranchElement::ReleaseObject()
{
   // -- Delete any object we may have allocated on a previous call to SetAddress.

   // Make sure kDeleteObject and fObject are valid before proceeding.
   // Note: We are *not* allowed to call ValidateAddress() because it
   //       may call SetAddress(), which may call us resulting in an
   //       infinite loop.
   //ValidateAddress();

   return; // FIXME: Disable the deletion of object owned by the TTree until we add a missing interface.

   if (fID < 0) {
      // -- We are a top-level branch.
      if (fAddress && (*((char**) fAddress) != fObject)) {
         // The semantics of fAddress and fObject are violated.
         // Assume the user changed the pointer on us.
         if (TestBit(kDeleteObject)) {
            Warning("ReleaseObject", "branch: %s, You have overwritten the pointer to an object which I owned!", GetName());
            Warning("ReleaseObject", "This is a memory leak.  Please use SetAddress() to change the pointer instead.");
            ResetBit(kDeleteObject);
         }
      }
   }

   // Delete any object we may have allocated during a call to SetAddress.
   if (fObject && TestBit(kDeleteObject)) {
      ResetBit(kDeleteObject);
      if (fType == 3) {
         // -- We are a TClonesArray master branch.
         TClonesArray::Class()->Destructor(fObject);
         fObject = 0;
         if ((fStreamerType == TStreamerInfo::kObjectp) ||
             (fStreamerType == TStreamerInfo::kObjectP)) {
            // -- We are a pointer to a TClonesArray.
            // We must zero the pointer in the object.
            *((char**) fAddress) = 0;
         }
      } else if (fType == 4) {
         // -- We are an STL container master branch.
         TVirtualCollectionProxy* proxy = GetCollectionProxy();
         if (!proxy) {
            Warning("ResetAddress", "Cannot delete allocated STL container because I do not have a proxy!  branch: %s", GetName());
            fObject = 0;
         } else {
            proxy->Destructor(fObject);
            fObject = 0;
         }
         if (fStreamerType == TStreamerInfo::kSTLp) {
            // -- We are a pointer to an STL container.
            // We must zero the pointer in the object.
            *((char**) fAddress) = 0;
         }
      } else {
         // We are *not* a TClonesArray master branch and we are *not* an STL container master branch.
         TClass* cl = fBranchClass.GetClass();
         if (!cl) {
            Warning("ResetAddress", "Cannot delete allocated object because I cannot instantiate a TClass object for its class!  branch: '%s' class: '%s'", GetName(), fBranchClass.GetClassName());
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

   // FIXME: Why do we do this?
   //if (fBranchClass.GetClass()) {
   //   // FIXME: Must protect against optimization here!
   //   // FIXME: This should probably turn into a forced version of GetInfo().
   //   fInfo = fBranchClass.GetClass()->GetStreamerInfo(fClassVersion);
   //   // FIXME: fInit = ???
   //}
   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i) {
      TBranch* branch = (TBranch*) fBranches[i];
      branch->Reset(option);
   }
   TBranch::Reset(option);
}

//______________________________________________________________________________
void TBranchElement::ResetAddress()
{
   // -- Reset the branch user i/o buffer address.

   // Make sure the user did not change the object pointer first.
   ValidateAddress();

   for (Int_t i = 0; i < fNleaves; ++i) {
      TLeaf* leaf = (TLeaf*) fLeaves.UncheckedAt(i);
      leaf->SetAddress(0);
   }

   // Note: We *must* do the sub-branches first, otherwise
   //       we may delete the object containing the sub-branches
   //       before giving them a chance to cleanup.
   Int_t nbranches = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nbranches; ++i)  {
      TBranch* br = (TBranch*) fBranches[i];
      br->ResetAddress();
   }

   //
   // SetAddress may have allocated an object.
   //

   ReleaseObject();

   fAddress = 0;
   fObject = 0;
}

//______________________________________________________________________________
void TBranchElement::ResetDeleteObject()
{
   // -- Clear the kDeleteObject flag (clones should not delete shared i/o buffers).

   ResetBit(kDeleteObject);
   Int_t nb = fBranches.GetEntriesFast();
   for (Int_t i = 0; i < nb; ++i)  {
      TBranch* br = (TBranch*) fBranches[i];
      // FIXME: This is a tail recursion.
      // FIXME: Change this to a dynamic cast attempt.
      if (br->InheritsFrom("TBranchElement")) {
         ((TBranchElement*) br)->ResetDeleteObject();
      }
   }
}

//______________________________________________________________________________
void TBranchElement::SetAddress(void* add)
{
   // -- Set user i/o buffer address of this branch.

   //
   //  Don't bother if we are disabled.
   //
   //  FIXME:  What if we are enabled later?

   if (TestBit(kDoNotProcess)) {
      return;
   }

   //
   //  FIXME: When would this happen?
   //

   if (fType < 0) {
      return;
   }

   //
   //  Special case when called from code generated by TTree::MakeClass.
   //

   if (Long_t(add) == -1) {
      // FIXME: Do we have to release an object here?
      fAddress = (char*) -1;
      fObject = (char*) -1;
      ResetBit(kDeleteObject);
      return;
   }

   //
   //  Reset last read entry number, we have a new i/o buffer now.
   //

   fReadEntry = -1;

   //
   // Make sure our branch class is instantiated.
   //

   TClass* clOfBranch = fBranchClass.GetClass();

   //
   // Try to build the streamer info.
   //

   GetInfo();

   // FIXME: Warn about failure to get the streamer info here?

   //
   // We may have allocated an object last time we were called.
   //

   ReleaseObject();

   //
   //  Remember the pointer to the pointer to our object.
   //

   fAddress = (char*) add;
   fObject = 0;
   ResetBit(kDeleteObject);

   //
   //  Do special stuff if we got called from a MakeClass class.
   //  Allow sub-branches to have independently set addresses.
   //

   if (fTree->GetMakeClass()) {
      if (fID > -1) {
         // We are *not* a top-level branch.
         if (!GetInfo()) {
            // No streamer info, give up.
            // FIXME: We should have an error message here.
            fObject = fAddress;
         } else {
            // Compensate for the fact that the i/o routines
            // will add the streamer offset to the address.
            fObject = fAddress - GetInfo()->GetOffsets()[fID];
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
      TClass* clm = gROOT->GetClass(fClonesName.Data());
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
               Warning("SetAddress", "The type of %s was changed from TClonesArray to %s but the content do not match (was %s)!", GetName(), newType->GetName(), fClonesName.Data());
            }
         } else {
            Warning("SetAddress", "The type of the %s was changed from TClonesArray to %s but we do not have a TVirtualCollectionProxy for that container type!", GetName(), newType->GetName());
         }
         if (matched) {
            // Change from 3/31 to 4/41
            SetType(4);
            SwitchContainer(GetListOfBranches());
            // Set the proxy.
            fSTLtype = TMath::Abs(TClassEdit::IsSTLCont(newType->GetName()));
            fCollProxy = newType->GetCollectionProxy()->Generate();
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
            fCollProxy = newType->GetCollectionProxy()->Generate();
         } else {
            // The new collection and the old collection are not compatible,
            // we cannot use the new collection to read the data.
            // Actually we could check if the new collection is a
            // compatible ROOT collection.
            if ((newType == TClonesArray::Class()) && (oldProxy->GetValueClass() && !oldProxy->HasPointers() && oldProxy->GetValueClass()->InheritsFrom(TObject::Class()))) {
               // We cannot insure that the TClonesArray is set for the
               // proper class (oldProxy->GetValueClass()), so we assume that
               // the transformation was done properly by the class designer.

               // Change from 4/41 to 3/31
               SetType(3);
               SwitchContainer(GetListOfBranches());
               // Reset the proxy.
               fSTLtype = kNone;
               switch(fStreamerType) {
                  case TStreamerInfo::kAny:
                  case TStreamerInfo::kSTL:
                     fStreamerType = TStreamerInfo::kObject;
                     break;
                  case TStreamerInfo::kAnyp:
                  case TStreamerInfo::kSTLp:
                     fStreamerType = TStreamerInfo::kObjectp;
                     break;
                  case TStreamerInfo::kAnyP:
                     fStreamerType = TStreamerInfo::kObjectP;
                     break;
               }
               fClonesName = oldProxy->GetValueClass()->GetName();
               delete fCollProxy;
               fCollProxy = 0;
               TClass* clm = gROOT->GetClass(fClonesName);
               if (clm) {
                  clm->BuildRealData(); //just in case clm derives from an abstract class
                  clm->GetStreamerInfo();
               }
            } else {
               // FIXME: We must maintain fObject here as well.
               fAddress = 0;
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
         if (fStreamerType == TStreamerInfo::kObject) {
            // -- We are *not* a top-level branch and we are *not* a pointer to a TClonesArray.
            // Case of an embedded TClonesArray.
            fObject = fAddress;
            // Check if it has already been properly built.
            TClonesArray* clones = (TClonesArray*) fObject;
            if (!clones->GetClass()) {
               new(fObject) TClonesArray(fClonesName.Data());
            }
         } else {
            // -- We are either a top-level branch or we are a subbranch which is a pointer to a TClonesArray.
            // Streamer type should be -1 (for a top-level branch) or kObject(p|P) here.
            if ((fStreamerType != -1) &&
                (fStreamerType != TStreamerInfo::kObjectp) &&
                (fStreamerType != TStreamerInfo::kObjectP)) {
               Error("SetAddress", "TClonesArray with fStreamerType: %d", fStreamerType);
            } else if (fStreamerType == -1) {
               // -- We are a top-level branch.
               TClonesArray** pp = (TClonesArray**) fAddress;
               if (!*pp) {
                  SetBit(kDeleteObject);
                  *pp = new TClonesArray(fClonesName.Data());
               }
               fObject = (char*) *pp;
            } else {
               // -- We are a pointer to a TClonesArray.
               // Note: We do this so that the default constructor,
               //       or the i/o constructor can be lazy.
               TClonesArray** pp = (TClonesArray**) fAddress;
               if (!*pp) {
                  SetBit(kDeleteObject);
                  *pp = new TClonesArray(fClonesName.Data());
               }
               fObject = (char*) *pp;
            }
         }
      } else {
         // -- We have been given a zero address, allocate for top-level only.
         if (fStreamerType == TStreamerInfo::kObject) {
            // -- We are *not* a top-level branch and we are *not* a pointer to a TClonesArray.
            // Case of an embedded TClonesArray.
            Error("SetAddress", "Embedded TClonesArray given a zero address for branch '%s'", GetName());
         } else {
            // -- We are either a top-level branch or we are a subbranch which is a pointer to a TClonesArray.
            // Streamer type should be -1 (for a top-level branch) or kObject(p|P) here.
            if ((fStreamerType != -1) &&
                (fStreamerType != TStreamerInfo::kObjectp) &&
                (fStreamerType != TStreamerInfo::kObjectP)) {
               Error("SetAddress", "TClonesArray with fStreamerType: %d", fStreamerType);
            } else if (fStreamerType == -1) {
               // -- We are a top-level branch.
               // FIXME: Consider making a zero address not allocate.
               SetBit(kDeleteObject);
               fObject = (char*) new TClonesArray(fClonesName.Data());
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
         if ((fStreamerType == TStreamerInfo::kObject) ||
             (fStreamerType == TStreamerInfo::kAny) ||
             (fStreamerType == TStreamerInfo::kSTL)) {
            // We are *not* a top-level branch and we are *not* a pointer to an STL container.
            // Case of an embedded STL container.
            fObject = fAddress;
         } else {
            // We are either a top-level branch or subbranch which is a pointer to an STL container.
            // Streamer type should be -1 (for a top-level branch) or kSTLp here.
            if ((fStreamerType != -1) && (fStreamerType != TStreamerInfo::kSTLp)) {
               Error("SetAddress", "STL container with fStreamerType: %d", fStreamerType);
            } else if (fStreamerType == -1) {
               // -- We are a top-level branch.
               void** pp = (void**) fAddress;
               if (!*pp) {
                  SetBit(kDeleteObject);
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
                  SetBit(kDeleteObject);
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
         if ((fStreamerType == TStreamerInfo::kObject) ||
             (fStreamerType == TStreamerInfo::kAny) ||
             (fStreamerType == TStreamerInfo::kSTL)) {
            // We are *not* a top-level branch and we are *not* a pointer to an STL container.
            // Case of an embedded STL container.
            Error("SetAddress", "Embedded STL container given a zero address for branch '%s'", GetName());
         } else {
            // We are either a top-level branch or sub-branch which is a pointer to an STL container.
            // Streamer type should be -1 (for a top-level branch) or kSTLp here.
            if ((fStreamerType != -1) && (fStreamerType != TStreamerInfo::kSTLp)) {
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
            SetBit(kDeleteObject);
            // FIXME:
            // If we end up creating a histogram here, we might not
            // want to add it to the current directory and
            // take ownership instead.  If we let it get added
            // to the directory, then it could be deleted by
            // the directory without us getting notified, and
            // then we would try to delete it a second time.
            // So we might want to call TH1::AddDirectory(kFALSE);
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

   if (!GetInfo()) {
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
      TBranch* abranch = (TBranch*) fBranches[i];
      // FIXME: This is a tail recursion!
      abranch->SetAddress(fObject + fBranchOffset[i]);
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

   if (TestBit(kDoNotProcess)) {
      // -- Do nothing if we have been told not to.
      return;
   }

   TBranchElement* mother = (TBranchElement*) GetMother();
   TClass* cl = gROOT->GetClass(mother->GetClassName());

   // FIXME: Should this go after the mother and cl test?
   if (GetInfo() && GetInfo()->GetOffsets()) {
      // If our streamer info has already been compiled,
      // then we must try to deal with schema evolution here.
      // FIXME: We must not optimize here or InitializeOffsets will crash!
      GetInfo()->BuildOld();
   }

   if (!mother || !cl) {
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
      TBranchElement::Class()->ReadBuffer(R__b, this);
      fParentClass.SetName(fParentName);
      fBranchClass.SetName(fClassName);

      // The fAddress and fObject data members are not persistent,
      // therefore we do not own anything.
      ResetBit(kDeleteObject);

      // Fixup a case where the TLeafElement was missing
      if ((fType == 0) && (fLeaves.GetEntriesFast() == 0)) {
         TLeaf* leaf = new TLeafElement(GetTitle(), fID, fStreamerType);
         leaf->SetTitle(GetTitle());
         leaf->SetBranch(this);
         fNleaves = 1;
         fLeaves.Add(leaf);
         fTree->GetListOfLeaves()->Add(leaf);
      }
   } else {
      TDirectory *dirsav = fDirectory;
      fDirectory = 0;  // to avoid recursive calls

      // FIXME: Should we clear the kDeleteObject bit before writing?
      //        If we did we would have to remember to old value and
      //        put it back, we wouldn't want to forget that we owned
      //        something just because we got written to disk.
      TBranchElement::Class()->WriteBuffer(R__b, this);

      // make sure that all TStreamerInfo objects referenced by
      // this class are written to the file
      if (GetInfo()) {
         GetInfo()->ForceWriteInfo((TFile *)R__b.GetParent(), kTRUE);
      }

      // if branch is in a separate file save this branch
      // as an independent key
      if (!dirsav) {
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
         TDirectory* cursav = gDirectory;
         dirsav->cd();
         Write();
         cursav->cd();
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

   Bool_t optim = TStreamerInfo::CanOptimize();
   if (splitlevel > 0) {
      TStreamerInfo::Optimize(kFALSE);
   }
   TStreamerInfo* sinfo = fTree->BuildStreamerInfo(cl);
   TStreamerInfo::Optimize(optim);

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
      // FIXME: Why?  We certainly could by switching to the value class.
      // FIXME: Partial Answer: Only the branch element constructor can split STL containers.
      return 1;
   }

   for (Int_t elemID = 0; elemID < ndata; ++elemID) {
      // -- Loop over all the streamer elements and create sub-branches as needed.
      TStreamerElement* elem = (TStreamerElement*) elems[elemID];
      Int_t offset = elem->GetOffset();
      // FIXME: An STL container as a base class gets TStreamerSTL as its class, so this test is not enough.
      // FIXME: See InitializeOffsets() for the proper test.
      if (elem->IsA() == TStreamerBase::Class()) {
         // -- This is a base class of cl.
         TClass* clOfBase = gROOT->GetClass(elem->GetName());
         if ((clOfBase->Property() & kIsAbstract) && cl->InheritsFrom("TCollection")) {
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
            Int_t unroll = Unroll(name, clParent, clOfBase, ptr + offset, basketsize, splitlevel-1, btype);
            if (unroll < 0) {
               // FIXME: We could not split because we are abstract, should we be doing this?
               if (strlen(name)) {
                  branchname.Form("%s.%s", name, elem->GetFullName());
               } else {
                  branchname.Form("%s", elem->GetFullName());
               }
               TBranchElement* branch = new TBranchElement(branchname, sinfo, elemID, 0, basketsize, 0, btype);
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
               TBranchElement* branch = new TBranchElement(name, sinfo, elemID, ptr + offset, basketsize, splitlevel, btype);
               // Then reset it to the proper name.
               branch->SetName(branchname);
               branch->SetTitle(branchname);
               branch->SetParentClass(clParent);
               fBranches.Add(branch);
            } else {
               branchname.Form("%s", elem->GetFullName());
               TBranchElement* branch = new TBranchElement(branchname, sinfo, elemID, ptr + offset, basketsize, splitlevel, btype);
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
            TClass* elemClass = gROOT->GetClass(elem->GetTypeName());
            if (elemClass->Property() & kIsAbstract) {
               return -1;
            }
            if (elem->CannotSplit()) {
               // We are not splitting.
               TBranchElement* branch = new TBranchElement(branchname, sinfo, elemID, ptr + offset, basketsize, 0, btype);
               branch->SetParentClass(clParent);
               fBranches.Add(branch);
            } else if (elemClass->InheritsFrom(TClonesArray::Class())) {
               // Splitting something derived from TClonesArray.
               Int_t subSplitlevel = splitlevel-1;
               if (btype == 31 || btype == 41 || elem->CannotSplit()) {
                  // -- We split the sub-branches of a TClonesArray or an STL container only once.
                  subSplitlevel = 0;
               }
               TBranchElement* branch = new TBranchElement(branchname, sinfo, elemID, ptr + offset, basketsize, subSplitlevel, btype);
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
               Int_t unroll = Unroll(branchname, clParent, elemClass, ptr + offset, basketsize, splitlevel-1, btype);
               if (unroll < 0) {
                  // FIXME: We could not split because we are abstract, should we be doing this?
                  TBranchElement* branch = new TBranchElement(branchname, sinfo, elemID, ptr + offset, basketsize, 0, btype);
                  branch->SetParentClass(clParent);
                  fBranches.Add(branch);
               }
            }
         } else if ((elem->IsA() == TStreamerSTL::Class()) && !elem->IsaPointer()) {
            // -- We have an STL container.
            // FIXME: What if splitlevel == 0 here?
            Int_t subSplitlevel = splitlevel - 1;
            if ((btype == 31) || (btype == 41) || elem->CannotSplit()) {
               // -- We split the sub-branches of a TClonesArray or an STL container only once.
               subSplitlevel = 0;
            }
            TBranchElement* branch = new TBranchElement(branchname, sinfo, elemID, ptr + offset, basketsize, subSplitlevel, btype);
            branch->SetParentClass(clParent);
            fBranches.Add(branch);
         } else if (((btype != 31) && (btype != 41)) && ptr && ((elem->GetClassPointer() == TClonesArray::Class()) || ((elem->IsA() == TStreamerSTL::Class()) && !elem->CannotSplit()))) {
            // -- We have a TClonesArray.
            // FIXME: We could get a ptr to a TClonesArray here by mistake.
            // FIXME: What if splitlevel == 0 here?
            // Note: ptr may be null in case of a TClonesArray inside another
            //       TClonesArray or STL container, see the else clause.
            TBranchElement* branch = new TBranchElement(branchname, sinfo, elemID, ptr + offset, basketsize, splitlevel-1, btype);
            branch->SetParentClass(clParent);
            fBranches.Add(branch);
         } else {
            // -- We are not going to split this element any farther.
            TBranchElement* branch = new TBranchElement(branchname, sinfo, elemID, 0, basketsize, 0, btype);
            branch->SetType(btype);
            branch->SetParentClass(clParent);
            fBranches.Add(branch);
         }
      }
   }

   return 1;
}

//______________________________________________________________________________
void TBranchElement::ValidateAddress() const
{
   // -- Check to see if the user changed the object pointer without telling us.

   if (fID < 0) {
      // -- We are a top-level branch.
      if (fAddress && (*((char**) fAddress) != fObject)) {
         // -- The semantics of fAddress and fObject are violated.
         // Assume the user changed the pointer on us.
         // Note: The cast is here because we want to be able to
         //       be called from the constant get functions.


	 // FIXME: Disable the check/warning TTree until we add a missing interface.
         if (false && TestBit(kDeleteObject)) {
            Warning("ValidateAddress", "branch: %s, You have overwritten the pointer to an object which I owned!", GetName());
            Warning("ValidateAddress", "This is a memory leak.  Please use SetAddress() to change the pointer instead.");
            const_cast<TBranchElement*>(this)->ResetBit(kDeleteObject);
         }
         const_cast<TBranchElement*>(this)->SetAddress(fAddress);
      }
   }
}

